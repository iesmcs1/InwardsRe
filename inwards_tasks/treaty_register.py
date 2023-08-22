from typing import Union
import pandas as pd
import numpy as np
import os
import re

try:
    from .utils import append_eof_to_txt_file, import_run_off_rates
except ImportError:
    from utils import append_eof_to_txt_file, import_run_off_rates

# {current: solvency}
COLUMN_NAME_MAPPING = {'Broker': 'Broker Code',
                       'Freq Accts': 'Prem Freq',
                       'Group Bus': 'Related',
                       'Pipe': 'Pipeline',
                       'Status Code': 'Status',
                       '100% EPI': 'Orig EPI',
                       'Signed EPI': 'Our Orig EPI',
                       'Signed EPI Euro': 'Our EPI Euro',
                       'Est Total Costs %': 'Est Total Cost %',
                       'UWLr%': 'ULR %',
                       'Commission': 'Comm %',
                       'Prof Comm': 'Profit Comm %',
                       'Broker Fees': 'Brokerage %',
                       'Mgmnt Exp': 'Mgmnt Exp. %',
                       'Ovrd Comm': 'O/R Comm %',
                       'Min %': 'Min Sliding Scale',
                       'LR for Min %': 'LR for Min',
                       'Max %': 'Max Sliding Scale',
                       'LR for Max %': 'LR for Max',
                       'Retention': 'Deductible',
                       'Protected Share %': 'Protected Share',
                       '100% Full Limit': '100% UW Limit',
                       'xs': 'Excess',
                       'Aggded': 'Agg Deductible',
                       }


class Register:    
    def __init__(self,
                 filepath: str,
                 report_date: Union[None, str] = None,
                 adjust_register: bool = False,
                 oldest_uwy: Union[None, int] = None,
                 timestamp: bool = False) -> None:
        """Class to import and process Treaty Register files.

        Parameters
        ----------
        filepath : str
            file path to register file. Can be .xlsx or .txt.
        report_date : Union[None, str], optional
            parameter needs to be in format 'DD/MM/YYYY, by default None
            If None is passed, then report_date will be infered from filepath.
        filepath : Union[None, str], optional
            file path to register file. Can be .xlsx or .txt., by default None
        adjust_register : bool, optional
            if it is to apply adjustment calculation to register,
            by default False
        oldest_uwy : Union[None, int], optional
            parameter used to filter dataframe to include only significant
            underwriting years, by default None
        
        timestamp : bool, optional
            if to include column with timestamp from report_date,
            by default False
        """        
        self.filepath = filepath
        self._adjusted_register = None
        self._solvency_register = None
        
        if oldest_uwy:
            self.oldest_uwy = int(oldest_uwy)
        else:
            self.oldest_uwy = oldest_uwy
        
        # Initialize self._report_date variable as None
        self._report_date = None

        # Set report_date using function
        self.set_report_date(date=report_date)

        # Flag to Timestamp Register or not
        self.timestamp = timestamp

        self._register = None
        
        # Call to update for self._register
        self._import_register()

        if adjust_register:
            # It Treaty register is being adjusted, then we don't calculate
            # run-off
            self.adjust_treaty_register()
            self.generate_solvency_register()
        
        
    
    @property
    def report_date(self) -> pd.Timestamp:
        return self._report_date
    
    @report_date.setter
    def report_date(self, date: Union[str, pd.Timestamp]) -> None:
        self._report_date = pd.to_datetime(date, dayfirst=True)
    
    @property
    def register(self) -> pd.DataFrame:
        return self._register
    
    @register.setter
    def register(self, df: pd.DataFrame) -> None:
        # Remove excess Lines where 2nd column is NaN
        df = df.loc[~df[1].isna()].copy()
        
        # Reset index so that I use first row as header
        df.reset_index(drop=True, inplace=True)
        df.columns = df.iloc[0]

        # No need for row with column headers
        df.drop(df.index[0], inplace=True)

        # Add processed register to model
        self._register = self._initial_processing(df)
    
    @property
    def adjusted_register(self) -> pd.DataFrame:
        if self._adjusted_register is None:
            # If Register has not been adjusted yet, run function
            # and return adjusted register.
            return self.adjust_treaty_register()
        else:
            return self._adjusted_register
    
    @adjusted_register.setter
    def adjusted_register(self, df: pd.DataFrame) -> None:
        self._adjusted_register = df
        return None
    
    @property
    def solvency_register(self) -> pd.DataFrame:
        if self._solvency_register is None:
            # If Solvency Register has not been created yet, run function
            # and return adjusted register.
            return self.generate_solvency_register()
        else:
            return self._solvency_register

    @solvency_register.setter
    def solvency_register(self, df: pd.DataFrame) -> None:
        self._solvency_register = df
        return None

    def set_report_date(self, date: Union[str, pd.Timestamp]) -> None:
        if date is None:
            # If user doesn't pass date, then infer from filepath
            self.report_date = self.get_report_date_from_file_name()
            pass
        else:
            # If user pass report_date string, then use that string
            self.report_date = date
        
        return None
    
    def get_report_date_from_file_name(self) -> pd.Timestamp:
        p = re.compile(r"""
            \d{4}  # 4 digits representing year
            \s     # white space
            \w+    # word representing the month
            \s     # white space
            \d{2}    # 2-digit day, followed by date position (st, nd, th)
        """, re.VERBOSE)

        # Full string with pattern e.g.: 2021 June 30
        if self.filepath.endswith('.xlsx'):
            # If Excel file, then probable that it's in Treaty
            # Register file name format
            try:
                t = p.search(self.filepath).group()
                return pd.to_datetime(t, format="%Y %B %d")
            except AttributeError:
                # if pattern above is not found in file name, then ignore
                err = "Can't infer report date from filename. "
                err += "Please add report_date parameter to class call."
                raise ValueError(err)
        elif self.filepath.endswith('.txt'):
            try:
                # SII text file always have the same format:
                # 202204061100_2022Q1_IRE_TREATYREG_04.txt'
                # yyyymmddHHMM_yyyyQX_IRE_xxxxxxxxxxxx.txt
                # What is needed is the date at quarter end
                p = re.compile(r"\d{8}\d{4}_(\d{4}Q\d{1})")
                
                # p.search returns yyyyQX
                t = p.search(self.filepath).group(1)
                # slicing first 4 digits to return year, then get only last digit
                # to find out the quarter.
                # Multiply by 3 to get the quarter end Month.
                t_formatted = t[:4] + f'-{3 * int(t[-1])}'

                # Using year and month returns date in the beginning of month
                # Add MonthEnd to return date at end of month.
                return (pd.to_datetime(t_formatted, format="%Y-%m")
                        + pd.tseries.offsets.MonthEnd(0))
            except AttributeError:
                return None
        
    def _import_register(self) -> None:
        df = None
        if self.filepath.endswith('.xlsx'):
            df = pd.read_excel(self.filepath, header=None)
        elif self.filepath.endswith('.txt'):
            df = pd.read_csv(self.filepath, header=None, sep='\t')
        else:
            err_msg = "File extension not supported. Needs to be .xlsx or .txt"
            raise ValueError(err_msg)

        self.register = df

        return None

    def _initial_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process any Register into correct format.
        
        Parameters
        ----------
        """
        df = df.copy()

        cols_to_strip = ['Comp', 'Seq', 'Cedant', 'Short', 'Curr', 'UPR']
        for col in cols_to_strip:
            # Remove blank spaces
            df[col] = df[col].str.strip()

        # Date columns to datetime format
        date_cols = ['Pd Beg',
                     'Pd End',
                     'Inception',
                     'Profit Comm Date',
                     'Last Booked Account',
                     'Last Update']
        for col in date_cols:
            try:
                # convert columns to datetime
                df[col] = pd.to_datetime(df[col],
                                        dayfirst=True,
                                        errors='coerce')
            except KeyError:
                # When user deletes columns to load file faster, columns
                # will disappear, and this should not raise an error.
                pass
        
        # Dealing with any Date errors. Entry 02068B3 0508 has Inception
        # of year 6008
        df.loc[df['Inception'].isna(), 'Inception'] = df['Pd Beg']

        df['UW Yr'] = df['UW Yr'].astype(int)
        df['UWY_str'] = df['UW Yr']
        df['UWY_str'] = df['UWY_str'].apply(lambda x: '{:0>2}'.format(x))

        # Correct Currencies
        df.loc[df['Curr'] == 'PEI', 'Curr'] = 'PEN'
        df.loc[df['Curr'] == 'MXP', 'Curr'] = 'MXN'
        df.loc[df['Curr'] == 'RUR', 'Curr'] = 'RUB'

        # Convert Signed lines to percent
        # Errors were happening if I didn't conver to float
        df['Signed %'] = df['Signed %'].astype(np.float64)
        df['Signed %'] = df['Signed %'] / 100

        # Convert columns to int64
        cols_to_int = ['100% EPI',  # most recent format
                       'Orig EPI',  # IM format
                       'EPI is Rev EPI or EPI',
                       'Ledger TPE',
                       'TPE',
                       'Our TPE',
                       'Ledger Our TPE',
                       'Limit']
        
        for col in cols_to_int:
            try:
                df[col] = df[col].astype(np.int64)
            except ValueError:
                df[col] = df[col].astype(np.float64)
                df[col] = df[col].astype(np.int64)
            except KeyError:
                # if column not found, next
                pass
        
        # Manually calculating Our TPE due to errors in the past
        # Errors were happening if I didn't conver to int64
        df['Our TPE'] = df['TPE'].astype(np.int64) * df['Signed %']

        # add 1 day to dates from 1900 to avoid upload error in Solvency II
        df.loc[
            df['Profit Comm Date'].dt.year == 1900,
            'Profit Comm Date'
        ] = pd.to_datetime('02/01/1900', dayfirst=True)

        # If Comp column starts with >, it means it's Ceded Register
        is_ceded_register = df['Comp'].str.startswith('>').all()

        if is_ceded_register:
            # Remove blank spaces when loading ceded register

            # There is one case with a space. RT16 08 -> RT1608
            df['Seq'].str.replace(" ", "")

            # Add leading zero to for Reference column
            df['Ceded Ref'] = df['Seq'] + df['UWY_str']
        else:
            # Make Comp as str in correct format
            df['Comp'] = df['Comp'].apply(lambda x: '{0:0>5}'.format(x))

            # Create Balloon ID for merges
            df['Balloon ID'] = df['Comp'] + df['Seq']

            # Ref1 is Balloon ID + UW Year
            df['CONTRACT_ID'] = df['Balloon ID'] + df['UWY_str']

            # Ref2 is Comp + first letter of Seq
            df['Ref2'] = df['Comp'] + df['Seq'].str[0] + df['UWY_str']

            try:
                # There is one case with a space. RT16 08 -> RT1608
                df['Retro Seq'] = df['Retro Seq'].str.replace(" ", "")
            except KeyError:
                # Including try-except clause because when deleting columns
                # to make register load faster, this probably gets deleted.
                # Since this column is not really important, we pass.
                pass

            # Sort values
            df.sort_values(by=['Comp', 'Seq', 'UW Yr'],
                           ascending=[True, True, False],
                           inplace=True)
        
        df['Treaty Type'] = self.get_treaty_type(df['Type/Form'])
        df['MonthsElapsed'] = calculate_months_elapsed(
            df,
            report_date=self.report_date)
        df['Run-off'] = df['CONTRACT_ID'].map(get_treaty_run_off_rate(df))
        df['REPORT_DATE'] = self.report_date
        df['YearFrac'] = calculate_year_fraction(df=df)

        return df
    
    @staticmethod
    def get_treaty_type(s : pd.Series) -> pd.Series:
        """Get the Treaty Type FAC, XL or Type/Form using Type/Form column.

        Parameters
        ----------
        s : pd.Series
            Type/Form column from main DataFrame.
        """
        conds = [(s == 'XL') | (s == 'FACX'), s == 'FAC']
        choices = ['XL', 'FAC']
        new_s = np.select(condlist=conds, choicelist=choices, default=s)
        return new_s
    
    def get_treaty_status(self, dataframe: pd.DataFrame) -> pd.Series:
        df = dataframe.copy()
        df['treaty_count'] = df['Balloon ID'].map(
            df['Balloon ID'].value_counts()
        )

        cond = [(df['Latest UWY']) & (df['Pd End'] < self.report_date),
                df['Treaty Type'].isin(['XL', 'FAC']),
                (df['Latest UWY']) & (df['treaty_count'] > 1),
                (df['Latest UWY']) & (df['treaty_count'] == 1)]

        choice = ['Run-off',
                  df['Treaty Type'],
                  'Renewed',
                  'New Treaty']

        df['Treaty Status'] = np.select(cond, choice, default='')

        return df['Treaty Status']
    
    def adjust_treaty_register(self) -> pd.DataFrame:
        """Adjusts Treaty Register. Also leaves all calculations columns.

        Parameters
        ----------
        oldest_uwy : int
        """

        # For Adjusted Registers, it's always Assumed Register
        df = self.register.copy()

        # Filter out older UW Yrs. Added if statement to avoid returning
        # empty DataFrame
        if self.oldest_uwy is not None:
            df = df.loc[df['UW Yr'].astype(int) >= self.oldest_uwy].copy()

        # Find Latest Underwriting Year
        df.loc[~df.duplicated(subset='Balloon ID'), 'Latest UWY'] = True
        df.loc[df.duplicated(subset='Balloon ID'), 'Latest UWY'] = False
        ### Manual Adjustment for Aserta. Antoine's Request
        df.loc[(df['Comp'] == '02585') & (df['UW Yr'] == 16),
               'Latest UWY'] = True

        df['Treaty Status'] = self.get_treaty_status(df)

        #######################################################################

        # Manual Adjustment to ASERTA Treaties. UW Year 20 is "New Treaty"
        df.loc[
            (df['Comp'] == '02585') & (df['UW Yr'] == 20),
            'Treaty Status'
        ] = 'New Treaty'

        #######################################################################

        renewed_status = df['Treaty Status'] == 'Renewed'

        df.loc[renewed_status, 'OUR_TPE_DIFF_SIGNED_LINE'] = (
            df['TPE'] * (df['Signed %'] - df['Signed %'].shift(-1))
        )

        df.loc[renewed_status, 'OUR_TPE_DIFF_ORG_GROWTH'] = (
            df['Signed %'].shift(-1) * (df['TPE'] - df['TPE'].shift(-1))
        )

        df.loc[
            df['OUR_TPE_DIFF_SIGNED_LINE'].isna(), 'OUR_TPE_DIFF_SIGNED_LINE'
        ] = 0

        df.loc[
            df['OUR_TPE_DIFF_ORG_GROWTH'].isna(), 'OUR_TPE_DIFF_SIGNED_LINE'
        ] = 0

        #######################################################################
        # Months earned
        cond_earned = [
            ((self.report_date - df['Pd Beg']) /
             np.timedelta64(1, 'Y')) > 0.76,
            ((self.report_date - df['Pd Beg']) / np.timedelta64(1, 'Y')) > 0.51,
            ((self.report_date - df['Pd Beg']) /
             np.timedelta64(1, 'Y')) > 0.26,
        ]

        choice_earned = [12, 9, 6]

        df['Months Earned'] = np.select(cond_earned, choice_earned, default=3)

        # TPE Phasing In or Out due to Signed Line
        df['TPE Phasing-Δ Signed %'] = (
            df['OUR_TPE_DIFF_SIGNED_LINE'] * df['Months Earned'] / 12
        )

        df['TPE Phasing-Δ Signed %'].fillna(0, inplace=True)

        #######################################################################
        # For Treaties that are new, Our TPE is phased in
        df['New Our TPE'] = np.where(
            df['Treaty Status'] == 'New Treaty',
            df['Our TPE'] * df['Months Earned'] / 12,
            (df['Our TPE']
             - (df['OUR_TPE_DIFF_SIGNED_LINE'] - df['TPE Phasing-Δ Signed %']))
        )

        df['New TPE'] = df['New Our TPE'] / df['Signed %']

        df['New Ledger TPE'] = (
            df['New TPE'] * df['Ledger TPE'] / df['TPE'].where(df['TPE'] != 0)
        )

        df.loc[df['New Ledger TPE'].isna(), 'New Ledger TPE'] = 0
        
        # Weird ZeroDivision error. Could not figure out. Had to use .where()
        df['New Ledger TPE'] = (
            df['New TPE'] * df['Ledger TPE'] / df['TPE'].where(df['TPE'] != 0)
        )
        df.loc[df['New Ledger TPE'].isna(), 'New Ledger TPE'] = 0

        # Weird ZeroDivision error. Could not figure out. Had to use .where()
        df['New Ledger Our TPE'] = (
            df['New Our TPE'] * df['Ledger Our TPE']
            / df['Our TPE'].where(df['Our TPE'] != 0)
        )
        df.loc[df['New Ledger Our TPE'].isna(), 'New Ledger Our TPE'] = 0

        #df.drop(columns=cols_to_remove, inplace=True)

        self.adjusted_register = df

        return df
    
    def generate_solvency_register(
        self,
        include_seadrill: bool = False
    ) -> pd.DataFrame:
        """Replace Old TPE columns with New TPE columns.

        Parameters
        ----------
        include_seadrill : bool, optional
            If True, then include Seadrill dummy treaty in final
            register, by default False.

        Returns
        -------
        pd.DataFrame
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        # Decided to make this less automated. Need to have more control
        # when the Register needs Adjustment etc.
        df = self.adjusted_register.copy()

        df['TPE'] = df['New TPE']
        df['Our TPE'] = df['New Our TPE']
        df['Ledger TPE'] = df['New Ledger TPE']
        df['Ledger Our TPE'] = df['New Ledger Our TPE']

        # Multiply Signed % Column by 100 for Internal Model
        df['Signed %'] = df['Signed %'] * 100

        # ---------------------------------------------------------------------
        # Seadrill Dummy
        if include_seadrill is True:
            seadrill_dummy = df.loc[
                (df['Comp'] == '09976')
                & (df['Seq'] == 'C1 03')
                & (df['UW Yr'] == 12)
            ].copy()

            # List with columns that have value = 3
            cols_same_val = ['100% EPI',
                            'Signed EPI',
                            'Signed EPI Euro',
                            'Revised EPI',
                            'Our Revised EPI Euro',
                            'EPI is Rev EPI or EPI',
                            'Tech Orig Prem',
                            'Tech Orig Net Paid',
                            'Tech Orig Brkg/Comm',
                            'Tech Orig OSLR',
                            'Tech Ledg Prem',
                            'Tech Ledg Net Paid',
                            'Tech Ledg Brkg/Comm',
                            'Tech Ledg OSLR',
                            'Fin Orig Prem',
                            'Fin Orig Accrd Prem',
                            'Fin Orig Net Paid',
                            'Fin Orig OSLR',
                            'Fin Ledg Prem',
                            'Fin Ledg Accrd Prem',
                            'Fin Ledg EPI',
                            'Fin Ledg Net Paid',
                            'Fin Ledg OSLR']
            
            for col in cols_same_val:
                if col not in df.columns:
                    # Raise to avoid creating additional columns.
                    raise ValueError(f"{col} is not in Register being used. Please update function code `generate_solvency_register()` in 'treaty_register.py'.")


            for col in cols_same_val:
                seadrill_dummy[col] = 3

            # Changing values according to previous Adjusted Treaty Registers
            seadrill_dummy['Seq'] = 'C1 99'
            seadrill_dummy['Pd End'] = pd.Timestamp('31/12/2020')
            seadrill_dummy['UPR'] = 'C03'
            seadrill_dummy['UPR Description'] = 'Credit - Calc1'
            seadrill_dummy['Last Booked Account'] = pd.Timestamp('30/09/2018')
            seadrill_dummy['Tech Inc Ratio (Orig)'] = 74.85977852
            seadrill_dummy['Tech Inc Ratio (Ledg)'] = 75.89347215
            seadrill_dummy['Fin Inc Ratio (Orig)'] = 71.55852225
            seadrill_dummy['Fin Inc Ratio (Ledg)'] = 72.02359481
            seadrill_dummy['Ledger TPE'] = 925883399.524691
            seadrill_dummy['TPE'] = 1165687200
            seadrill_dummy['Our TPE'] = 30307867.2
            seadrill_dummy['Ledger Our TPE'] = 24072968.3875483

            df = pd.concat([df, seadrill_dummy], ignore_index=True)

        # ---------------------------------------------------------------------
        # Formatting for Solvency II runs
        
        # Rename columns for Solvency II
        df.rename(columns=COLUMN_NAME_MAPPING, inplace=True)

        cols_to_2_dec = ['Orig EPI',
                         'Our Orig EPI',
                         'Our EPI Euro',
                         'Our Revised EPI Euro',
                         'Tech Orig Prem',
                         'Tech Orig Net Paid',
                         'Tech Orig Brkg/Comm',
                         'Tech Orig OSLR',
                         'Tech Ledg Prem',
                         'Tech Ledg Net Paid',
                         'Tech Ledg Brkg/Comm',
                         'Tech Ledg OSLR',
                         'Tech Inc Ratio (Orig)',
                         'Tech Inc Ratio (Ledg)',
                         'Fin Orig Prem',
                         'Fin Orig Accrd Prem',
                         'Fin Orig Net Paid',
                         'Fin Orig OSLR',
                         'Fin Ledg Prem',
                         'Fin Ledg Accrd Prem',
                         'Fin Ledg EPI',
                         'Fin Ledg Net Paid',
                         'Fin Ledg OSLR',
                         'Fin Inc Ratio (Orig)',
                         'Fin Inc Ratio (Ledg)',
                         'Deductible',
                         'Protected Share',
                         '100% UW Limit',
                         'Limit',
                         'Excess',
                         'Adj Rate %',
                         'Ceded %',
                         'Retro Share %']
        
        for col in cols_to_2_dec:
            try:
                df[col] = df[col].apply(lambda x: '{:.2f}'.format(x))
            except TypeError:
                df[col] = df[col].round(2)
        
        # Format as integer
        cols_to_int = ['Comp',
                       'UW Yr',
                       'T of T',
                       'Revised EPI',
                       'EPI is Rev EPI or EPI',
                       'Ledger TPE',
                       'TPE',
                       'Our TPE',
                       'Ledger Our TPE',
                       'LR for Min',
                       'LR for Max',
                       'Agg Limit',
                       'EEL Limit',
                       'Agg Deductible']

        for col in cols_to_int:
            try:
                df[col] = df[col].apply(lambda x: '{:.0f}'.format(x))
            except ValueError:
                print(f"{col} is of type str. Converting to int")
                df[col] = df[col].astype(int)
        
        # Remove Cedant names for commercial purposes
        df['Short'] = ''
        df['Cedant'] = ''

        df['Signed %'] = df['Signed %'].apply(lambda x: '{:.4f}'.format(x))
        
        cols_to_5_dec = ['Est Total Cost %',
                         'ULR %',
                         'Actuarial ULR %',
                         'Comm %',
                         'Profit Comm %',
                         'Brokerage %',
                         'Mgmnt Exp. %',
                         'O/R Comm %',
                         'Min Sliding Scale',
                         'Max Sliding Scale',
                         'Placed Share %',
                         'ROL %']

        for col in cols_to_5_dec:
            try:
                df[col] = df[col].apply(lambda x: '{:.5f}'.format(x))
            except TypeError:
                df[col] = df[col].round(5)

        self.solvency_register = df.loc[:, 'Comp':'Reinstatements']

        return df.loc[:, 'Comp':'Reinstatements']
    
    def export_solvency_register_to_txt(self):
        file_name = self._solvency_register_txt_filename()
        self.solvency_register.to_csv(
            file_name,
            index=False,
            sep='\t',
            date_format='%d/%m/%Y')
        
        append_eof_to_txt_file(filepath=file_name)
        
        print(f"{file_name} was created inside folder {os.getcwd()}.")

    def _solvency_register_txt_filename(self):
        # localize and convert to deal with Daylight Saving Time
        ts = pd.to_datetime('now', utc=True).tz_convert("Europe/Dublin")
        # This format is very specific.
        file_name = "{yr}{month:0>2}{day:0>2}{hr:0>2}{min:0>2}_{rep_yr}Q{rep_qtr}".format(
            yr = ts.year,
            month = ts.month,
            day = ts.day,
            hr = ts.hour + 1 if ts.minute > 30 else ts.hour,
            min = 0 if ts.minute > 30 else 30,
            rep_yr = self.report_date.year,
            rep_qtr = self.report_date.quarter,
        )

        file_name += "_IRE_TREATYREG_01.txt"
        return file_name

    def export_to_excel(self, solvency_reg: bool = False,
                        adjusted_reg_with_helper_cols: bool = False):
        
        excel_file = "Adjusted Register {}Q{}-produced on {}.xlsx".format(
            self.report_date.year,
            self.report_date.quarter,
            pd.to_datetime('now', utc=True).strftime('%Y%m%d-%Hh%M')
        )

        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(excel_file,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')

        if adjusted_reg_with_helper_cols:
            self._adjusted_reg_with_helper_cols_to_excel(excel_writer=writer)

        if solvency_reg:
            self.solvency_register.to_excel(
                writer,
                sheet_name='Solvency Register',
                index=False
            )

        writer.save()

        print("'{}' file created.".format(excel_file))
    
    def _adjusted_reg_with_helper_cols_to_excel(self, excel_writer):
        sh_name = 'Adjusted Register-Helper Cols'

        df = self.adjusted_register
        
        df.to_excel(excel_writer=excel_writer,
                    sheet_name=sh_name,
                    index=False)

        ws = excel_writer.sheets[sh_name]

        comment = "OUR_TPE_DIFF_ORG_GROWTH:\n"
        comment += "Difference between Current Treaty and Previous\n"
        comment += "using last year's Signed %. Full difference. No run-in.\n\n"
        comment += "Amount not being used to calculate <New Our TPE> column."

        col_position = df.columns.get_loc('OUR_TPE_DIFF_ORG_GROWTH')
        row_position = 0  # always header row
        
        ws.write_comment(row_position,
                         col_position,
                         comment,
                         {'width': 300, 'height': 150})
    
    

def calculate_months_elapsed(
    df: pd.DataFrame,
    report_date: Union[str, pd.Timestamp]
) -> pd.Series:
        """Calculate months elapsed since report date.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.Series
            Series with elapsed months since report date.
        """        
        df = df.copy()

        if isinstance(report_date, str):
            rep_date = pd.to_datetime(report_date, dayfirst=True)
        elif isinstance(report_date, pd.Timestamp):
            rep_date = report_date
        else:
            err = "report_date accepts str or Timestamp. "
            err += f"{type(report_date)} was passed."
            raise TypeError(err)

        
        # First we convert to days elapsed 
        df['DaysElapsed'] = report_date - df['Pd End']
        
        # If the treaty isn't expired, the result will be negative,
        # so we set to 0 (zero)
        df['MonthsElapsedRaw'] = np.where(
            # Check if treaty is not expired
            df['DaysElapsed'] <= pd.Timedelta(0),
            # if treaty not expired, set to zero (0)
            0,
            # if expired, then do calculation below
            # Add 0.01 due to edge cases. E.g.: if treaty starts 
            # in 2020-02-14, and IM run is on 2022-03-31, this will
            # produce 25.495, which will be converted to 25, but IM
            # will use 26
            df['DaysElapsed'] / np.timedelta64(1, 'Y') * 12 + 0.01
        )
        
        # round to make months integers
        df['MonthsElapsed'] = df['MonthsElapsedRaw'].round(0)
        df['MonthsElapsed'] = df['MonthsElapsed'].astype(np.int64)

        return df['MonthsElapsed']


def get_treaty_run_off_rate(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    run_off_rates = import_run_off_rates()

    if 'MonthsElapsed' not in df.columns.values:
        df['MonthsElapsed'] = calculate_months_elapsed(df)

    # Bring in Run-off Rates to Calculate TPE
    df = df.merge(run_off_rates,
                  on=['MonthsElapsed', 'UPR'],
                  how='left')

    # Returning Series because merge() operation sort rows.
    # This was causing the df['Run-off'] column to be associated with
    # wrong CONTRACT_IDs. Returning Series enables the use of map()
    return pd.Series(data=df['Run-off'].values, index=df['CONTRACT_ID'])


def calculate_year_fraction(df: pd.DataFrame) -> pd.Series:
    df = df.copy()

    # YearFrac represents the duration of a treaty, in years
    df['YearFrac'] = (df['Pd End'] - df['Pd Beg']) / np.timedelta64(1, "Y")

    return df['YearFrac'].round(decimals=2)
