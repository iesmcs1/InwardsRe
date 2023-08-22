from sqlite3 import Timestamp
from sys import path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import os
import logging

from .utils import append_eof_to_txt_file

# SII File name function
def generate_filename_for_solvency(file: str, version: str = '01') -> str:
    # localize and convert to deal with Daylight Saving Time
    ts = pd.to_datetime('now', utc=True).tz_convert("Europe/Dublin")
    quarter_ts = ts - pd.tseries.offsets.QuarterEnd()

    solvency_ts = "{yr}{month:0>2}{day:0>2}{hr:0>2}{min:0>2}".format(
        yr=ts.year,
        month=ts.month,
        day=ts.day,
        hr=ts.hour + 1 if ts.minute > 30 else ts.hour,
        min=0 if ts.minute > 30 else 30
    )

    year_quarter_ts = "{yr}Q{qtr}".format(
        yr=quarter_ts.year,
        qtr=quarter_ts.quarter,
    )

    return f"{solvency_ts}_{year_quarter_ts}_IRE_{file}_{version:0>2}.txt"

def _load_and_process_run_off_correction_file(filepath: str) -> pd.DataFrame:
    """Load and process file with the run-off corrected.

    Parameters
    ----------
    filepath : str
        Filepath to file

    Returns
    -------
    pd.DataFrame
        DataFrame to be used when generating the corrected SYMY file.
    """
    if filepath.endswith('.txt'):
        df = pd.read_csv(filepath, sep='\t')
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)

    # correct Dates
    df['CLD_DATE'] = df['CLD_DATE'].dt.strftime('%d/%m/%Y')
    df['CLA_DATE'] = df['CLA_DATE'].dt.strftime('%d/%m/%Y')

    return df


def correct_run_off(symy_filepath: str,
                    run_off_correction_filepath: str,
                    to_csv: bool = True,
                    return_df: bool = False) -> Union[None, pd.DataFrame]:
    """Correct and export SYMY extract to a text file.

    Parameters
    ----------
    symy_filepath : str
        Filepath to the SYMY extract. This extract is the one with reduced
        columns.
    run_off_correction_filepath : str
        Filepath to the corrected run-off. This file is manually produced.
    to_csv : bool, optional
        If True, function will export corrected DataFrame to text file,
        by default True
    return_df : bool, optional
        If True, function will return DataFrame, else None. By default, False.

    Returns
    -------
    Union[None, pd.DataFrame]
        If return_df is True, then returns DataFrame, else None.

    Raises
    ------
    TypeError
       if SYMY extract is not in a text file, an error will be raised.
        SYMY needs to be in a text file to speed up the process.
    """
    
    if symy_filepath.endswith('.txt'):
        df_symy = pd.read_csv(symy_filepath, sep='\t')
    else:
        raise TypeError("SYMY file needs to have extension `.txt`.")

    df_lapsed_rev = _load_and_process_run_off_correction_file(
        filepath=run_off_correction_filepath)
    
    # Create array with unique LEGACY_POLICYs (Balloon IDs)
    # This will be used to remove all named exp in SYMY file
    unique_leg_pol = df_lapsed_rev['LEGACY_POLICY'].unique()

    # Remove everything that is in run-off
    df_symy = df_symy.loc[~df_symy['LEGACY_POLICY'].isin(unique_leg_pol)]

    # Concat corrected DF with SYMY DF
    df_final = pd.concat([df_symy, df_lapsed_rev])

    try:
        # try to remove REFRESH_DATE column, given that it's not
        # necessary
        df_final.drop(columns=['REFRESH_DATE'], inplace=True)
    except KeyError:
        pass

    if to_csv is True:
        curr_date = pd.to_datetime('now', utc=True).strftime('%Y%m%d')
        filename = f'{curr_date}-SYMY run-off corrected.txt'
        df_final.to_csv(filename, sep='\t', index=False)

        print(f'`{filename}` created.')
    
    if return_df is True:
        return df_final
    else:
        return None

def format_col_to_int(s: pd.Series) -> pd.Series:
    return s.astype(np.int64)

def get_date_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if 'DATE' in col]

def format_col_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True)

def format_col_datetime_to_str(s: pd.Series) -> pd.Series:
    return s.dt.strftime('%d/%m/%Y')

def validate_report_date_arg(date: str) -> Timestamp:
    dat = pd.to_datetime(date, dayfirst=True)

    if not dat.is_quarter_end:
        msg = f"{date} is not quarter end. "
        msg += "When producing Solvency II text files, it should always "
        msg += "be quarter end (31/03, 30/06, 30/09 or 31/12)."
        print(msg)
    
    return dat

class SolvencyII:
    RENAME_COLS_DICT = {
        'DEPTOEPAR': {
            'BUYER_NUMBER': 'BusID',
            'BUYER_NAME': 'Debtor',
            'BUYER_COUNTRY_ISO': 'Country',
            'PARENT_NUMBER': 'ParentBusID'
        },
        'FACULTEXP': {
            'LEGACY_POLICY': 'BAlloon ID',
            'BUYER_NUMBER': 'Bus ID',
            'CLA_AMOUNT': 'Amount Requested',
            'CLD_TOTAL_AMOUNT': 'Amount Decided',
            'POLICY_CURRENCY': 'Currency',
            'Signed %': 'AttradiusReshare',
            'COMMITMENT_TYPE': 'LOB',
            'CLD_DATE': 'Answer Date',
            'CLA_DATE': 'Request Date',
            'CLD_EXPIRY_DATE': 'Expiry Date'
        },
        'TREATYEXP': {
            'LEGACY_POLICY': 'Balloon ID',
            'BUYER_NUMBER': 'Bus ID',
            'CLD_TOTAL_AMOUNT': 'Current Exposure',
            'POLICY_CURRENCY': 'Currency',
            'CLD_DATE': 'Last Updated',
            'COMMITMENT_TYPE': 'LOB',
        },
        'SPECIFEXP': {
            'LEGACY_POLICY': 'Balloon ID',
            'BUYER_NUMBER': 'Bus ID',
            'CLA_AMOUNT': 'Amount Requested',
            'CLD_TOTAL_AMOUNT': 'Amount Decided',
            'POLICY_CURRENCY': 'Currency',
            'Atradius Re Share': 'AttradiusReshare',
            'COMMITMENT_TYPE': 'LOB',
            'CLA_DATE': 'Request Date',
            'CLD_DATE': 'Answer Date',
            'CLD_EXPIRY_DATE': 'Expiry Date'
        },
        'TARENTTAB': {
            'PARENT_NUMBER': 'BusID',
            'PARENT_NAME': 'Parent',
            'PARENT_COUNTRY_ISO': 'Country'
        }
    }
    
    SOLVENCY_FILES = ['DEPTOEPAR',
                      'TARENTTAB',
                      'FACULTEXP',
                      'SPECIFEXP',
                      'TREATYEXP']
    
    COLS_IN_SYMY_EXTRACT = ['CREDIT_LIMIT_ID',
                            'BUYER_NUMBER',
                            'BUYER_NAME',
                            'BUYER_COUNTRY_ISO',
                            'PARENT_NUMBER',
                            'PARENT_NAME',
                            'PARENT_COUNTRY_ISO',
                            'LEGACY_POLICY',
                            'CLA_DATE',
                            'CLD_DATE',
                            'CLA_AMOUNT',
                            'CLD_TOTAL_AMOUNT',
                            'POLICY_CURRENCY',
                            'COMMITMENT_TYPE',
                            'CLD_EXPIRY_DATE']

    def __init__(self, report_date: str) -> None:
        self.report_date = validate_report_date_arg(date=report_date)
        self.symy = None
        self.register = None
        self.sharepoint_data = None
        self.expiry_dates = None
        self.retro_parameters = None
        self._data_correction = None

        # Starts log file for each run
        log_filename = "solvency_run_log_{}.log".format(
            pd.to_datetime('now', utc=True).strftime("%Y%m%d_%Hh%M")
        )
        # Apply log file configuration
        root_logger= logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_filename, 'w', 'utf-8')
        root_logger.addHandler(handler)
    
    @property
    def data_correction(self) -> dict:
        return self._data_correction
    
    @data_correction.setter
    def data_correction(self, data_correction: dict) -> None:
        # need to check if dict has LEGACY_POLICY and BUYER_NUMBER
        # for each correction to be made.
        # No need to check Treaty Register because Register is generated from
        # another code.
        for k in data_correction.keys():
            # Iterate through each table to be corrected
            for correction in data_correction[k]:
                # Iterate through each correction in specific tables
                if 'LEGACY_POLICY' not in correction.keys():
                    err_msg = 'LEGACY_POLICY not defined for '
                    err_msg += f'{correction} in {k}.'
                    raise ValueError(err_msg)
                elif 'BUYER_NUMBER' not in correction.keys():
                    err_msg = 'BUYER_NUMBER not defined for '
                    err_msg += f'{correction} in {k}.'
                    raise ValueError(err_msg)
        
        # if no errors were Raised, then OK to proceed
        self._data_correction = data_correction

        logging.info("Data correction loaded.")

        return None

    @property
    def DEPTOEPAR(self) -> pd.DataFrame:
        df = self.symy[['BUYER_NUMBER',
                        'BUYER_NAME',
                        'BUYER_COUNTRY_ISO',
                        'PARENT_NUMBER']].copy()

        df.drop_duplicates(subset='BUYER_NUMBER', inplace=True)

        # In case we have buyers with no parents, it means that the buyer
        # is the ultimate parent
        df['PARENT_NUMBER'].fillna(value=df['BUYER_NUMBER'], inplace=True)

        return df
    
    @property
    def TARENTTAB(self) -> pd.DataFrame:
        df = self.symy[['PARENT_NUMBER',
                        'PARENT_NAME',
                        'PARENT_COUNTRY_ISO']].copy()

        # For records in Symphony where the Parent columns are blank.
        # This usually means that the company is the parent company
        df.dropna(inplace=True)

        df.drop_duplicates(subset='PARENT_NUMBER', inplace=True)

        return df
    
    @property
    def FACULTEXP(self) -> pd.DataFrame:
        fac_table_name = 'FACULTEXP'
        
        # Copy Raw data from SYMY Extract
        symy_raw_data = self.symy.copy()

        # Filter only Exposure that begins with F in COMMITMENT_TYPE
        filter_is_fac = symy_raw_data['COMMITMENT_TYPE'].str[0] == 'F'
        facs_raw = symy_raw_data.loc[filter_is_fac]

        # Slice data with following columns
        facs = facs_raw[['CREDIT_LIMIT_ID',
                         'LEGACY_POLICY',
                         'BUYER_NUMBER',
                         'CLA_AMOUNT',
                         'CLD_TOTAL_AMOUNT',
                         'POLICY_CURRENCY',
                         'COMMITMENT_TYPE',
                         'CLA_DATE',
                         'CLD_DATE']].copy()

        # Merge in Expiry Dates using only CREDIT_LIMIT_ID
        facs = facs.merge(self.expiry_dates,
                          on='CREDIT_LIMIT_ID',
                          how='left')

        # If there is no date inside Expiry Date query,
        # then exposure should not be excluded
        # Including funciton here to log any buyers that do not have
        # an expiry date
        self.does_col_contain_nan(df=facs,
                                  column='CLD_EXPIRY_DATE',
                                  file_name=fac_table_name)

        # Add Register Data (Signed %)
        facs = self.add_register_share(facs)

        # Calculate Exposures at 100%
        facs['CLD_TOTAL_AMOUNT'] = (
            facs['CLD_TOTAL_AMOUNT'] * 100 / facs['Signed %'])

        facs['CLA_AMOUNT'] = facs['CLA_AMOUNT'] * 100 / facs['Signed %']

        # Adds product by returning B/C from COMMITMENT_TYPE
        facs['COMMITMENT_TYPE'] = facs['COMMITMENT_TYPE'].str[1]

        # Check if there's a correction to be made
        facs = self.apply_correction_to_data(df=facs,
                                             text_file=fac_table_name)

        # Formatting to match Solvency II files requirements
        facs['Signed %'] = facs['Signed %'].round(decimals=2)
        facs['CLA_AMOUNT'] = format_col_to_int(s=facs['CLA_AMOUNT'])
        facs['CLD_TOTAL_AMOUNT'] = format_col_to_int(facs['CLD_TOTAL_AMOUNT'])

        # Order columns for Solvency II
        return facs[['LEGACY_POLICY',
                     'BUYER_NUMBER',
                     'CLA_AMOUNT',
                     'CLD_TOTAL_AMOUNT',
                     'POLICY_CURRENCY',
                     'Signed %',
                     'COMMITMENT_TYPE',
                     'CLD_DATE',
                     'CLA_DATE',
                     'CLD_EXPIRY_DATE']].copy()
    
    @property
    def SPECIFEXP(self) -> pd.DataFrame:
        spec_table_name = 'SPECIFEXP'

        symy_raw = self.symy

        # From SYMY Main, keep rows where LEGACY POLICY ends in SA
        filter_is_sa = symy_raw['LEGACY_POLICY'].str.endswith('SA')

        # Remove exposures that are Zero
        filter_is_not_zero = symy_raw['CLD_TOTAL_AMOUNT'] > 0

        # Filter data
        sa_raw = symy_raw.loc[
            (filter_is_sa) & (filter_is_not_zero)].copy()
        
        # If CLD_EXPIRY_DATE column is present in SYMY data, then drop it
        # Otherwise, this will create a duplicate, and the merge will append
        # an _x and _y at the end of both columns.
        if 'CLD_EXPIRY_DATE' in sa_raw.columns:
            sa_raw.drop(columns=['CLD_EXPIRY_DATE'], inplace=True)

        # Merge in Expiry Dates from SYMY Expiry dates extract
        sa = sa_raw.merge(self.expiry_dates,
                          on='CREDIT_LIMIT_ID',
                          how='left')

        # Normalize LEGACY_POLICY column for mergin SharePoint data
        sa['LEGACY_POLICY'] = sa['LEGACY_POLICY'].str[:10]

        # Merge in SharePoint data
        sa = sa.merge(self.sharepoint_data,
                      left_on=['LEGACY_POLICY', 'BUYER_NUMBER'],
                      right_on=['Balloon ID', 'BUS ID'],
                      how='left')

        # If there is no date inside Expiry Date query,
        # then exposure should not be excluded
        # Including funciton here to log any buyers that do not have
        # an expiry date
        self.does_col_contain_nan(df=sa,
                                  column='CLD_EXPIRY_DATE',
                                  file_name=spec_table_name)

        # Adds product by returning B/C from COMMITMENT_TYPE
        sa['COMMITMENT_TYPE'] = sa['COMMITMENT_TYPE'].str[1]

        #spec_symy['Atradius Re Share'] = spec_symy['Atradius Re Share'].round(decimals=4)

        # Calculate Exposures at 100%
        sa['CLD_TOTAL_AMOUNT'] = sa['CLD_TOTAL_AMOUNT'] / sa['Atradius Re Share']
        sa['CLA_AMOUNT'] = sa['CLA_AMOUNT'] / sa['Atradius Re Share']

        # Format Shares for Solvency II
        sa['Atradius Re Share'] = (
            sa['Atradius Re Share'] * 100).round(decimals=2)

        # Check if there's a correction to be made
        sa_corrections = self.apply_correction_to_data(
            df=sa,
            text_file=spec_table_name)

        # Remove TPE that has CLD date after report date
        sa_date_filter = self._remove_cld_date_after_report_date(
            df=sa_corrections,
            text_file=spec_table_name)
        
        sa_date_filter['CLD_TOTAL_AMOUNT'] = format_col_to_int(
            s=sa_date_filter['CLD_TOTAL_AMOUNT'])
        sa_date_filter['CLA_AMOUNT'] = format_col_to_int(
            s=sa_date_filter['CLA_AMOUNT'])

        # Order columns for Solvency II
        return sa_date_filter[['LEGACY_POLICY',
                               'BUYER_NUMBER',
                               'CLA_AMOUNT',
                               'CLD_TOTAL_AMOUNT',
                               'POLICY_CURRENCY',
                               'Atradius Re Share',
                               'COMMITMENT_TYPE',
                               'CLD_DATE',
                               'CLA_DATE',
                               'CLD_EXPIRY_DATE']].copy()
    
    @property
    def TREATYEXP(self) -> pd.DataFrame:
        treaty_table_name = 'TREATYEXP'
        symy_raw = self.symy

        # Remove Commitment type if begings with F or S
        filter_only_treaty_exp = (
            ~symy_raw['COMMITMENT_TYPE'].str[0].isin(['F', 'S'])
        )

        # Remove CLD = 0
        filter_amounts_above_zero = symy_raw['CLD_TOTAL_AMOUNT'] > 0

        # Apply filters to Raw SYMY Extract
        tty_exp = symy_raw.loc[
            (filter_only_treaty_exp) & (filter_amounts_above_zero)
        ].copy()

        # Add Signed % column
        tty_exp = self.add_register_share(tty_exp)

        # Calculate Total Limit (from our share)
        # # Because the Signed % is in whole number (50 instead of 0.5),
        # # we multiple by 100
        tty_exp['CLD_TOTAL_AMOUNT'] = (
            tty_exp['CLD_TOTAL_AMOUNT'] * 100 / tty_exp['Signed %']
        )

        # Remove retro, as to reach 100% of Exposure
        if isinstance(self.retro_parameters, dict):
            for leg_pol, retro_share in self.retro_parameters.items():
                # Retro share is how much we are Retroceding. To find our
                # amount without Retro, we divide by how much we keep
                print(f"Retro: {leg_pol} with retro share {retro_share}")
                leg_pol_filter = tty_exp['LEGACY_POLICY'] == leg_pol
                # tty_exp.loc[leg_pol_filter, 'CLD_TOTAL_AMOUNT'] = (
                #     tty_exp['CLD_TOTAL_AMOUNT'] / (1 - retro_share)
                # )

                tty_exp['CLD_TOTAL_AMOUNT'] = np.where(
                    leg_pol_filter,
                    tty_exp['CLD_TOTAL_AMOUNT'] / (1 - retro_share),
                    tty_exp['CLD_TOTAL_AMOUNT']
                )
        
        # Turn CLD TOTAL AMOUNT to integer
        tty_exp['CLD_TOTAL_AMOUNT'] = format_col_to_int(
            s=tty_exp['CLD_TOTAL_AMOUNT'])

        # Keep only first letter of commitment type
        tty_exp['COMMITMENT_TYPE'] = tty_exp['COMMITMENT_TYPE'].str[0]
        
        # Check if there's a correction to be made
        tty_exp_corrections = self.apply_correction_to_data(
            df=tty_exp,
            text_file=treaty_table_name
        )
        
        # Remove TPE that has CLD date after report date
        tty_exp_date_filter = self._remove_cld_date_after_report_date(
            df=tty_exp_corrections,
            text_file=treaty_table_name)
        
        tty_exp_date_filter['CLD_TOTAL_AMOUNT'] = format_col_to_int(
            s=tty_exp_date_filter['CLD_TOTAL_AMOUNT'])
        
        # Order columns for Solvency II
        return tty_exp_date_filter[['LEGACY_POLICY',
                                    'BUYER_NUMBER',
                                    'CLD_TOTAL_AMOUNT',
                                    'POLICY_CURRENCY',
                                    'CLD_DATE',
                                    'COMMITMENT_TYPE']].copy()
    
    @staticmethod
    def does_col_contain_nan(df: pd.DataFrame,
                             column: str,
                             file_name: str) -> bool:
        """Returns true of false, if column contains NaN values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be checked
        column : str
            Column in DataFrame to be checked.

        Returns
        -------
        bool
            True if column contains NaN, else False.
        """
        if df[column].notnull().all():
            # If DataFrame does not contain NaN, then return False
            return False
        else:
            # If column contain NaN, then filter only NaN
            df_only_nan = df.loc[df[column].isna()]
            for _, row in df_only_nan.iterrows():
                # log warning message for each buyer that was excluded from
                # DataFrame
                log_msg = f'{file_name} # removal - '
                log_msg += f"LEG_POL: {row['LEGACY_POLICY']}"
                log_msg += ' - BUYER: ' + str(row['BUYER_NUMBER'])
                log_msg += ' - CLA_DATE: ' + str(row['CLA_DATE'])
                log_msg += ' - CLD_DATE: ' + str(row['CLD_DATE'])
                log_msg += f' - *{column}: ' + str(row[column]) + '*'
                logging.warning(log_msg)

        return True
    
    def load_data_correction(
        self,
        data_correction_filepath: str = None
    ) -> None:
        """Load data correction to model, to be applied when generating
        text files.

        Parameters
        ----------
        data_correction_filepath : str
            Path to file with data corrections. This has to be an Excel file
            with the tabs `FACULTEXP`, `SPECIFEXP` and `TREATYEXP`.

            Every data correction has to have a BALLOON_ID and a BUYER_NUMBER
            in order for the correction to be mapped.
        """
        if data_correction_filepath is not None:
            if isinstance(data_correction_filepath, str):
                self.data_correction = self._load_correction_excel(
                    filepath=data_correction_filepath)
            else:
                err = 'data_correction_dict has to be a dictionary. '
                err += f'{type(data_correction_filepath)} was passed.'
                raise TypeError(err)

    def _load_correction_excel(self, filepath: str) -> dict:
        """Load correction from Excel file.

        Parameters
        ----------
        filepath : str
            Path to file.

        Returns
        -------
        dict
            All corrections in Excel file will be returned as a dict, which
            has as keys all sheet names.
        """
        correction_dict = {}
        tables_with_corrections = ['FACULTEXP', 'SPECIFEXP', 'TREATYEXP']

        # Read excel file with corrections
        excel_sheets = pd.read_excel(filepath,
                                     sheet_name=tables_with_corrections,
                                     dtype={'BUYER_NUMBER': str})
        
        # iterate through each table
        for sheet, df_in_sheet in excel_sheets.items():
            for col in get_date_columns(df_in_sheet):
                # Need to convert datetime to string to avoid 
                # errors in ITS text files
                try:
                    df_in_sheet[col] = format_col_datetime_to_str(
                        s=df_in_sheet[col]
                    )
                except AttributeError:
                    # In case Date column has no corrections
                    pass

            # Convert each DataFrame to dict by T (transposing).
            # This creates a DataFrame where the column headers turn
            # into the indexes.
            corrections_group = df_in_sheet.T.to_dict()

            # Variable will hold each dictionary with corrections
            # for a specific Table
            correction_array = []

            # We only need the values, since the keys are the row numbers
            for correction in corrections_group.values():
                correction_array.append(self._clean_corrections(correction))

            # Save cleaned correction array to correction dict
            correction_dict[sheet] = correction_array
        
        return correction_dict
    
    @staticmethod
    def _clean_corrections(corr: dict) -> dict:
        """Remove any NaN from correction dictionary.

        Parameters
        ----------
        corr : dict
            dictionary with the corrections

        Returns
        -------
        dict
            Dictionary without any NaN values.
        """
        clean_dict = {}
        
        # Iterate through each column
        for col, val in corr.items():
            if not pd.isna(val):
                # If column has value other than NaN, then is valid
                # Then we save it to new dictionary, as to not alter the
                # original dictionary.
                clean_dict[col] = val
    
        return clean_dict
    
    def apply_correction_to_data(self,
                                 df: pd.DataFrame,
                                 text_file: str) -> pd.DataFrame:
        df = df.copy()

        if self.data_correction is None:
            return df
        elif not self.data_correction[text_file]:
            # If list is empty, then return original df
            return df
        else:
            print(f"Applying corrections to {text_file}.")
            # If table has corrections to be made
            corrections = self.data_correction[text_file]

        for corr in corrections:
            # iterate through each individual dictionary representing 
            # correction

            # Create filters
            leg_pol_filter = df['LEGACY_POLICY'] == corr['LEGACY_POLICY']
            buyer_number_filter = df['BUYER_NUMBER'] == corr['BUYER_NUMBER']

            # Create list with all the columns to be changed, excluding the
            # reference columns (to be used to uniquely identify an exposure)
            ref_cols = ['LEGACY_POLICY', 'BUYER_NUMBER']
            cols_to_change = [
                col for col in corr.keys() if col not in ref_cols]
            
            # Create filtered DataFrame to check if Reference exists
            df_filtered = df.loc[(leg_pol_filter) & (buyer_number_filter)]
            if df_filtered.shape[0] == 0:
                # If DataFrame has 0 rows, then warn user
                msg = f"{text_file} # error - "
                msg += f"{corr['LEGACY_POLICY']} (Balloon ID)"
                msg += f" + {corr['BUYER_NUMBER']} (Bus ID) not found."
                logging.warning(msg)
            else:
                # If Reference exists in DataFrame, then apply corrections
                for col in cols_to_change:
                    # Iterate through each column and Apply corrections
                    # to DataFrame

                    # Save value that is about to be corrected in variable
                    previous_value = df.loc[
                        (leg_pol_filter) & (buyer_number_filter),
                        col
                    ].values[0]

                    # Check if previous value and current value are different
                    if previous_value != corr[col]:
                        # Log to file each correction before correction
                        # only if value to be changed and correction are
                        # different
                        msg = f"{text_file} # correction - "
                        msg += f"LEG_POL: {corr['LEGACY_POLICY']} - "
                        msg += f"BUYER: {corr['BUYER_NUMBER']} - "
                        msg += f"{col}: {previous_value} => {corr[col]}"

                        logging.warning(msg)

                        # Make correction
                        df.loc[
                            (leg_pol_filter) & (buyer_number_filter),
                            col
                        ] = corr[col]

        return df
    
    def add_retro_parameters(self, balloon_id_filter: Dict[str, float]) -> None:
        """Add Retro parameters to calculate 100% TPE.

        Notes
        -----
        For Solvency II, ITS requests that the TPE be at 100%. This means that
        if there is a treaty with retrocession, we need to calculate the 
        exposure before the retrocession. This will result in a bigger
        exposure than recorded in Symphony, for example.

        Parameters
        ----------
        balloon_id_filter : Dict[str, float]
            Dictionary with legacy_policy and the amount retroceded.

        Returns
        -------
        None
            The dictionary passed will be stored in the class instance.
        """
        self.retro_parameters = balloon_id_filter

        print("Retro parameters added to model.")
        return None
    
    def is_ok_to_proceed(self) -> bool:
        is_ok_to_proceed = True
        files_missing = 'The following files still need to be uploaded:\n'

        data_needed = [(self.symy, 'SYMY Extract'),
                       (self.register, 'Solvency Register'),
                       (self.sharepoint_data, 'SharePoint data'),
                       (self.expiry_dates, 'SYMY Expiry Date extract')]

        for data in data_needed:
            if data[0] is None:
                files_missing += f'- {data[1]}\n'

        if is_ok_to_proceed:
            return True
        else:
            print(files_missing)
            return False
        
    @staticmethod
    def _import_file(file_path: str, dtype: Union[None, dict] = None) -> pd.DataFrame:
        """Checks if file is xlsx or txt, and imports it.

        Parameters
        ----------
        filepath : str
            path to file.
        dtype : Union[None, dict], optional
            dtypes to be used when loading data using Pandas library,
            by default None. When passing None, it will not affect
            data loading.

        Returns
        -------
        pd.DataFrame
            [description]
        """
        if file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, dtype=dtype)
        elif file_path.endswith('.txt'):
            return pd.read_csv(file_path, sep='\t', dtype=dtype)

    def import_symy_data(self, filepath: str) -> None:
        dtype = {'BUYER_NUMBER': str}
        
        # Read SYMY Extract file
        df = self._import_file(file_path=filepath, dtype=dtype)

        cols_not_present = [
            col for col in self.COLS_IN_SYMY_EXTRACT if col not in df.columns
        ]

        if len(cols_not_present) > 0:
            err = 'The following columns are missing from the SYMY file:\n'
            err += f'{cols_not_present}.'
            raise ValueError(err)
        
        # For some reason, dtypes are not converting BUYER_NUMBER to str,
        # so I had to include this line
        df['BUYER_NUMBER'] = df['BUYER_NUMBER'].astype(np.int64).astype(str)

        # Correcting PARENT_NUMBER containing decimals
        ## Substitute NaNs in PARENT_NUMBER col with Buyer Numbers.
        df.loc[df['PARENT_NUMBER'].isna(), 'PARENT_NUMBER'] = df['BUYER_NUMBER']
        df['PARENT_NUMBER'] = format_col_to_int(
            df['PARENT_NUMBER']).astype(str)

        # Remove chars that cause errors in NAME columns
        for col in ['BUYER_NAME', 'PARENT_NAME']:
            df[col] = df[col].str.replace('[\'\"\*]', '', regex=True)

        self.symy = df

        print("SYMY Extract added to model.")
        return None
    
    def _remove_cld_date_after_report_date(
        self,
        df: pd.DataFrame,
        text_file: str
    ) -> pd.DataFrame:
        """Remove all credit limits that were approved after report date.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that will be filtered.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        # Check how many limits have CLD_DATE after report_date
        df_invalid_cld = df.loc[
            pd.to_datetime(df['CLD_DATE'], dayfirst=True) > self.report_date
        ]

        if df_invalid_cld.shape[0] > 0:
            print_msg = "Removing buyers with CLD_Date after report date "
            print_msg += f"{self.report_date} on table {text_file}."
            print(print_msg)

            for _, row in df_invalid_cld.iterrows():
                msg = f"{text_file} # removal - "
                msg += f"LEG_POL: {row['LEGACY_POLICY']} - "
                msg += f"BUYER: {row['BUYER_NUMBER']} - "
                msg += f"CLD_DATE: {row['CLD_DATE']}"

                logging.warning(msg)

        return df.loc[
            pd.to_datetime(df['CLD_DATE'], dayfirst=True) <= self.report_date
        ].copy()

    def import_register_data(self, filepath: str) -> None:
        """Load solvency register and process it.

        Notes
        -----
        The register being loaded is the text file sent to ITS,
        not the original register.

        Parameters
        ----------
        filepath : str
            [description]

        Returns
        -------
        [type]
            [description]
        """
        # Read SYMY Extract file
        dtype = {'Comp': str}
        df = self._import_file(file_path=filepath, dtype=dtype)

        # Sort values as to have latest first
        df.sort_values(
            by=['Comp', 'Seq', 'UW Yr'],
            ascending=[True, True, False],
            inplace=True)

        # Remove later UW Years, and keep only latest year
        df.drop_duplicates(subset=['Comp', 'Seq'], inplace=True)

        # Make sure Comp is str and has len 5
        df['Comp'] = df['Comp'].apply(lambda x: '{0:0>5}'.format(x))

        # Create KEY to be used by the other files when merging data
        df['LEGACY_POLICY'] = df['Comp'] + df['Seq']

        # Assign register to class instance
        self.register = df[['LEGACY_POLICY', 'Signed %']]

        print("Register data added to model.")

        return None

    def import_sharepoint_data(self, filepath: str) -> None:
        dtype={'BUS ID': str}
        df = self._import_file(file_path=filepath, dtype=dtype)

        # Sort values so we keep most recent when dropping duplcates
        df.sort_values(by='Answer date', ascending=False, inplace=True)
        df.drop_duplicates(subset=['Balloon ID', 'BUS ID'], inplace=True)
        
        # Slice data to keep only what is needed
        sharepoint_cols = ['Balloon ID', 'BUS ID', 'Atradius Re Share']

        self.sharepoint_data = df[sharepoint_cols]

        print("SharePoint data added to model.")
        return None

    def import_expiry_dates_data(self, filepath: str) -> None:
        df = self._import_file(file_path=filepath)
        
        # Only 2 columns are needed from this file
        self.expiry_dates = df[['CREDIT_LIMIT_ID', 'CLD_EXPIRY_DATE']]
        
        print("SYMY Expiry Dates added to model.")
        
        return None
    
    def add_register_share(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add Register Data (Signed %)
        return df.merge(self.register,
                        on='LEGACY_POLICY',
                        how='left')
    
    def _generate_txt_file(self,
                           df: pd.DataFrame,
                           file: str,
                           file_version: str) -> None:

        if file in ['FACULTEXP', 'SPECIFEXP']:
            # Add Comments column due to Solvency II file requirements
            df['Comments'] = '-'

        # Rename to match Solvency II files requirements. [file] is accessing
        # the correct file naming convention.
        df.rename(columns=self.RENAME_COLS_DICT[file], inplace=True)

        filename = generate_filename_for_solvency(file=file,
                                                  version=file_version)

        df.to_csv(filename, sep='\t', date_format='%d/%m/%Y', index=False)

        append_eof_to_txt_file(
            filepath=os.path.join(os.getcwd(), filename))

        print(f"{filename} file created in `{os.getcwd()}`.")

        return None
        
    def generate_txt_files(self, file_version: str = '01'):

        if self.is_ok_to_proceed() is False:
            return None
        
        for file_name in self.SOLVENCY_FILES:
            # iterate through list and create each file
            # Using eval here so we can access class property that
            # is the file name
            self._generate_txt_file(df=eval(f"self.{file_name}"),
                                    file=file_name,
                                    file_version=file_version)

        print("Files generated successfully.")
