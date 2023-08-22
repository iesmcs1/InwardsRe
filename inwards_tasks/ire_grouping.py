import re
from typing import Union
import pandas as pd
import os

try:
    from treaty_register import Register
except ModuleNotFoundError:
    from .treaty_register import Register

class IREGrouping:
    def __init__(self, report_date: str) -> None:
        self.report_date = pd.to_datetime(report_date, dayfirst=True)
        self._register = None
        self._ire_grouping = None
    
    @property
    def ire_grouping(self) -> pd.DataFrame:
        return self._ire_grouping
    
    @ire_grouping.setter
    def ire_grouping(self, df: pd.DataFrame) -> None:
        # Apply formatting to Comp column
        df['Comp'] = df['Comp'].apply(lambda x: '{:0>5}'.format(x))
        
        self._ire_grouping = df

        return None
    
    @property
    def register(self) -> pd.DataFrame:
        return self._register

    @register.setter
    def register(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # in Dec/2020, the Column Name changed. This deals with that
        df.rename(columns={'Layer No': 'Layer Number'}, inplace=True)

        df.loc[~df['Treaty Type'].isin(['XL', 'FAC']), 'Treaty Type'] = 'QS'

        # This is exclusive for IRE_GROUPINGS, as the other processing methods use UPR Code
        df['B/C'] = df['Seq'].str[0]

        self._register = df
    
    @property
    def max_uw_year(self) -> int:
        return self.register['UW Yr'].max()

    def import_assumed_register(self, filepath: str) -> None:
        # Create Register class
        reg_class = Register(filepath=filepath,
                             report_date=self.report_date)
        
        self.register = reg_class.register

        print(f'{filepath} imported successfully.')
        return None
    
    def import_ire_grouping(self, filepath : str) -> pd.DataFrame:
        df = pd.read_csv(filepath,
                         sep='\t',
                         dtype={'UW_Yr': int},
                         skipfooter=1,
                         engine='python')
        
        self.ire_grouping = df

        print(f'{filepath} imported successfully.')
        return None
    
    def extract_date_from_filepath(self, filepath: str) -> pd.Timestamp:
        file_name = os.path.basename(filepath)

        # Searching for format "2021Q4" | format "2022 January 31st"
        p = re.compile('(\d{4}[Q]\d{1}|\d{4}\s\w+\s\d{2})')

        t = p.search(os.path.basename(filepath))

        if t is None:
            err_msg = f'Cant extract date from file {file_name}.\n'
            err_msg += 'File should have original name format.'
            raise ValueError(err_msg)
        
        t_str = t.group(1)

        if 'Q' in t_str:
            # If dealing with text files delivered to ITS
            t_for_datetime = t_str[:-2] + "{0:0>2}".format(str(int(t_str[-1]) * 3))
            datetime_format = '%Y%m'
        else:
            t_for_datetime = t_str
            datetime_format = "%Y %B %d"

        return pd.to_datetime(t_for_datetime, format=datetime_format) + pd.tseries.offsets.MonthEnd()
    
    def generate_detection_pivot(self, latest_uwy: Union[None, int] = None):
        """Returns pandas.DataFrame with Treaties that are ambiguous 
        and do not a have valid Grouping_ID in last Solvency II run.
        
        Parameters
        ----------
        latest_uwy : None or int
            if none, then it will use the highest UW Yr in
            treaty Register
        """
        if latest_uwy is None:
            latest_uwy = self.max_uw_year
        
        # To keep only useful columns
        # CONTRACT_ID -> Balloon ID + UWY
        main_cols = ['Comp',
                     'Seq',
                     'UW Yr',
                     'B/C',
                     'Treaty Type',
                     'CONTRACT_ID']
        
        df = self.register[main_cols].copy()

        # Need to analyse only Lastest UW Year and QS/XL. FAC not needed
        df = df.loc[(df['UW Yr'] == latest_uwy) & (df['Treaty Type'] != 'FAC')]
        
        # To count the amount of treaties belonging to QS/LX and B/C
        df_piv = df.pivot_table(
            values='Seq',
            index='Comp',
            columns=['Treaty Type', 'B/C'],
            aggfunc='count'
        )

        # to_flat_index(), reset_index() and fillna() to make life easier
        df_piv.columns = df_piv.columns.to_flat_index()
        df_piv.reset_index(inplace=True)
        df_piv.fillna(0, inplace=True)

        # Count QS to see which ones we have to investigate
        df_piv['Count QS'] = df_piv[('QS', 'B')] + df_piv[('QS', 'C')]

        # Reduce visual pollution of float numbers
        # In this case, I know first column is Comp, so skip it
        df_piv.iloc[:, 1:] = df_piv.iloc[:,1:].astype(int)
        
        # We only need to check if there're more than 1 QS and XL is present
        multi_QS = df_piv['Count QS'] > 1
        has_XL_B = df_piv[('XL', 'B')] > 0
        has_XL_C = df_piv[('XL', 'C')] > 0
        
        df_piv = df_piv.loc[multi_QS]
        df_piv = df_piv.loc[(has_XL_B) | (has_XL_C)]

        return df_piv
    
    def summary_for_underwriters(self, latest_uwy: Union[None, int] = None):
        """Generates table with previous UW Yr QS and XL structure.
        Parameters
        ----------
        latest_uwy : None or int
            if none, then it will use the highest UW Yr in treaty Register.
        """
        # If None (False), then use highest value in UW Yr column
        if latest_uwy is None:
            latest_uwy = self.max_uw_year
        
        grp = self.ire_grouping
        piv = self.generate_detection_pivot(latest_uwy=latest_uwy)

        # Create Reference Column for merge()
        grp['Reference'] = (
            grp['Grouping_ID'].str[1:6]  # Comp code
            + grp['Seq']
            + grp['Grouping_ID'].str[-2:]  # Last 2 digits (UWY)
        )

        # For Summary, we only want treaties that have a Grouping ID
        has_grouping_id = grp['Comp'].isin(piv['Comp'])
        
        # We also want only tha latest
        most_recent_uwy = grp['UW_Yr'] == grp['UW_Yr'].max()
        
        # Filter IRE_Grouping DataFrame
        grp = grp.loc[(has_grouping_id) & (most_recent_uwy)]

        # Insert Layer Number
        grp = grp.merge(
            self.register[['CONTRACT_ID', 'Layer Number']],
            how='left',
            right_on='CONTRACT_ID',
            left_on='Reference'
        )

        # Produce DF with columns for each layer, with corresponding QS
        try:
            grp_pivoted = grp.pivot(index='Grouping_ID',
                                    columns='Layer Number',
                                    values='Seq').reset_index()
        except ValueError:
            # If for the same Grouping_ID we have duplicated Layer Number
            duplicated_vals = grp[grp.duplicated(
                subset=['Grouping_ID', 'Layer Number'],
                keep=False
            )]
            
            # To check if we're iterating over 1st or later entries
            count = 0

            # chr(97) is "a"
            a_chr = 97

            # Store the values to change in this array, to substitute later on
            array_to_sub = []

            for _, row in duplicated_vals.iterrows():
                if count > 0:
                    # If it's not the first entry
                    array_to_sub.append(
                        str(row['Layer Number']) + chr(a_chr+count)
                    )
                else:
                    # If it's first entry, we leave as it is
                    array_to_sub.append(row['Layer Number'])
                count += 1

            #duplicated_vals['Layer Number'] = array_to_sub
            grp.loc[
                grp.duplicated(subset=['Grouping_ID', 'Layer Number'],
                                keep=False),
                'Layer Number'
            ] = array_to_sub

            grp_pivoted = grp.pivot(index='Grouping_ID',
                                    columns='Layer Number',
                                    values='Seq').reset_index()

        # Balloon ID and Comp ID UWs
        grp_pivoted['Comp'] = grp_pivoted['Grouping_ID'].str[1:6]
        grp_pivoted['Balloon ID'] = grp_pivoted['Comp'] + grp_pivoted[0]

        grp_pivoted.columns.name = None

        # make sure all Comp ID's are in the final DataFrame
        grp_pivoted = grp_pivoted.fillna("").merge(
            piv['Comp'],
            on='Comp',
            how='outer'
        )

        # Bring in Cedant name
        grp_pivoted = grp_pivoted.merge(
            self.register[[
                'Comp', 'Cedant', 'UW Yr'
            ]].drop_duplicates(subset=['Comp', 'Cedant']),
            on='Comp',
            how='left'
        )

        # Bring in Short name
        grp_pivoted = grp_pivoted.merge(
            self.register[['Balloon ID', 'Short']].drop_duplicates(),
            on='Balloon ID',
            how='left'
        )

        grp_pivoted.rename(columns={0: 'QS'}, inplace=True)

        # Organize columns
        cols = list(grp_pivoted.columns)
        cols = cols[-5:] + cols[0:-5]
        return grp_pivoted[cols]
    
    def get_treaties(self, comp_id):
        df = self.register[['Comp',
                            'Seq',
                            'Cedant',
                            'UW Yr',
                            'Short',
                            'Treaty Type',
                            'Layer Number']]

        df_comp = df.loc[df['Comp'] == comp_id]
        
        return  df_comp.loc[df_comp['UW Yr'] == df_comp['UW Yr'].max()]
    
    def to_excel(self):
        """Creates Excel file with Detection Pivot, Summary for UW and
        Assumed Treaty Register.
        """
        file_name = 'Construct IRE_GROUPING {}Q{}.xlsx'.format(
            self.report_date.year,
            self.report_date.quarter,
        )

        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(file_name,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')

        # Number and text formatting
        wb  = writer.book
        format_num_0f = wb.add_format({'num_format': '#,##0'})
        format_pct_2f = wb.add_format({'num_format': '0.00%'})

        # Write Dataframes to Excel file
        
        # ---------------------------------------------------------------------
        # Assumed Treaty Register
        reg = self.register
        reg_sh_name = 'Treaty Register'
        reg.to_excel(writer, sheet_name=reg_sh_name, index=False)

        # Formatting
        reg_sh = writer.sheets[reg_sh_name]
        reg_sh.set_column("D:E", 35)  # Short and Cedant cols
        reg_sh.set_column("K:M", 11)  # Pd Beg, Pd End and Inception cols
        reg_sh.set_column("AC:AC", 9, format_pct_2f)  # Signed % col
        reg_sh.set_column("AD:AI", 19, format_num_0f)  # EPI cols
        reg_sh.set_column("BF:BI", 19, format_num_0f)  # EPI cols
        

        # ---------------------------------------------------------------------
        # IRE_GROUPING txt file
        ire_txt_sh_name = 'IRE_GROUPING'
        self.ire_grouping.to_excel(writer,
                                   sheet_name=ire_txt_sh_name,
                                   index=False)
        
        # Formatting
        ire_txt_sh = writer.sheets[ire_txt_sh_name]
        ire_txt_sh.set_column("A:D", 15)

        # ---------------------------------------------------------------------
        # Detection
        piv_det = self.generate_detection_pivot()
        piv_det.to_excel(writer,
                         sheet_name='Detection',
                         index=False)

        detection_text1 = "Treaties to be investigated."
        detection_text2 = "2 or more QS with XL."
        sheet_detection = writer.sheets['Detection']
        sheet_detection.write('J1', detection_text1)
        sheet_detection.write('J2', detection_text2)

        # ---------------------------------------------------------------------
        # Summary for Underwriters
        summary = self.summary_for_underwriters()
        summary.to_excel(writer,
                         sheet_name='Summary for UW',
                         index=False)
        
        # ---------------------------------------------------------------------
        # List of Renewed Treaties
        filter1 = reg['Comp'].isin(piv_det['Comp'])
        filter2 = reg['UW Yr'] == self.max_uw_year
        filter3 = reg['Treaty Type'] != 'FAC'
        renewed_treaties = reg.loc[(filter1) & (filter2) & (filter3)]
        renewed_treaties_cols = ['Comp',
                                 'Seq',
                                 'Short',
                                 'Treaty Type',
                                 'Layer Number']

        renewed_treaties_startcol = len(summary.columns) + 2
        renewed_treaties[renewed_treaties_cols].to_excel(
            writer,
            sheet_name='Summary for UW',
            index=False,
            startcol=renewed_treaties_startcol
        )

        # Dealing with New Treaties in Summary
        na_treaties = summary.loc[summary['Balloon ID'].isna(), 'Comp']

        # Space between end of Summary and Beginning of other data
        row = len(summary) + 3
        
        # Iterate through New Treaties and Save QS's to Excel
        for company_id in na_treaties:
            na_treaty_summary = self.get_treaties(company_id)

            na_treaty_summary.to_excel(writer,
                                       sheet_name='Summary for UW',
                                       index=False,
                                       startrow=row)

            row += len(na_treaty_summary) + 3
        
        # ---------------------------------------------------------------------
        # Formatting Summary for UW

        uw_summary_sh = writer.sheets['Summary for UW']
        
        uw_summary_sh.set_column("A:B", 10.14)
        uw_summary_sh.set_column("C:C", 31)
        uw_summary_sh.set_column("E:E", 31)
        uw_summary_sh.set_column("F:G", 14)

        
        uw_summary_sh.set_column(
            renewed_treaties_startcol,      # Comp column - right-hand side
            renewed_treaties_startcol + 1,  # Seq column - right-hand side
            6.14
        )

        uw_summary_sh.set_column(
            renewed_treaties_startcol + 2,  # Short column - right-hand side
            renewed_treaties_startcol + 2,
            34
        )

        uw_summary_sh.set_column(
            renewed_treaties_startcol + 3,  # Treaty Type col -right-hand side
            renewed_treaties_startcol + 4,  # Layer Number col -right-hand side
            13.14
        )

        writer.save()

        print(f"{file_name} created in folder {os.getcwd()}")
