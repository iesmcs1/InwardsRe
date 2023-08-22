import pandas as pd

class IREGrouping:
    def __init__(self,
                 asumed_reg_file,
                 ire_grouping_file,
                 report_date) -> None:
        self.register = self._import_assumed_reg(file_path=asumed_reg_file)
        self.ire_grouping = self._import_ire_grouping(ire_grouping_file)
        self.report_date = pd.to_datetime(report_date, dayfirst=True)
        self.max_uw_year = self.register['UW Yr'].max()
    
    @staticmethod
    def _import_assumed_reg(file_path):
        df = pd.read_excel(file_path, skiprows=6)

        # For some reason, in Dec-2020, the Column Name changed. This deals
        # with that
        if "Layer No" in df.columns:
            df.rename(columns={'Layer No': 'Layer Number'}, inplace=True)

        df['Cedant'] = df['Cedant'].str.strip()
        df['Short'] = df['Short'].str.strip()
        
        # Format Comp
        df['Comp'] = df['Comp'].apply(lambda x: '{:0>5}'.format(x))
        
        df['Balloon ID'] = df['Comp'] + df['Seq']
        df['Reference'] = df['Comp'] + df['Seq'] + df['UW Yr'].astype(str)
        
        # Set all to FAC, then see which is QS and XL
        df['Type'] = 'FAC'
        df.loc[df['Seq'].str[1:2] == '1', 'Type'] = 'QS'
        df.loc[df['Seq'].str[1:2] == '2', 'Type'] = 'XL'
        
        # Set all to C (Credit), then see which are B (Bond)
        df['B/C'] = 'C'
        df.loc[df['Seq'].str[0] == 'B', 'B/C'] = 'B'
        
        df.sort_values(
            by=['Comp', 'Seq', 'UW Yr'],
            ascending=[True, True, False],
            inplace=True)
        
        return df
    
    @staticmethod
    def _import_ire_grouping(file) -> pd.DataFrame:
        df = pd.read_csv(file,
                         sep='\t',
                         dtype={'UW_Yr': int},
                         skipfooter=1,
                         engine='python')

        # Format Comp
        df['Comp'] = df['Comp'].apply(lambda x: '{:0>5}'.format(x))
        return df
    
    def generate_detection_pivot(self, latest_uwy=None):
        """Returns pandas.DataFrame with Treaties that are ambiguous 
        and do not a have valid Grouping_ID in last Solvency II run.

        Parameters
        ----------
        latest_uwy : None or int
            if none, then it will use the highest UW Yr in
            treaty Register
        """
        # If None (False), then use highest value in UW Yr column
        if not latest_uwy:
            latest_uwy=self.max_uw_year
        
        # To keep only useful columns
        main_cols = ['Comp', 'Seq', 'UW Yr', 'B/C', 'Type', 'Reference']
        df = self.register[main_cols]

        # Need to analyse only Lastest UW Year and QS/XL. FAC not needed
        df = df.loc[(df['UW Yr'] == latest_uwy) & (df['Type'] != 'FAC')]
        
        # To count the amount of treaties belonging to QS/LX and B/C
        df_piv = df.pivot_table(
            values='Seq',
            index='Comp',
            columns=['Type', 'B/C'],
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
        df_piv.iloc[:,1:] = df_piv.iloc[:,1:].astype(int)
        
        # We only need to check if the're more than 1 QS and XL is present
        multi_QS = df_piv['Count QS'] > 1
        has_XL_B = df_piv[('XL', 'B')] > 0
        has_XL_C = df_piv[('XL', 'C')] > 0
        
        df_piv = df_piv.loc[multi_QS]
        df_piv = df_piv.loc[(has_XL_B) | (has_XL_C)]

        return df_piv
    
    def summary_for_underwriters(self, latest_uwy=None):
        """Generates table with previous UW Yr QS and XL structure.

        Parameters
        ----------
        latest_uwy : None or int
            if none, then it will use the highest UW Yr in
            treaty Register
        """
        # If None (False), then use highest value in UW Yr column
        if not latest_uwy:
            latest_uwy=self.max_uw_year
        
        grp = self.ire_grouping
        piv = self.generate_detection_pivot(latest_uwy)

        # Create Reference Column for merge()
        grp['Reference'] = (
            grp['Grouping_ID'].str[1:6]
            + grp['Seq']
            + grp['Grouping_ID'].str[-2:]
        )

        # For Summary, we only want treaties that have a Grouping ID
        has_grouping_id = grp['Comp'].isin(piv['Comp'])
        # We also want only tha latest
        most_recent_uwy = grp['UW_Yr'] == grp['UW_Yr'].max()
        
        # Filter IRE_Grouping DataFrame
        grp = grp.loc[(has_grouping_id) & (most_recent_uwy)]

        # Insert Layer Number
        grp = grp.merge(
            self.register[['Reference', 'Layer Number']],
            how='left',
            on='Reference'
        )

        try:
            # Produce DF with columns for each layer, with corresponding QS
            grp_pivoted = grp.pivot(
                index='Grouping_ID',
                columns='Layer Number',
                values='Seq'
            ).reset_index()
        except ValueError:
            # This deals with the erorr for 05401C2 0121 and 05401C2 1221
            # Both attach to layer 1 of M05401C10121
            grp.loc[
                grp[['Grouping_ID', 'Layer Number']].duplicated(keep='first'),
                'Layer Number'
            ] = grp['Layer Number'].astype(str) + 'a'

            # Produce DF with columns for each layer, with corresponding QS
            grp_pivoted = grp.pivot(
                index='Grouping_ID',
                columns='Layer Number',
                values='Seq'
            ).reset_index()

        # Balloon ID and Comp ID UWs
        grp_pivoted['Comp'] = grp_pivoted['Grouping_ID'].str[1:6]
        grp_pivoted['Balloon ID'] = grp_pivoted['Comp'] + grp_pivoted[0]

        grp_pivoted.columns.name = None

        # make sure all Comp ID's from the generate_detection_pivot() 
        # are in the final DataFrame.
        grp_pivoted = grp_pivoted.fillna("").merge(
            piv['Comp'],
            on='Comp',
            how='outer'
        )

        # Bring in Cedant name.
        grp_pivoted = grp_pivoted.merge(
            self.register[[
                'Comp', 'Cedant', 'UW Yr'
            ]].drop_duplicates(subset=['Comp', 'Cedant']),
            on='Comp',
            how='left'
        )

        # Bring in Short name
        grp_pivoted = grp_pivoted.merge(
            self.register[['Balloon ID', 'Short']].drop_duplicates(subset=['Balloon ID']),
            on='Balloon ID',
            how='left'
        )

        grp_pivoted.rename(columns={0: 'QS'}, inplace=True)

        # Organize columns
        cols = list(grp_pivoted.columns)
        cols = cols[-5:] + cols[0:-5]
        return grp_pivoted[cols]
    
    def get_treaties(self, comp_id):
        df = self.register[[
            'Comp',
            'Seq',
            'Cedant',
            'UW Yr',
            'Short',
            'Type',
            'Layer Number',
        ]]

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
        writer = pd.ExcelWriter(
            file_name,
            engine='xlsxwriter',
            date_format='dd/mm/yyyy',
            datetime_format='dd/mm/yyyy'
        )

        # Write Dataframes to Excel file
        
        ## Assumed Treaty Register
        reg = self.register
        reg.to_excel(
            writer,
            sheet_name='Treaty Register',
            index=False
        )

        ## Detection
        piv_det = self.generate_detection_pivot()
        piv_det.to_excel(
            writer,
            sheet_name='Detection',
            index=False
        )

        detection_text1 = "Treaties to be investigated."
        detection_text2 = "2 or more QS with XL."
        sheet_detection = writer.sheets['Detection']
        sheet_detection.write('J1', detection_text1)
        sheet_detection.write('J2', detection_text2)

        ## Summary for Underwriters
        summary = self.summary_for_underwriters()
        summary.to_excel(
            writer,
            sheet_name='Summary for UW',
            index=False
        )

        # List of Renewed Treaties
        filter1 = reg['Comp'].isin(piv_det['Comp'])
        filter2 = reg['UW Yr'] == self.max_uw_year
        filter3 = reg['Type'] != 'FAC'
        renewed_treaties = reg.loc[(filter1) & (filter2) & (filter3)]
        renewed_treaties_cols = ['Comp',
                                 'Seq',
                                 'Short',
                                 'Type',
                                 'Layer Number']

        renewed_treaties[renewed_treaties_cols].to_excel(
            writer,
            sheet_name='Summary for UW',
            index=False,
            startcol=len(summary.columns) + 2
        )

        # Dealing with New Treaties in Summary
        na_treaties = summary.loc[summary['Balloon ID'].isna(), 'Comp']

        # Space between end of Summary and Beginning of other data
        row = len(summary) + 3
        
        # Iterate through New Treaties and Save QS's to Excel
        for company_id in na_treaties:
            na_treaty_summary = self.get_treaties(company_id)

            na_treaty_summary.to_excel(
            writer,
            sheet_name='Summary for UW',
            index=False,
            startrow=row
        )

            row += len(na_treaty_summary) + 3
        
        # ---------------------------------------------------------------------
        # Format Summary for UW

        ws_suw = writer.sheets['Summary for UW']

        # A = Comp
        ws_suw.set_column('A:A', 8.57)
        # C = Cedant
        ws_suw.set_column('C:C', 31)
        # D = UW Yr
        ws_suw.set_column('D:D', 8.57)
        # E = Short
        ws_suw.set_column('E:E', 25)
        # F = Grouping ID and Type
        ws_suw.set_column('E:E', 14)
        # G = QS and Layer Number.
        # Used 12.86 because of Layber Number col header
        ws_suw.set_column('G:G', 12.86)
        # H to L = Layer Numbers (1, 2, ...)
        ws_suw.set_column('H:L', 8.57)

        writer.save()

        print(f"{file_name} was created.")