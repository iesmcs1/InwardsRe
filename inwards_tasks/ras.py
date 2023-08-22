from typing import Dict, Tuple
from .treaty_register import Register
import pandas as pd
import numpy as np

REGISTER_COLS = ['CONTRACT_ID',
                 'Balloon ID',
                 'Comp',
                 'Seq',
                 'UW Yr',
                 'Cedant',
                 'Short',
                 'Broker Name',
                 'Pd Beg',
                 'Pd End',
                 'Type/Form',
                 'UPR',
                 'Signed %',
                 '100% EPI',
                 'EPI is Rev EPI or EPI',
                 'Est Total Costs %',
                 'UWLr%',
                 'Commission',
                 '100% Full Limit',
                 'Limit']

# Dictonary with all columns that will be in Excel file, with 
# corresponding formatting
COLS_FORMATTING = {
    'Ref': {
        'col_width': 12,
        'formatting': None
    },
    'CONTRACT_ID': {
        'col_width': 15,
        'formatting': None
    },
    'Comp': {
        'col_width': 8,
        'formatting': None
    },
    'Balloon ID': {
        'col_width': 11,
        'formatting': None
    },
    'UW Yr': {
        'col_width': 8,
        'formatting': None
    },
    'Cedant': {
        'col_width': 32,
        'formatting': None
    },
    'Short': {
        'col_width': 24,
        'formatting': None
    },
    'Type/Form': {
        'col_width': 12,
        'formatting': None
    },
    'Pd Beg': {
        'col_width': 10,
        'formatting': None
    },
    'Pd End': {
        'col_width': 10,
        'formatting': None
    },
    'EPI-Liability Ratio': {
        'col_width': 17,
        'formatting': '0.00%'
    },
    'UPR': {
        'col_width': 8,
        'formatting': None
    },
    'Commission': {
        'col_width': 8,
        'formatting': '0.00%'
    },
    '100% EPI': {
        'col_width': 14,
        'formatting': '#,##0'
    },
    '100% Full Limit': {
        'col_width': 16,
        'formatting': '#,##0'
    },
    'Limit': {
        'col_width': 16,
        'formatting': '#,##0'
    },
    'ROL': {
        'col_width': 9,
        'formatting': '0.00%'
    },
    'Comments': {
        'col_width': 37,
        'formatting': None
    },
    'Net Profit': {
        'col_width': 11,
        'formatting': '#,##0'
    },
    'Broker Name' : {
        'col_width': 29,
        'formatting': None
    },
    'EPI is Rev EPI or EPI' : {
        'col_width': 19,
        'formatting': '#,##0'
    },
    '% of Total' : {
        'col_width': 12,
        'formatting': '0.00%'
    }
}

class RAS(Register):
    def __init__(self,
                 filepath: str,
                 report_date: str,
                 min_epi_liab_ratio : float,
                 min_xl_rol : float,
                 max_commission : dict,
                 ecap_filepath : str = None) -> None:
        
        Register.__init__(self,
                          filepath=filepath,
                          adjust_register=False,
                          report_date=report_date)
        
        self._ras_register = None
        
        self.min_epi_liabi_ratio = min_epi_liab_ratio
        self.min_xl_rol = min_xl_rol
        
        if 'bond' in max_commission and 'cred' in max_commission:
            self.max_commission = max_commission
        else:
            raise KeyError('Dictionary must contain "bond" and "cred" keys.')
        
        if ecap_filepath:
            self.ecap = self._import_ecap_file(ecap_filepath)

        self.checks = {}

        # Running Functions because this is what we want every time.
        self.epi_liab_ratio()
        self.xl_rol()
        self.commission()
        self.return_on_capital()
        self.concentration(by='broker')
        self.concentration(by='cedant')
    
    @property
    def ras_register(self) -> pd.DataFrame:
        df = self.register.copy()
        
        # Exclude Atradius Actuarial Treaty
        df = df.loc[df['Comp'] != '35003']

        # Columns to percentage
        cols_to_percent = ['Commission',
                           'Est Total Costs %',
                           'UWLr%']

        for col in cols_to_percent:
            df[col] /= 100

        df = self.normalize_broker_name(data = df).copy()

        df.loc[df['Type/Form'] == 'XL', '100% EPI'] = (df['Signed EPI']
                                                       / df['Signed %'])
        
        return df[REGISTER_COLS].copy()
    
    @staticmethod
    def normalize_broker_name(data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Using startswith below was producing errors and creating more code
        # because it doesn't accept regex
        cond_list = [
            df['Broker Name'].isna(),
            df['Broker Name'].str.startswith('Guy', na=False) | df['Broker Name'].str.startswith('GUY', na=False),
            df['Broker Name'].str.startswith('Aon', na=False) | df['Broker Name'].str.startswith('AON', na=False),
            df['Broker Name'].str.startswith('Willis', na=False) | df['Broker Name'].str.startswith('WILLIS', na=False),
            df['Broker Name'].str.startswith('JLT', na=False),
            df['Broker Name'].str.startswith('ASIA REINSURANCE', na=False)
        ]

        choice_list = [
            'No Broker',
            'Guy Carpenter & Ltd.',
            'Aon',
            'Willis',
            'JLT',
            'ASIA REINSURANCE BROKERS'
        ]

        df['Broker Name'] = np.select(cond_list,
                                      choice_list,
                                      df['Broker Name'])

        return df
    
    def filter_register(self,
                        pd_beg: bool = False,
                        pd_end: bool = False) -> pd.DataFrame:
        """Filter Register DataFrame using report_date and Pd Beg and Pd End
        columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame to be filtered
        pd_beg : bool, optional
            If True, then filters dates in Pd Beg that are less or equal than 
            report_date (`df['Pd Beg'] <= self.report_date`), by default False
        pd_end : bool, optional
            If True, then filters dates in Pd End that are greater or equal
            than report_date (`df['Pd Beg'] >= self.report_date`),
            by default False

        Returns
        -------
        pd.DataFrame
            copy of filtered Register (DataFrame)
        """
        df = self.ras_register

        if pd_beg:
            df = df.loc[df['Pd Beg'] <= self.report_date]
        
        if pd_end:
            df = df.loc[df['Pd End'] >= self.report_date]
        
        return df

    def epi_liab_ratio(self, min_ratio=None):
        if min_ratio is None:
            min_ratio = self.min_epi_liabi_ratio
        
        # Filter Register
        df = self.filter_register(pd_beg=True, pd_end=True)

        # Exclude XL and FAC
        df = df.loc[~df['Type/Form'].isin(['XL', 'FAC', 'FACX'])]

        df['EPI-Liability Ratio'] = df['100% EPI'] / df['100% Full Limit']

        # Filtering only EPI / Liability < min_ratio
        df = df.loc[df['EPI-Liability Ratio'] < min_ratio]
        df.sort_values(by=['Pd End', 'Cedant'], inplace=True)

        df['Comments'] = ''

        df.rename(columns={'CONTRACT_ID': 'Ref'}, inplace=True)

        cols_to_use = ['Ref',
                       'Balloon ID',
                       'UW Yr',
                       'Cedant',
                       'Short',
                       'Pd Beg',
                       'Pd End',
                       'EPI-Liability Ratio',
                       '100% EPI',
                       '100% Full Limit',
                       'Comments']
        
        self.checks['EPI-LIABI Ratio'] = df[cols_to_use]

        return df[cols_to_use]

    def xl_rol(self, min_rol=None):
        if min_rol is None:
            min_rol = self.min_xl_rol
        
        # Filter Register 
        df = self.filter_register(pd_beg=True, pd_end=True)

        # Only XL
        df_xl = df.loc[df['Type/Form'] == 'XL'].copy()

        cols_to_use = ['Comp', 'UW Yr', 'Cedant', 'Type/Form',
                       'Pd Beg', 'Pd End', '100% EPI', 'Limit']
        
        # For some reason, 100% EPI is treated as object if not converted
        # to float or int
        for col in ['100% EPI', 'Limit']:
            df_xl[col] = df_xl[col].astype(np.int64)

        group_by_cols = ['Comp',
                         'UW Yr',
                         'Cedant',
                         'Type/Form',
                         'Pd Beg',
                         'Pd End']

        # Groupby to sum 100% EPI and Limit
        df_grp = df_xl[cols_to_use].groupby(by=group_by_cols,
                                            as_index=False).sum()
        
        df_grp.sort_values(by=['Pd Beg', 'Comp'], inplace=True)

        df_grp['ROL'] = df_grp['100% EPI'].div(df_grp['Limit'])

        # Add Ref column to be able to add comments
        df_grp.insert(0, 'Ref', df_grp['Comp'] + df_grp['UW Yr'].astype(str))
        
        df_min_rol = df_grp.loc[df_grp['ROL'] < min_rol]

        self.checks['XL-ROL'] = df_min_rol

        return df_min_rol

    def commission(self, max_for_bond=None, max_for_cred=None):
        if max_for_bond is None:
            max_for_bond = self.max_commission['bond']
        
        if max_for_cred is None:
            max_for_cred = self.max_commission['cred']

        # Filter Register 
        df = self.filter_register(pd_beg=True, pd_end=True)

        df['UPR'] = df['UPR'].str[0]
        # Use only Bond or Credit. No XL
        df = df.loc[df['UPR'].str[0].isin(['B', 'C'])]
        use_cols = ['CONTRACT_ID', 'Balloon ID', 'UW Yr', 'Pd Beg', 'Pd End',
                    'Cedant', 'Short', 'UPR', 'Commission']
        df = df[use_cols]
        
        bond_comm = ((df['UPR'].str[0] == 'B')
                    & (df['Commission'] > max_for_bond))
        cred_comm = ((df['UPR'].str[0] == 'C')
                    & (df['Commission'] > max_for_cred))
        
        self.checks['Commission'] = df.loc[(bond_comm) | (cred_comm)]
        
        return df.loc[(bond_comm) | (cred_comm)]
    
    def return_on_capital(self):
        
        # Filter Register 
        df = self.filter_register(pd_beg=False, pd_end=True)

        # Evaluate only new Treaties signed in the Last 3 months
        df = df.loc[
            df['Pd Beg'] >= self.report_date - pd.tseries.offsets.MonthBegin(3)
        ]

        # Evaluate only QS
        df = df.loc[df['Seq'].str[1] == "1"]

        df['CR'] = (df['Est Total Costs %'] + df['UWLr%'])/100
        df['Net Profit'] = (1 - df['CR']) * df['EPI is Rev EPI or EPI']

        df['Net Profit'] = df['Net Profit'].astype(np.int64)

        use_cols = ['Comp', 'Cedant', 'Net Profit']

        df_grouped = df[use_cols].groupby(by=['Comp', 'Cedant'],
                                          as_index=False).sum()

        try:
            # Try to merge in ECap numbers for each Cedant
            df_with_ecap = df_grouped.merge(self.ecap, how='left', on='Comp')

            # Divide Net Profit by ECap to find Return On Capital
            df_with_ecap['ROC'] = df_with_ecap['Net Profit'] / df_with_ecap['ECAP']
            df_with_ecap.sort_values(by='ROC', inplace=True)
            self.checks['Treaty-ROC'] = df_with_ecap
            return df_with_ecap
        except AttributeError:
            # If no ECap file was loaded into model, then we don't divide 
            # Net Profit by ECap, and just return absolute numbers
            self.checks['Treaty-ROC'] = df_grouped
            return df_grouped

    
    def _import_ecap_file(self, filepath):
        df = pd.read_csv(filepath, sep='\t', dtype={'COMP':str})

        if 'COMP' not in df.columns:
            df['Comp'] = df['CUSTOMER_ID'].str[:5]
        
        df.rename(columns={'COMP':'Comp',
                           'EC_CONSUMPTION_ND': 'ECAP'},
                  inplace=True)
        
        df_grp = df[['Comp', 'ECAP']].groupby('Comp').sum()

        return df_grp

    def concentration(self, by):
        """Return concentration by Cedant or Broker

        Parameters
        ----------
        by : {'broker' or 'cedant'}
        """
        epi_col = 'EPI is Rev EPI or EPI'

        col = None
        if by == 'broker':
            col = 'Broker Name'
        elif by == 'cedant':
            col = 'Cedant'
        else:
            raise ValueError(
                'by parameter "{}" is not "broker" or "cedant".'.format(by)
            )

        # Filter Register 
        df = self.filter_register(pd_beg=False, pd_end=True)

        df_group = df[[col, epi_col]].groupby(by=col, as_index=False).sum()

        df_group['% of Total'] = df_group[epi_col] / df_group[epi_col].sum()

        df_group.sort_values(by='% of Total', ascending=False, inplace=True)

        self.checks[f'Concentration-{by}'] = df_group

        return df_group

    def apply_comments(self,
                       epi_liab_ratio: dict = None,
                       xl_rol: dict = None):
        """Apply comments to dataframe specified in the parameters.

        Parameters
        ----------
        epi_liab_ratio : dict, optional
            dictionary containing column name as key, and another dictionary
            in the value, containing the corresponding values and comments,
            by default None
        xl_rol : dict, optional
            [description], by default None
        
        Examples
        --------
        >>> epi_liab_comments = {
            'CONTRACT_ID': X{
                '12345B1 0121': '10% on pink sheet',
                '02126C1 0421': 'top up to main treaty',
            },
            'Cedant': {
                'CEDANT_NAME' : 'comment goes here'
            }
        }
        """ 
        if epi_liab_ratio:
            self._apply_comments_helper('EPI-LIABI Ratio', epi_liab_ratio)
        
        if xl_rol:
            self._apply_comments_helper('XL-ROL', xl_rol)
    
    def _apply_comments_helper(self, check_name, comments):
        # If there are XL ROL comments
        df_xl_rol = self.checks[check_name]
        print(f'Applying comments on sheet {check_name}:')
        for ref, comment in comments.items():
            # Declare filter to be used to apply comment
            ref_filter = df_xl_rol['Ref'] == ref
            
            # Apply comment
            df_xl_rol.loc[ref_filter, 'Comments'] = comment

            try:
                # Print statement informing about the change.
                print_row_info = df_xl_rol.loc[ref_filter, 'Cedant'].values[0]
                print(f"* Comment applied to {ref}-{print_row_info}.")
            except IndexError:
                print(f'#### Could not find Reference {ref}.')

        
    def _apply_epi_liab_comments(self, col_name: str, comment_val: dict):
        # defined as df for less typing
        df = self.checks['EPI-LIABI Ratio']
        # Iterate through commment_val to apply comments
        for k, v in comment_val.items():
            if df[col_name].isin([k]).any():
                # if key is in column, then apply comment
                df.loc[df[col_name] == k, 'Comments'] = v
                print(f'Comment applied to {k} in column {col_name}.')
            else:
                # else print and do nothing
                print(f'{k} not found in column {col_name}.')
        return None

    def export_to_excel(self):
        excel_file = "RAS-UW {}Q{}.xlsx".format(self.report_date.year,
                                                self.report_date.quarter)

        writer = pd.ExcelWriter(excel_file,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')
        
        #self._add_xl_rol_sheet(excel_writer=writer)
        #self._add_epi_liabi_ratio_sheet(excel_writer=writer)

        for check_name, df in self.checks.items():
            self._add_df_as_table_in_excel(excel_writer=writer,
                                           df=df,
                                           sheet_name=check_name)
            
            formatting_dict = self.get_cols_formatting(df=df)

            self._format_sheet(excel_writer=writer,
                               sheet_name=check_name,
                               formatting_dict=formatting_dict)
        
        print("File {} created.".format(excel_file))

        writer.save()
    
    @staticmethod
    def _add_df_as_table_in_excel(excel_writer: pd.ExcelWriter,
                                  df: pd.DataFrame,
                                  sheet_name: str):

        # Write DataFrame to Excel
        df.to_excel(excel_writer,
                    index=False,
                    sheet_name=sheet_name)

        # Format as Excel Table
        (max_row, max_col) = df.shape
        column_settings = [{'header': column} for column in df.columns]
        excel_writer.sheets[sheet_name].add_table(
            0, 0, max_row, max_col - 1, {'columns': column_settings}
        )

        return None
    
    @staticmethod
    def get_cols_formatting(df: pd.DataFrame) -> dict:
        excel_col = 'A'
        formatting_dict = {}
        for col in df.columns:
            # Range to format (e.g.: A:A)
            excel_col_rng = f"{excel_col}:{excel_col}"
            
            # Add excel_col_rng as key, and formatting dict as value
            formatting_dict[excel_col_rng] = COLS_FORMATTING[col]

            # Add one as to have letter after excel_col.
            # If it's first time looping, then after line below, excel_col
            # will be 'B', and so on.
            excel_col = chr(ord(excel_col) + 1)

        return formatting_dict

    @staticmethod
    def _format_sheet(excel_writer: pd.ExcelWriter,
                      sheet_name: str,
                      formatting_dict: Dict[str, Dict[str, str]]):
        """Apply formatting to columns in formatting_dict.

        Parameters
        ----------
        excel_writer : pd.ExcelWriter
            Excel writer.
        sheet_name : str
            Sheet name
        formatting_dict : Dict[str, Dict[str, str]]
            This will always have the following format:
                {'A:A': {
                    width_col: 12,
                    formatting: '#,##0
                }}
                key : column range

        Returns
        -------
        None
            Applies formatting to worksheet.
        """
        wb = excel_writer.book
        ws = excel_writer.sheets[sheet_name]

        for range, format in formatting_dict.items():
            if format['formatting'] is None:
                col_num_format = None
            else:
                col_num_format = wb.add_format({
                    'num_format': format['formatting']})


            ws.set_column(range,
                          format['col_width'],
                          col_num_format)
        
        return None
