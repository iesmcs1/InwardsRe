import pandas as pd
import numpy as np
import os

from markdown2 import Markdown

try:
    from .market_risk import MarketRisk
    from .market_risk_data import MarketRiskData
    from ..utils import get_report_period
except ImportError:
    import sys
    from market_risk import MarketRisk
    from market_risk_data import MarketRiskData
    if os.getcwd().endswith('InwardsRe'):
        sys.path.insert(0, os.path.abspath('./modules'))
    from utils import get_report_period


class CurrencyData(MarketRiskData):
    def __init__(self, folder_path: str, report_date: str) -> None:

        MarketRiskData.__init__(self,
                                folder_path=folder_path,
                                report_date=report_date)

        self._DATA_NAME = 'CURRENCY'

        # Currency Data is a little more complicated because it involves
        # 2 files. Therefore, I initialize a dict to hold both data sources.
        self.data = {}

        self.import_data()
        self.calculate_scr()

    def import_data(self):
        self._import_balance_sheet()
        
        self._import_tp_position()
        self._process_tp_position()

    def _import_balance_sheet(self):
        filepath = os.path.join(self.folder_path, 'RW_FX_BALANCE_SHEET.txt')
        df = pd.read_csv(filepath, sep='\t')
        # In case Excel file has extra lines, and they are imported with NaN
        df.dropna(axis='index', how='all', inplace=True)

        self.data['BALANCE_SHEET'] = df

        return None

    def _import_tp_position(self):
        filepath = os.path.join(self.folder_path,
                                'IR_TP_POSITION_GRANULAR_CURRENCY.txt')

        df = pd.read_csv(filepath, sep='\t')

        # In case Excel file has extra lines, and they are imported with NaN
        df.dropna(axis='index', how='all', inplace=True)

        self.data['TP_POSITION'] = df

        return None
    
    def _process_tp_position(self):
        self._add_ref_column()
    
    def _add_ref_column(self):
        df = self.data['TP_POSITION']
        df['REFERENCE'] = (df['FK_RW_MODEL_TYP'] + '-'
                           + df['FK_RW_PRODUCT']  + '-'
                           + df['FK_RW_LINE_OF_BUSINESS'].astype(str)  + '-'
                           + df['FK_RW_FINANCIAL_CF_TYP']  + '-'
                           + df['IN_OUT_FLOW_IND'])
        
        self.data['TP_POSITION'] = df

        return None
        

    def calculate_scr(self):
        self._merge_tp_and_bs_data()
        self._insert_shock_parameters()
        self._calculate_scr()

        return None

    def _pivot_for_scr_calculation(self, data: str) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        data : str
            'tp' for TP_POSITION or 'bs' for BALANCE_SHEET.

        Returns
        -------
        [type]
            [description]
        """
        if data == 'tp':
            data_key = 'TP_POSITION'
            use_cols = ['FK_RW_ISO_CURRENCY',
                        'FK_RW_FINANCIAL_CF_TYP',
                        'NPV_INTEREST_RATE_VAL']
        elif data == 'bs':
            use_cols = ['PK_FK_RW_ISO_CURRENCY_ORIGINAL',
                        'PK_FK_CD_ASSET_LIABILITY_IND',
                        'SOLVENCY_II_VALUATION']
            data_key = 'BALANCE_SHEET'
        else:
            raise ValueError(f"data must be 'tp' or 'bs'. {data} was passed.")

        df = self.data[data_key]

        # Index 0 is Currency, Index 1 is Type, Index 2 is the Valuation
        df_pvt = df[use_cols].pivot_table(values=use_cols[2],
                                          index=use_cols[0],
                                          columns=use_cols[1],
                                          aggfunc=np.sum)

        # Remove row axis name
        df_pvt.rename_axis(None, inplace=True)
        # Remove column axis name
        df_pvt.rename_axis(None, axis='columns', inplace=True)

        return df_pvt

    def _merge_tp_and_bs_data(self):
        df_bs = self._pivot_for_scr_calculation(data='bs')
        df_tp = self._pivot_for_scr_calculation(data='tp')

        df = df_bs.merge(df_tp, right_index=True, left_index=True, how='outer')

        self.data['SCR'] = df

        return None

    def _insert_shock_parameters(self):
        no_shock_factor_curr = ['EUR', 'ATS', 'BEF', 'CYP', 'DEM',
                                'EEK', 'ESP', 'FIM', 'FRF', 'GRD',
                                'IEP', 'ITL', 'LTL', 'LUF', 'LVL',
                                'MTL', 'NLG', 'PTE', 'SIT', 'SKK']

        df = self.data['SCR']

        condlist = [df.index.isin(no_shock_factor_curr),
                    df.index == 'BGN',
                    df.index == 'DKK',
                    df.index == 'XOF',
                    df.index == 'XAF',
                    df.index == 'KMF']

        choicelist = [0,
                      0.0181,
                      0.0039,
                      0.0218,
                      0.0196,
                      0.02]

        df['SHOCK_FACTOR'] = np.select(condlist,
                                       choicelist,
                                       default=0.25)

        return None

    def _calculate_scr(self):
        df = self.data['SCR']
        df.fillna(0, inplace=True)
        df['ABS_CASHFLOW'] = (
            df['ASSET'] - df['LIABI'] - df['PCO'] - df['PR']
        ).abs()
        df['SCR'] = df['ABS_CASHFLOW'] * df['SHOCK_FACTOR']

        return None
    
    def get_balance_sheet_by_natural_acc(self, currency: str) -> pd.DataFrame:
        df_raw = self.data['BALANCE_SHEET']
        
        # Create filter by specified Currency
        currency_filter = df_raw['PK_FK_RW_ISO_CURRENCY_ORIGINAL'] == currency
        
        # Filter DataFrame
        df = df_raw.loc[currency_filter]

        cols_for_grouping = ['PK_FK_RW_NATURAL_ACCOUNT',
                             'SOLVENCY_II_VALUATION']

        df_grp = df[cols_for_grouping].groupby(
            by='PK_FK_RW_NATURAL_ACCOUNT',
            as_index=False
        ).sum()

        df_grp.rename(columns={'PK_FK_RW_NATURAL_ACCOUNT':'NATURAL_ACCOUNT'},
                      inplace=True)
        
        df_grp['NAT_ACC_TYPE'] = np.where(
            df_grp['NATURAL_ACCOUNT'].str[4] == 'L',
            'LIAB',
            'ASSET'
        )
        
        return df_grp
    
    def get_tp_by_reference(self, currency: str) -> pd.DataFrame:
        df_raw = self.data['TP_POSITION']

        # Create filter by specified Currency
        currency_filter = df_raw['FK_RW_ISO_CURRENCY'] == currency

        df = df_raw.loc[currency_filter]

        cols_for_grouping = ['REFERENCE', 'NPV_INTEREST_RATE_VAL']

        df_grp = df[cols_for_grouping].groupby(
            by='REFERENCE',
            as_index=False
        ).sum()
        
        return df_grp
    
    def get_currency_summary(self, currency: str) -> pd.DataFrame:
        asset_sum = self._get_currency_balance_sheet_sum(
            currency=currency, type='ASSET')
        
        liab_sum = self._get_currency_balance_sheet_sum(
            currency=currency, type='LIAB')
        
        pco_sum = self._get_tp_sum(currency=currency, type='PCO')
        pr_sum = self._get_tp_sum(currency=currency, type='PR')

        df = pd.DataFrame(
            {self.report_period: [asset_sum, liab_sum, pco_sum, pr_sum]},
            index=['Assets', 'Non-TP Liab', 'PCO', 'PR']
        )

        return df

    def _get_currency_balance_sheet_sum(self, currency: str, type: str):
        """[summary]

        Parameters
        ----------
        currency : str
            Currency being analysed
        type : str
            'asset' or 'liab'.

        Returns
        -------
        int or float
            The sum of the type chosen in the balance sheet.
        """
        df = self.get_balance_sheet_by_natural_acc(currency=currency)
        df_sum = df.groupby('NAT_ACC_TYPE').sum()

        return df_sum.loc[type, 'SOLVENCY_II_VALUATION']
    
    def _get_tp_sum(self, currency: str, type: str):
        df = self.get_tp_by_reference(currency=currency)
        filter_type = df['REFERENCE'].str.contains(type)

        return df.loc[filter_type, 'NPV_INTEREST_RATE_VAL'].sum()


    
class CurrencyRisk(MarketRisk):

    NATURAL_ACCOUNT_DESCRIPTIONS = {
        'DCL_50900': 'Shares AFS',
        'DCL_50920': 'Bonds AFS',
        'DCL_51100FX': 'Deposits withheld by ceding companies',
        'DCL_51110': 'Bank deposits',
        'DCL_51120': 'Restricted financial investments',
        'DCL_53211FX': 'Cash in bank CIA',
        'DCL_71761': 'Accounts payable',
        'DCL_AD001': 'Accounts receivable on insurance and reinsurance business',
        'DCL_AE005': 'Cash and cash equivalents',
        'DCL_AE011': 'DCL_AE011',
        'DCL_AE062': 'Miscellaneous Assets and Accruals - Non-Technical',
        'DCL_LF001': 'Accounts payable on insurance and reinsurance business',
        'DCL_LF002': 'Other accounts payable',
        'DCL_LG044': 'Current income tax liabilities',
        'DCL_LG072': 'Miscellaneous Liabilities and Accruals - Non-Technical',
        'DCL_AE040': 'Current income tax receivables',
        'DCL_LG006': 'Other taxes',
        'DCL_51109': 'Deposits withheld by ceding companies IC',
        'DCL_62000': 'Subordinated loans',
        'DCL_71100': 'Deposits received from reinsurers',
        'DCL_AE014': 'Buildings Own Use',
        'DCL_AE041': 'Deferred tax asset',
        'DCL_LG008': 'Suspense accounts total',
        'DCL_LG045': 'Deferred income tax liabilities'
    }

    def __init__(self) -> None:
        MarketRisk.__init__(self)

        self.market_risk_type = 'currency'
    
    def import_and_process_sourcedata_files(self,
                                            report_date: str,
                                            sourcedata_folder: str = None):

        # import and process Equity Data
        df = CurrencyData(folder_path=sourcedata_folder,
                          report_date=report_date)

        # Append Equity Data Class to Equity Risk Class
        self.market_risk_data = df
        
        print("{} data was added.".format(get_report_period(report_date)))

        return None
    
    def calculate_quarter_movements(self) -> pd.DataFrame:
        df_old = self.comparison_data['old'].data['SCR']
        df_new = self.comparison_data['new'].data['SCR']

        df = df_old.merge(df_new,
                          how='outer',
                          left_index=True,
                          right_index=True,
                          suffixes=(f"-{self.periods['old']}",
                                    f"-{self.periods['new']}"))

        # Since the Diff cannot be broken down into smaller parts,
        # Decided to calculate the Diff here
        df['Diff-SCR'] = (df[f"SCR-{self.periods['new']}"]
                          - df[f"SCR-{self.periods['old']}"])

        df['Diff-ABS_CASHFLOW'] = (df[f"ABS_CASHFLOW-{self.periods['new']}"]
                                   - df[f"ABS_CASHFLOW-{self.periods['old']}"])

        df.sort_values(by='Diff-SCR', ascending=False, inplace=True)

        return df
    
    def get_natural_account_comparison(self, currency) -> pd.DataFrame:
        df_old = self.comparison_data['old'].get_balance_sheet_by_natural_acc(
            currency=currency)

        df_new = self.comparison_data['new'].get_balance_sheet_by_natural_acc(
            currency=currency)

        df = df_old.merge(df_new,
                          how='outer',
                          on=['NATURAL_ACCOUNT', 'NAT_ACC_TYPE'],
                          suffixes=(f"-{self.periods['old']}",
                                    f"-{self.periods['new']}"))
        
        df['NATURAL_ACCOUNT_DESC'] = df['NATURAL_ACCOUNT'].map(
            self.NATURAL_ACCOUNT_DESCRIPTIONS)

        # If we don't sort, NAT_ACC_TYPE will be between the amount columns
        df_ordered = df[df.columns.sort_values()].fillna(0)
        df_ordered['Diff'] = (
            df_ordered[f'SOLVENCY_II_VALUATION-{self.periods["new"]}']
            - df_ordered[f'SOLVENCY_II_VALUATION-{self.periods["old"]}']
        )

        return df_ordered.sort_values(by='NATURAL_ACCOUNT', ascending=True)
    
    def get_tp_comparison(self, currency) -> pd.DataFrame:
        df_old = self.comparison_data['old'].get_tp_by_reference(
            currency=currency)

        df_new = self.comparison_data['new'].get_tp_by_reference(
            currency=currency)

        df = df_old.merge(df_new,
                          how='outer',
                          on='REFERENCE',
                          suffixes=(f"-{self.periods['old']}",
                                    f"-{self.periods['new']}"))

        df['Diff'] = (
            df[f'NPV_INTEREST_RATE_VAL-{self.periods["new"]}']
            - df[f'NPV_INTEREST_RATE_VAL-{self.periods["old"]}']
        )

        return df.sort_values(by='Diff', ascending=False)

    def get_currency_summary_table(self,
                                   currency: str,
                                   total_row: bool = False) -> pd.DataFrame:
        """Return DataFrame with currency's TP and Non-TP breakdown.

        Parameters
        ----------
        currency : str
            currency to be analysed
        total_row : bool, optional
            If True, add total row to table, by default False

        Returns
        -------
        pd.DataFrame
            Return a DataFrame with TP and Non-TP values, divided by
            previous quarter, current quarter and the delta between quarters
        """        
        df_old = self.comparison_data['old'].get_currency_summary(currency)
        df_new = self.comparison_data['new'].get_currency_summary(currency)
        df = pd.concat([df_old, df_new], axis=1)

        df['Diff'] = df[self.periods['new']] - df[self.periods['old']]
    
        if total_row:
            df.loc['Total'] = (
                df.loc['Assets']
                - (df.loc['Non-TP Liab'] + df.loc['PCO'] + df.loc['PR'])
            )

        return df
    
    def get_movement_summary_text(self,
                                  result_interpretation_text: bool = False,
                                  to_html: bool = False) -> str:

        text = self._get_largest_movements(to_html=to_html)

        for curr in self._get_key_currencies().index:
            text += self.get_currency_movement_explanation(
                curr, to_html=to_html)

        if result_interpretation_text:
            text += self._get_results_interpretation_text(to_html=to_html)

        return text
    
    def _get_key_currencies(self) -> pd.DataFrame:
        """Return DataFrame with currencies with largest SCR movement.

        Returns
        -------
        pd.DataFrame
            return DataFrame where the index is made of 3 main currencies
            with the same sign as the SCR movement, and 1 last currency 
            that is offsetting the movement.
        """        
        df = self.calculate_quarter_movements()

        if df.sum()['Diff-SCR'] > 0:
            df_large = df.nlargest(3, columns='Diff-SCR')
            df_large = pd.concat([df_large,
                                  df.nsmallest(1, columns='Diff-SCR')])
        elif df.sum()['Diff-SCR'] < 0:
            df_large = df.nsmallest(3, columns='Diff-SCR')
            df_large = pd.concat([df_large,
                                  df.nlargest(1, columns='Diff-SCR')])
        else:
            print("No movement detected between quarters.")
            return None

        return df_large

    def _get_largest_movements(self, to_html: bool = False) -> str:
        df = self._get_key_currencies()

        text = "Largest differences in SCR coming from:\n"

        # First 3 rows are same sign as total SCR movement.
        # Last row is the offset amount
        counter = 0
        for ind, row in df.iterrows():
            if counter <= 2:
                text += "- {} (EUR {:+,.1f}m),\n".format(
                    ind,
                    # to show value in millions
                    row['Diff-SCR'] / 1e6
                )
            else:
                text += "- offset by {} (EUR {:+,.1f}m).\n\n".format(
                    ind,
                    row['Diff-SCR'] / 1e6
                )

            counter += 1

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def get_currency_movement_explanation(self,
                                          currency: str,
                                          to_html: bool = False) -> str:
        # Get the Largest QoQ movement in Assets, Non-TP Liab, PCO and PR
        df = self.get_currency_summary_table(currency=currency).nlargest(
            1,
            columns='Diff'
        )

        if df.index == 'Assets':
            df_breakdown = self.get_natural_account_comparison(
                currency=currency)
            breakdown_filter = df_breakdown['NAT_ACC_TYPE'] == 'ASSET'
            currency_movement_side = 'Assets'
        elif df.index == 'Non-TP Liab':
            df_breakdown = self.get_natural_account_comparison(
                currency=currency)
            breakdown_filter = df_breakdown['NAT_ACC_TYPE'] == 'LIAB'
            currency_movement_side = 'Liability'
        elif df.index in ['PCO', 'PR']:
            df_breakdown = self.get_tp_comparison(currency=currency)
            breakdown_filter = df_breakdown['REFERENCE'].str.contains(
                df.index[0])
            currency_movement_side = f'{df.index[0]}'

        # Filter DF with only the most significant movement
        df_breakdown_filtered = df_breakdown.loc[breakdown_filter]

        if self._currency_scr_increase(currency):
            # If it's an increase, return largest 3 values
            df_breakdown_top = df_breakdown_filtered.nlargest(
                3, columns='Diff')
        else:
            # If it's a decrease, return smallest 3 because it's negative
            df_breakdown_top = df_breakdown_filtered.nsmallest(
                3, columns='Diff')

        text = f"From {currency} ({currency_movement_side}):\n"

        if 'NATURAL_ACCOUNT' in df_breakdown_top.columns:
            # If Natural Account, then we need to return account description
            # for clarity
            for ind, row in df_breakdown_top.iterrows():
                text += "- {} ({}): {:+,.0f}\n".format(
                    row['NATURAL_ACCOUNT_DESC'],
                    row['NATURAL_ACCOUNT'],
                    row['Diff']
                )
        else:
            for ind, row in df_breakdown_top.iterrows():
                text += "- {}: {:+,.0f}\n".format(
                    row['REFERENCE'],
                    row['Diff']
                )

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text

    def _currency_scr_increase(self, currency: str) -> bool:
        scr_movement = self._get_key_currencies().loc[currency, 'Diff-SCR']

        if scr_movement > 0:
            # If SCR movement is positive, then return True
            return True
        # Else, return False, since it's decreasing
        return False

    @staticmethod
    def _get_results_interpretation_text(to_html: bool = False) -> str:
        text = "\nFor 'Assets' and 'Liability', it's from the Non-TP side of the accounts.\n"
        text += "For 'PCO' and 'PR':\n"

        text += '- BO: Bonding\n'
        text += '- CI: Credit Insurance\n'
        text += '- IR: Inward Reinsurance\n'
        text += '- ICP: Instalment Credit Protection\n'
        text += '- SP: Special Products\n\n'

        text += "- CI: Credit Insurance\n"
        text += "- BO: Bonding\n\n"

        text += "- 21: Credit and suretyship proportional reinsurance\n"
        text += "- 28: Non-proportional property reinsurance\n\n"

        text += "- PCO: Provision Claims Outstanding\n"
        text += "- PR: Premium Reserve\n\n"

        text += "- I: In flow\n"
        text += "- O: Out flow\n"

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text

    def generate_excel_report(self, open_file: bool = False):
        filename = self._get_report_name('xlsx')
        writer = pd.ExcelWriter(filename,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')

        # Create Sheet with SCR Summary
        self._scr_summary_to_excel(excel_writer=writer)

        self._quarter_over_quarter_movements_to_excel(excel_writer=writer)

        self._key_currencies_to_excel(excel_writer=writer)

        writer.save()

        print(f'File "{filename}" created.')

        if open_file:
            os.startfile(filename)

    def _scr_summary_to_excel(self, excel_writer):
        sheetname = 'SCR Summary'

        scr_summary_df = self.get_scr_summary_table()
        scr_summary_df.to_excel(excel_writer,
                                sheet_name=sheetname,
                                startcol=1,
                                startrow=1)

        # DF to Excel Table
        (max_row, max_col) = scr_summary_df.shape

        column_settings = [
            {'header': column} for column in scr_summary_df.columns]

        # We are using the index, so we have to insert the index name manually
        column_settings.insert(0, {'header': scr_summary_df.index.name})

        excel_writer.sheets[sheetname].add_table(
            1, 1, max_row + 1, max_col + 1, {'columns': column_settings}
        )

        # Get the xlsxwriter workbook and worksheet objects.
        wb = excel_writer.book
        num_format = wb.add_format({'num_format': '#,##0'})
        pct_format = wb.add_format({'num_format': '0.00%'})

        ws = excel_writer.sheets[sheetname]

        # Explanation as string, divided into an array so as to copy each line
        # in a new cell
        mov_explanation = self.get_movement_summary_text(
            to_html=False).split('\n')

        # Write explanation to cells below
        for line in range(len(mov_explanation)):
            # go over each index in the str split array
            ws.write(1 + line, 7, mov_explanation[line])

        # Formatting
        # SCR Summary Worksheet
        ws.set_column('B:B', 18)
        ws.set_column('C:D', 13, num_format)
        ws.set_column('E:E', 13, pct_format)
        ws.set_column('F:F', 23, num_format)
        ws.set_column('H:H', 78, num_format)

        return None

    def _quarter_over_quarter_movements_to_excel(self, excel_writer):
        sheetname = 'SCR QoQ Movements'

        df = self.calculate_quarter_movements()

        df.to_excel(excel_writer,
                    sheet_name=sheetname,
                    startcol=0,
                    startrow=0)

        # DF to Excel Table
        (max_row, max_col) = df.shape

        column_settings = [{'header': column} for column in df.columns]

        # We are using the index, so we have to insert the index name manually
        column_settings.insert(0, {'header': 'Currency'})

        excel_writer.sheets[sheetname].add_table(
            0, 0, max_row, max_col, {'columns': column_settings}
        )

        # Get the xlsxwriter workbook and worksheet objects.
        wb = excel_writer.book
        num_format = wb.add_format({'num_format': '#,##0'})
        pct_format = wb.add_format({'num_format': '0.00%'})

        ws = excel_writer.sheets[sheetname]

        # Formatting
        ws.set_column('A:A', 11)
        ws.set_column('B:E', 14, num_format)
        ws.set_column('F:F', 22)              # shock column
        ws.set_column('G:G', 23, num_format)  # abs cashflow column
        ws.set_column('H:L', 14, num_format)
        ws.set_column('M:M', 22)              # shock column
        ws.set_column('N:N', 23, num_format)  # abs cashflow column
        ws.set_column('O:P', 14, num_format)
        ws.set_column('Q:Q', 23, num_format)  # abs cashflow column

        return None
    
    def _key_currencies_to_excel(self, excel_writer):
        for curr in self._get_key_currencies().index:
            sheetname = curr
            # Array with main tables for each currency
            dfs = [self.get_natural_account_comparison(curr),
                   self.get_tp_comparison(curr),
                   self.get_currency_summary_table(curr)]

            # Row and Column Position in Excel
            df_positions = [(0, 0),
                            # Column position below is using Nat Acc col count
                            (0, dfs[0].shape[1] + 2),
                            # Row position below is using Nat Acc row count
                            (dfs[0].shape[0] + 3, 0)]
            print(df_positions)

            for i in range(len(dfs)):

                dfs[i].to_excel(excel_writer,
                                sheet_name=sheetname,
                                startrow=df_positions[i][0],
                                startcol=df_positions[i][1],
                                # i == 2 is the Currency Summary Table
                                # which uses the index to identify each row
                                index=True if i == 2 else False)
                
                (max_row, max_col) = dfs[i].shape

                column_settings = [
                    {'header': column} for column in dfs[i].columns]

                # We are using the index, so we have to insert the index name manually
                if i == 2:
                    column_settings.insert(0, {'header': 'Type'})
                else:
                    max_col -= 1

                excel_writer.sheets[sheetname].add_table(
                    df_positions[i][0],
                    df_positions[i][1],
                    max_row + df_positions[i][0],
                    max_col + df_positions[i][1],
                    {'columns': column_settings}
                )
                rgn = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I'}
                print(f"df {i}--RANGE: {rgn[df_positions[i][1]]}{df_positions[i][0]+1}:{rgn[max_col]}{max_row+1}")

                # Get the xlsxwriter workbook and worksheet objects.
                wb = excel_writer.book
                num_format = wb.add_format({'num_format': '#,##0'})

                ws = excel_writer.sheets[sheetname]

                # Formatting
                ws.set_column('A:A', 19)
                ws.set_column('B:C', 19, num_format)
                ws.set_column('D:E', 30, num_format)
                ws.set_column('F:F', 12, num_format)
                ws.set_column('I:I', 14)
                ws.set_column('J:K', 29, num_format)
                ws.set_column('L:L', 12, num_format)
