import webbrowser
from .market_risk_data import MarketRiskData
from .market_risk import MarketRisk
import pandas as pd
import numpy as np
import os
from ..utils import get_report_period, get_inwards_folder
from ..utils import CORPORATE_COLOR, PLOT_TITLE_FONT_FORMAT
import plotly.express as px
import plotly.graph_objects as go
from markdown2 import Markdown


class SpreadData(MarketRiskData):
    def __init__(self, folder_path: str, report_date: str) -> None:

        MarketRiskData.__init__(self,
                                folder_path=folder_path,
                                report_date=report_date)

        self._DATA_NAME = 'SPREAD'

        self.import_data()
        self.process_data()
    
    @staticmethod
    def _import_shock_parameters() -> dict:
        shock_parameters = {}
        filepath = os.path.join(get_inwards_folder(),
                                'modules',
                                'aux_files',
                                'spread_shock_parameters.xlsx')
        
        df = pd.read_excel(filepath, engine='openpyxl')

        shock_parameters['a'] = df.loc[df['fupItem'] == 'Absolute']
        shock_parameters['b'] = df.loc[df['fupItem'] == 'D_Relative']
        shock_parameters['subtract'] = df.loc[df['fupItem'] == 'D_Subtract']

        return shock_parameters
    
    def import_data(self):
        filepath = os.path.join(self.folder_path, 'BOND_DATA.xlsx')
        df = pd.read_excel(filepath, engine='openpyxl')

        # In case Excel file has extra lines, and they are imported with NaN
        df.dropna(axis='index', how='all', inplace=True)

        self.data = df

        return None
    
    def _insert_shock_parameters(self):
        # Modifying class directly
        df = self.data.copy()

        shock_parameters = self._import_shock_parameters()

        merge_on = ['DURATION_END_YEAR',
                    'ECAI',
                    'COUNTERPARTY_RISK_TYPE_CODE',
                    'TYPE_OF_BOND']

        shock_param_cols = ['DURATION_END_YEAR',
                            'ECAI',
                            'COUNTERPARTY_RISK_TYPE_CODE',
                            'TYPE_OF_BOND',
                            'value']

        for k, v in shock_parameters.items():
            df = df.merge(v[shock_param_cols], on=merge_on, how='left')
            df.rename(columns={'value': k}, inplace=True)

        CR1_filter = df['COUNTERPARTY_RISK_TYPE_CODE'] == 'CR1'
        # Need to set to 0, as to remove NaN values
        for col in ['a', 'b', 'subtract']:
            df.loc[CR1_filter, col] = 0
        
        self.data = df
    
    def _calculate_shock(self):
        df = self.data
        md_col = 'MODIFIED_DURATION'
        cond_list = [df[md_col] < 1,
                     df[md_col] <= 5,
                     df[md_col] <= 20,
                     df[md_col] > 20]

        choice_list = [
            df['b'],
            df['b'] * df[md_col],
            df['a'] + df['b'] * (df[md_col] - df['subtract']),
            np.minimum(df['a'] + df['b'] * (df[md_col] - df['subtract']), 1)
        ]

        df['SHOCK_FACTOR'] = np.select(cond_list, choice_list)

        return None
    
    def _calculate_scr(self):
        df = self.data
        df['SCR_ORIG_CURR'] = df['SHOCK_FACTOR'] * df['MARKET_VALUE']
        df['SCR'] = df['SHOCK_FACTOR'] * df['MARKET_VALUE_EUR']
        return None
    
    def _calulate_duration_to_maturity(self):
        df = self.data

        # Convert everything to pandas datetime
        df['MATURITY_DAT'] = pd.to_datetime(df['MATURITY_DAT'], dayfirst=True)

        if 'DURATION_TO_MATURITY' not in df.columns:
            df['DURATION_TO_MATURITY'] = (
                (df['MATURITY_DAT'] - df['REPORTING_DAT']) / np.timedelta64(1, 'Y')
            )
            df['DURATION_QTR'] = np.floor(df['DURATION_TO_MATURITY'] * 4).astype(np.int64)
            df['DURATION_YR'] = np.floor(df['DURATION_TO_MATURITY']).astype(np.int64)
        
        return None
    
    def _normalize_duration_qtr_yr(self):
        "Transform duration columns into sections for table and graphs."        
        df = self.data

        df['DURATION_QTR'] = df['DURATION_QTR'].apply(
            lambda x: f'{int(x):0>2}-{int((x + 1)):0>2} quarters')
        
        df['DURATION_YR'] = df['DURATION_YR'].apply(
            lambda x: f'{int(x)}-{int(x + 1)} years')
        
        return None
    
    def _normalize_modified_duration(self):
        df = self.data
        cond = [df['MODIFIED_DURATION'] < 0.4,
                df['MODIFIED_DURATION'] < 0.8,
                df['MODIFIED_DURATION'] < 1.2,
                df['MODIFIED_DURATION'] < 1.6,
                df['MODIFIED_DURATION'] < 2.0,
                df['MODIFIED_DURATION'] < 2.4,
                df['MODIFIED_DURATION'] < 2.8,
                df['MODIFIED_DURATION'] < 3.2,
                df['MODIFIED_DURATION'] < 3.6,
                df['MODIFIED_DURATION'] < 4.0,
                df['MODIFIED_DURATION'] < 4.4,
                df['MODIFIED_DURATION'] < 4.8,
                df['MODIFIED_DURATION'] < 5.2,
                df['MODIFIED_DURATION'] < 5.6,
                df['MODIFIED_DURATION'] < 6.0]

        choice = ["0.0 - 0.4",
                  "0.4 - 0.8",
                  "0.8 - 1.2",
                  "1.2 - 1.6",
                  "1.6 - 2.0",
                  "2.0 - 2.4",
                  "2.4 - 2.8",
                  "2.8 - 3.2",
                  "3.2 - 3.6",
                  "3.6 - 4.0",
                  "4.0 - 4.4",
                  "4.4 - 4.8",
                  "4.8 - 5.2",
                  "5.2 - 5.6",
                  "5.6 - 6.0", ]

        df['MODIFIED_DURATION_RANGE'] = np.select(cond, choice)

        return None

    def process_data(self):
        self._insert_shock_parameters()
        self._calculate_shock()
        self._calculate_scr()
        self._calulate_duration_to_maturity()
        self._normalize_duration_qtr_yr()
        self._normalize_modified_duration()

        print("{} data was added.".format(
            get_report_period(self.data['REPORTING_DAT'].unique()[0])
        ))

        return None
    
    def get_scr_sum(self):
        return self.data['SCR'].sum()
    
    def get_market_value_sum(self):
        return self.data['MARKET_VALUE_EUR'].sum()
    
    def match_fc_computed_amount(self):
        return self.FC_COMPUTED.round(1) == self.get_scr_sum().round(1)


class SpreadRisk(MarketRisk):
    QTR_COMPARISON_USE_COLS = ['BOND_IDENTIFIER',
                               #'ASSET_MANAGER',
                               'COUNTERPARTY_RISK_TYPE_CODE',
                               'BOND_CURRENCY_CODE',
                               'FX_RATE',
                               'MODIFIED_DURATION',
                               'MODIFIED_DURATION_RANGE',
                               'ECAI',
                               'MARKET_VALUE',
                               'MARKET_VALUE_EUR',
                               'DURATION_QTR',
                               'DURATION_YR',
                               'SHOCK_FACTOR',
                               'SCR_ORIG_CURR',
                               'SCR']

    QTR_COMPARISON_MERGE_ON_COLS = ['BOND_IDENTIFIER',
                                    #'ASSET_MANAGER',
                                    'COUNTERPARTY_RISK_TYPE_CODE']

    def __init__(self) -> None:
        
        MarketRisk.__init__(self)
        self.market_risk_type = 'spread'
        self.market_value_col = 'MARKET_VALUE'
        self.market_value_eur_col = 'MARKET_VALUE_EUR'

    def import_and_process_sourcedata_files(self,
                                            sourcedata_folder: str,
                                            report_date: str):

        # Add SpreadData to array
        df = SpreadData(folder_path=sourcedata_folder, report_date=report_date)
        self.market_risk_data.append(df)

        # To make sure last item in Array is latest quarter
        self.market_risk_data.sort()

        # # Update current report date and period with latest data
        # self._update_current_report_date_and_periods()

    def calculate_quarter_movements(self):
        # List with all data added to model
        dates_in_data = [i.report_date for i in self.market_risk_data]

        # Find latest 2 dates
        dates_for_comparison = pd.Series(dates_in_data).nlargest(2)

        # Set these dates to old and new, to compare Quarter over Quarter (QoQ)
        old_date = dates_for_comparison.min()
        new_date = dates_for_comparison.max()

        bond_data_old = None
        bond_data_new = None

        previous_qtr_index = -2
        current_qtr_index = -1


        for bond_data in self.market_risk_data:
            if bond_data.report_date == old_date:
                bond_data_old = bond_data
            elif bond_data.report_date == new_date:
                bond_data_new = bond_data

        # Merge data as to have everything side by side to see what
        # changed over the Quarter
        df = pd.merge(left=bond_data_old.data[self.QTR_COMPARISON_USE_COLS],
                      right=bond_data_new.data[self.QTR_COMPARISON_USE_COLS],
                      how='outer',
                      on=self.QTR_COMPARISON_MERGE_ON_COLS,
                      suffixes=('-'+bond_data_old.report_period,
                                '-'+bond_data_new.report_period))

        df_bond_status = self._get_bond_status(df)
        df_scr_movements = self._calculate_scr_movement(df_bond_status)
        df_scr_purch_sold = self._calculate_scr_purchased_sold_movements(
            df_scr_movements)
        df_scr_org_growth = self._calculate_scr_org_growth_movements(
            df_scr_purch_sold)
        df_scr_shock = self._calculate_scr_shock_movements(df_scr_org_growth)
        df_scr_fx = self._calculate_scr_fx_movements(df_scr_shock)

        return df_scr_fx

    def _get_bond_status(self, dataframe):
        df = dataframe

        filter_unchanged = ((df['SCR-'+self.periods['old']].notna())
                            & (df['SCR-'+self.periods['new']].notna()))

        filter_sold = ((df['SCR-'+self.periods['old']].notna())
                       & (df['SCR-'+self.periods['new']].isna()))

        filter_purchased = ((df['SCR-'+self.periods['old']].isna())
                            & (df['SCR-'+self.periods['new']].notna()))

        cond = [filter_unchanged,
                filter_sold,
                filter_purchased]

        choice = ['UNCHANGED',
                  'SOLD',
                  'PURCHASED']

        df['BOND_STATUS'] = np.select(cond, choice, '')

        return df

    def _calculate_scr_movement(self, dataframe):
        df = dataframe

        # fillna(0) to avoid errors where Bonds where sold or purchased
        df['Diff'] = (df['SCR-'+self.periods['new']].fillna(0)
                      - df['SCR-'+self.periods['old']].fillna(0))

        return df

    def _calculate_scr_purchased_sold_movements(self, dataframe):
        df = dataframe

        filter_sold = df['BOND_STATUS'] == 'SOLD'
        filter_purchased = df['BOND_STATUS'] == 'PURCHASED'

        df['Diff-Purchased/Sold'] = np.where(
            (filter_sold) | (filter_purchased),
            df['Diff'],
            0
        )

        return df

    def _calculate_scr_org_growth_movements(self, dataframe):
        df = dataframe

        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        df['Diff-Org Growth'] = np.where(
            filter_unchanged,
            # ( MV(new) - MV(old) ) * Shock(old) / FX_rate(old)
            ((df[f'{self.market_value_col}-'+self.periods['new']]
             - df[f'{self.market_value_col}-'+self.periods['old']])
             * df['SHOCK_FACTOR-'+self.periods['old']]
             / df['FX_RATE-'+self.periods['old']]),
            0
        )

        return df
    
    def get_quarter_movements_diff_sum(self):
        df = self.calculate_quarter_movements()
        return df[[i for i in df.columns if 'Diff' in i]].sum()

    def _calculate_scr_shock_movements(self, dataframe):
        df = dataframe

        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        df['Diff-Shock'] = np.where(
            filter_unchanged,
            # ( Shock(new) - Shock(old) ) * MV(new) / FX_rate(old)
            ((df['SHOCK_FACTOR-'+self.periods['new']]
             - df['SHOCK_FACTOR-'+self.periods['old']])
             * df[f'{self.market_value_col}-'+self.periods['new']]
             / df['FX_RATE-'+self.periods['old']]),
            0
        )

        return df

    def _calculate_scr_fx_movements(self, dataframe):
        df = dataframe

        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        df['Diff-FX'] = np.where(
            filter_unchanged,
            # ( Shock(new) - Shock(old) ) * MV(new) / FX_rate(old)
            (df['SHOCK_FACTOR-'+self.periods['new']]
             * df[f'{self.market_value_col}-'+self.periods['new']]
             * (1 / df['FX_RATE-'+self.periods['new']]
             - 1 / df['FX_RATE-'+self.periods['old']])),
            0
        )

        return df
    
    def get_quarter_movements_diff_sum(self):
        df = self.calculate_quarter_movements()
        return df[[i for i in df.columns if 'Diff' in i]].sum()
    
    def get_scr_sum(self, period: str) -> str:
        df = self.calculate_quarter_movements()
        return df['SCR-'+self.periods[period]].sum()
    
    def group_market_value_by(self,
                              by_col: str,
                              func: str,
                              remove_cr1: bool = True) -> pd.DataFrame:
        """Return DataFrame grouped by "by_col" variable.

        Parameters
        ----------
        by : str
            Column name to group by.
        func : str
            'sum' or 'count'
        remove_cr1 : bool, optional
            remove bonds where 
            `COUNTERPARTY_RISK_TYPE_CODE == CR1`, by default True

        Returns
        -------
        pd.DataFrame
            [description]
        """

        """[summary]

        Parameters
        ----------
        by : str
            Column name to group by.
        func : str
            'sum' or 'count'
        """
        df = self.data
        if remove_cr1:
            # CR1 Bonds have SCR 0, so there's no need to keep them.
            df = df.loc[df['COUNTERPARTY_RISK_TYPE_CODE'] != 'CR1']
        # Make copy of only what we need
        df = df[['PERIOD', f'{self.market_value_eur_col}', by_col]].copy()

        if 'MODIFIED_DURATION' in df.columns:
            df['MODIFIED_DURATION'] = df['MODIFIED_DURATION'].round(1)

        if func == 'sum':
            return df.groupby(by=['PERIOD', by_col]).sum().reset_index()
        elif func == 'count':
            return df.groupby(by=['PERIOD', by_col]).count().reset_index()
        else:
            raise ValueError(f"func must be 'sum' or 'count'. Got {func} ")
    
    def get_movement_summary_text(self, to_html: bool = False) -> str:

        text = self._get_overall_movement_text(to_html=to_html)

        text += self._get_purchased_sold_movement_text(to_html=to_html)

        text += self._get_organic_movement_text(to_html=to_html)

        text += self._get_spread_shock_movement_text(to_html=to_html)

        text += self._report_obs(to_html=to_html)

        # if to_html:
        #     text = text.replace('(I)', '<sup>(I)</sup>')
        #     text = text.replace('(II)', '<sup>(II)</sup>')
        #     text = text.replace('<sup>(II)</sup> Shock change',
        #                         '</p><p><sup>(II)</sup> Shock change')


        return text
    
    def _helper_weighted_average(self, by: str,  bond_status: str = None):
        """Return Weighted Average for specific column (by parameter).

        Parameters
        ----------
        by : str
            Column to calculate Weighted Average
        bond_status : str, optional
            Bond status for latest REPORTING_DAT, by default None

        Notes
        -----
        bond_status is used to filter the bond status in latest reporting
        period. Those bonds that match the criteria are then matched 
        in the previous quarter.

        Returns
        -------
        pandas.Series
        """        
        # Array with dates, ascending order
        df = self.calculate_quarter_movements()

        # # Rename ECAI rating of 9
        # df_raw.loc[df_raw['ECAI'] == 'Unrated', 'ECAI'] = 9
        
        if bond_status is not None:
            # If not None, then we get only the Bond IDs that were
            # SOLD, PURCHASED or UNCHANGED.
            df = df.loc[df['BOND_STATUS'] == bond_status].copy()

        if self.periods['old'] in by:
            mv_col = f'{self.market_value_eur_col}-'+self.periods['old']
        elif self.periods['new'] in by:
            mv_col = f'{self.market_value_eur_col}-'+self.periods['new']

        df['HELPER_WA'] = df[by] * df[mv_col]
        df['HELPER_WA'] = df['HELPER_WA'].fillna(0)
        s = df.sum()
        
        return s['HELPER_WA'] / s[mv_col]
    
    def _get_spread_shock_movement_text(self, to_html: bool) -> str:
        shock_wa_old = self._helper_weighted_average(
            by='SHOCK_FACTOR-'+self.periods['old'],
            bond_status='UNCHANGED'
        )
        
        shock_wa_new = self._helper_weighted_average(
            by='SHOCK_FACTOR-'+self.periods['new'],
            bond_status='UNCHANGED'
        )

        text = '#### Shock(I) (II):\n'
        text += f'- Shock from {shock_wa_old:.1%} to {shock_wa_new:.1%}'

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text

    @staticmethod
    def _report_obs(to_html: bool) -> str:

        obs1 = '\n(I) Only Bonds that were not purchased or sold.\n'
        obs2 = '(II) Shock change driven by change to modified duration or rating.'

        text = obs1 + obs2

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def get_quarter_movements_by(self,
                                 by: str,
                                 period: str,
                                 exclude_cr1: bool) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        by : str
            by : str
            'yr', 'qtr', 'md' and 'ecai'
            - yr: year
            - qtr: quarter
            - md: modified duration
            - ecai: ecai
        period : str
            'old' or 'new', refering to previous or current quarter

        Returns
        -------
        pd.DataFrame
            [description]
        """
        if by =='yr':
            by_col = "DURATION_YR"
        elif by == 'qtr':
            by_col = "DURATION_QTR"
        elif by == 'md':
            by_col = "MODIFIED_DURATION_RANGE"
        elif by == 'ecai':
            by_col = 'ECAI'
        else:
            raise ValueError(f"Parameter 'by' can be 'yr', 'qtr', 'md', 'ecai'. {by} was passed")

        # Get Quarter Movements DF
        df_raw = self.calculate_quarter_movements()

        if exclude_cr1:
            df_raw = df_raw.loc[
                df_raw['COUNTERPARTY_RISK_TYPE_CODE'] != 'CR1'
            ].copy()
        
        # Select columns only for period in question
        cols = [col for col in df_raw.columns if self.periods[period] in col]
        
        # Drop NA to count the correct number of Bonds
        df = df_raw[cols].dropna()
        
        # Count column to count number of bonds
        df['Count-'+self.periods[period]] = 1
        
        # Group by column in "by".
        df_grp = df.groupby(
            f'{by_col}-'+self.periods[period]
        ).sum().rename_axis(None)
        
        return df_grp
    
    def get_mv_and_count_by(self, by: str) -> pd.DataFrame:
        """[summary]

        Parameters
        ----------
        by : str
            'yr', 'qtr', 'md' and 'ecai'
            - yr: year
            - qtr: quarter
            - md: modified duration
            - ecai: ecai

        Returns
        -------
        pd.DataFrame
            [description]
        """        
        df_old = self.get_quarter_movements_by(by=by,
                                               period='old',
                                               exclude_cr1=True)
        df_new = self.get_quarter_movements_by(by=by,
                                               period='new',
                                               exclude_cr1=True)

        # Merge everything together so we can have an Excel graph
        df = df_old.merge(df_new,
                          left_index=True,
                          right_index=True,
                          how='outer')

        # Keeping only MV and Count
        cols_to_keep = [
            f'{self.market_value_eur_col}-'+self.periods['old'],
            f'{self.market_value_eur_col}-'+self.periods['new'],
            'Count-'+self.periods['old'],
            'Count-'+self.periods['new'],
            'SCR-'+self.periods['old'],
            'SCR-'+self.periods['new']
        ]

        return df[cols_to_keep].fillna(0)
    
    def plot_maturity(self, by: str, value: str) -> px.line:
        """Return plot of Market Value, according to parameters.

        Notes
        -----
        This function DOES NOT plot bonds where
        `COUNTERPARTY_RISK_TYPE_CODE == 'CR1'`.

        Parameters
        ----------
        by : str
            'yr', 'qtr', 'md' or 'ecai'.
        value : str
            'mv' or 'count'

        Returns
        -------
        px.line
            return line plot.
        """

        if value == 'mv':
            col = f'{self.market_value_eur_col}-'
            y_axis_title = 'Market Value (EUR)'
        elif value == 'count':
            col = 'Count-'
            y_axis_title = 'Market Value (Count)'
        else:
            raise ValueError(f"Value can be 'mv' or 'count'. {value} was passed.")
        
        if by == 'yr':
            x_axis_title = "Duration (Years)"
        elif by == 'qtr':
            x_axis_title = "Duration (Quarters)"
        elif by == 'md':
            x_axis_title = "Modified Duration Brackets"
        elif by == 'ecai':
            x_axis_title = "ECAI"
        else:
            raise ValueError(f"'by' can be 'yr', 'qtr', 'md' or 'ecai'. {by} was passed.")

        plot_title = f"{y_axis_title} by {x_axis_title}"

        df = self.get_mv_and_count_by(by=by)

        df['Diff'] = df[col+self.periods['new']] - df[col+self.periods['old']]

        # Create Figure object
        fig = go.Figure()

        # Add Previous Quarter Data
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df[col+self.periods['old']],
                                 mode='lines+markers',
                                 name=col+self.periods['old'],
                                 line={'color': CORPORATE_COLOR[0]}))
        
        # Add Current Quarter Data
        fig.add_trace(go.Scatter(x=df.index,
                                 y=df[col+self.periods['new']],
                                 mode='lines+markers',
                                 name=col+self.periods['new'],
                                 line={'color': CORPORATE_COLOR[1]}))
        
        # Add Difference QoQ
        fig.add_trace(go.Bar(x=df.index,
                             y=df['Diff'],
                             name='Difference QoQ',
                             marker_color=CORPORATE_COLOR[-1]))

        fig.update_layout(title=plot_title,
                          title_font=PLOT_TITLE_FONT_FORMAT,
                          hovermode='x',
                          template='plotly_white')
        if value == 'mv':
            fig.update_layout(yaxis_tickformat='.2s')
        
        if by == 'ECAI':
            fig.update_xaxes(type='category')

        return fig
    
    

    def generate_excel_report(self):
        filename = self._get_report_name('xlsx')
        writer = pd.ExcelWriter(filename,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')

        # Create Sheet with SCR Summary
        self._scr_summary_to_excel(excel_writer=writer)

        # Create Sheet with Quarter Movements
        self._quarter_movements_to_excel(excel_writer=writer)

        # Create Sheets with Raw Bond Data
        self._raw_bond_data_to_excel(excel_writer=writer)

        # Create Sheet with Duration to Maturity analysis
        self._duration_to_maturity_to_excel(excel_writer=writer)

        writer.save()

        print(f'File "{filename}" created.')

    def _quarter_movements_to_excel(self, excel_writer):
        # Export Quarters comparison
        df = self.calculate_quarter_movements()
        sheetname = 'QoQ Movements'
        df.to_excel(excel_writer=excel_writer,
                    sheet_name=sheetname,
                    index=False)

        # DF to Excel Table
        (max_row, max_col) = df.shape

        column_settings = [{'header': column} for column in df.columns]

        excel_writer.sheets[sheetname].add_table(
            0, 0, max_row, max_col - 1, {'columns': column_settings}
        )

        wb = excel_writer.book
        num_format = wb.add_format({'num_format': '#,##0'})
        curr_format = wb.add_format({'num_format': '#,##0.0000'})

        # Formatting
        ws = excel_writer.sheets[sheetname]
        ws.set_column('A:B', 19)

        # Old Quarter
        ws.set_column('C:C', 33, num_format)
        ws.set_column('D:D', 18, curr_format)
        ws.set_column('E:E', 30)
        ws.set_column('F:F', 12)
        ws.set_column('G:H', 23, num_format)
        ws.set_column('I:J', 21)
        ws.set_column('K:K', 13)
        ws.set_column('L:L', 23, num_format)
        ws.set_column('M:M', 12, num_format)

        # New Quarter
        ws.set_column('N:N', 33, num_format)
        ws.set_column('O:O', 18, curr_format)
        ws.set_column('P:P', 30)
        ws.set_column('Q:Q', 12)
        ws.set_column('R:S', 23, num_format)
        ws.set_column('T:U', 21)
        ws.set_column('V:V', 13)
        ws.set_column('W:W', 23, num_format)
        ws.set_column('X:X', 12, num_format)
        ws.set_column('Y:Y', 21)

        # Deltas
        ws.set_column('Z:AD', 21, num_format)

    def _raw_bond_data_to_excel(self, excel_writer):
        # Export raw BOND data
        for i in self.market_risk_data:
            sheetname = f'BOND_DATA {i.report_period}'
            df = i.data
            df.to_excel(excel_writer,
                        sheet_name=sheetname,
                        index=False)

            (max_row, max_col) = df.shape
            column_settings = [{'header': column} for column in df.columns]
            excel_writer.sheets[sheetname].add_table(
                0, 0, max_row, max_col - 1, {'columns': column_settings}
            )

            wb = excel_writer.book
            num_format = wb.add_format({'num_format': '#,##0'})
            curr_format = wb.add_format({'num_format': '#,##0.0000'})

            # Formatting
            ws = excel_writer.sheets[sheetname]
            ws.set_column('A:D', 18)
            ws.set_column('E:E', 11)
            ws.set_column('F:H', 33)
            ws.set_column('I:I', 23)
            ws.set_column('J:J', 11, curr_format)
            ws.set_column('K:M', 17)
            ws.set_column('N:N', 23)
            ws.set_column('O:P', 9)
            ws.set_column('Q:R', 21, num_format)
            ws.set_column('S:V', 16)
            ws.set_column('W:Z', 10)
            ws.set_column('AA:AB', 18, num_format)

    def _scr_summary_to_excel(self, excel_writer):

        sheetname = 'SCR Summary'
        self.get_scr_summary_table().to_excel(excel_writer,
                                              sheet_name=sheetname,
                                              startcol=1,
                                              startrow=1)

        # Get the xlsxwriter workbook and worksheet objects.
        wb = excel_writer.book
        num_format = wb.add_format({'num_format': '#,##0'})
        pct_format = wb.add_format({'num_format': '0.00%'})
        merge_format = wb.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'vcenter'
        })
        
        ws = excel_writer.sheets[sheetname]

        # Write explanation to cell below
        ws.merge_range('B7:D28', self.get_movement_summary_text(), merge_format)

        # Formatting
        # SCR Summary Worksheet
        ws.set_column('B:B', 15)
        ws.set_column('C:D', 13, num_format)
        ws.set_column('E:E', 13, pct_format)
        ws.set_column('F:G', 21, num_format)
    
    def _duration_to_maturity_to_excel(self, excel_writer):

        wb = excel_writer.book

        row = 0
        excel_row_beg = 2
        for i in ['qtr', 'yr']:

            sheetname = 'Duration to Maturity analysis'

            df = self.get_mv_and_count_by(by=i)
            df.to_excel(excel_writer,
                        sheet_name=sheetname,
                        startcol=0,
                        startrow=row)
            
            ws = excel_writer.sheets[sheetname]

            # Inserting graphs
            # ----------------
            
            excel_row_end = df.shape[0] + 1

            # Market Value Chart
            mv_chart = wb.add_chart({'type': 'line'})

            # Add old series to the chart.
            mv_chart.add_series({
                'categories': f"='{sheetname}'!$A${excel_row_beg}:$A${excel_row_end}",
                'values':     f"='{sheetname}'!$B${excel_row_beg}:$B${excel_row_end}",
                'marker': {'type': 'diamond'}
            })
            
            # Add new series to the chart.
            mv_chart.add_series({
                'categories': f"='{sheetname}'!$A${excel_row_beg}:$A${excel_row_end}",
                'values':     f"='{sheetname}'!$C${excel_row_beg}:$C${excel_row_end}",
                'marker': {'type': 'diamond'}
            })

            ws.insert_chart('H2', mv_chart)

            # Count Chart
            count_chart = wb.add_chart({'type': 'line'})

            # Add old series to the chart.
            count_chart.add_series({
                'categories': f"='{sheetname}'!$A${excel_row_beg}:$A${excel_row_end}",
                'values':     f"='{sheetname}'!$D${excel_row_beg}:$D${excel_row_end}",
                'marker': {'type': 'diamond'}
            })

            # Add new series to the chart.
            count_chart.add_series({
                'categories': f"='{sheetname}'!$A${excel_row_beg}:$A${excel_row_end}",
                'values':     f"='{sheetname}'!$E${excel_row_beg}:$E${excel_row_end}",
                'marker': {'type': 'diamond'}
            })

            ws.insert_chart('U2', count_chart)

            # 3 lines between one table and the other
            row += df.shape[0] + 3

            # The plus one is to normalize for Excel, since here
            # The row number starts at 0 (zero)
            excel_row_beg += row + 1

        # Formatting
        num_format = wb.add_format({'num_format': '#,##0'})

        ws.set_column('A:A', 26)
        ws.set_column('B:C', 26, num_format)
        ws.set_column('D:E', 13, num_format)

    def generate_html_report(self):
        from jinja2 import Environment, FileSystemLoader

        templates_dir = os.path.join(get_inwards_folder(),
                                     'modules',
                                     'market_risk',
                                     'templates')
                                     
        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('spread_risk.html')

        filename = os.path.join(os.getcwd(),
                                self._get_report_name(file_extension='html'))
        
        # Dict with all elements to complete HTML report
        elem = self._get_html_elements()
        
        with open(filename, 'w') as fh:
            fh.write(template.render(
                current_period=self.periods['new'],
                previous_period=self.periods['old'],
                current_scr=elem['current_scr'],
                previous_scr_in_m=elem['previous_scr_in_m'],
                total_diff_qoq=elem['total_diff_qoq'],
                scr_increase=elem['scr_increase'],
                scr_diff_pct_formatted=elem['scr_diff_pct_formatted'],
                scr_movement_summary_text=elem['scr_movement_summary_text'],
                waterfall_plot=elem['waterfall_movement_summary'],
                mv_scr_plot=elem['mv_scr_plot'],
                maturity_analysis_plots=elem['maturity_analysis_plots']
            ))
        
        print(f"HTML report created at {filename}.")
        webbrowser.open(filename, new=2)
    
    def _get_html_elements(self) -> dict:
        elem = {}
        # First Box
        elem['current_scr'] = "EUR {:,.0f}m".format(
            self.get_scr_sum(period='new'))
        
        # Second Box
        elem['previous_scr'] = "EUR {:,.0f}m".format(
            self.get_scr_sum(period='old'))
        elem['previous_scr_in_m'] = "EUR {:,.0f}m".format(
            self.get_scr_sum(period='old') / 1e6)
        elem['total_diff_qoq'] = "EUR {:+,.2f}m".format(
            self.get_quarter_movements_diff_sum()['Diff'] / 1e6
        )

        elem['scr_diff_pct'] = self.get_scr_summary_table().loc[
            self.current_report_date, 'Diff %']
        elem['scr_diff_pct_formatted'] = "{:+,.1%}".format(elem['scr_diff_pct'])
        elem['scr_increase'] = elem['scr_diff_pct'] > 0

        elem['scr_movement_summary_text'] = self.get_movement_summary_text(
            to_html=True)
        elem['waterfall_movement_summary'] = self.plot_waterfall_movement_summary(
            for_html_report=True)
        
        elem['mv_scr_plot'] = self.plot_market_value_and_scr(
            add_mv_scr_ratio=True,
            for_html_report=True)
        
        elem['maturity_analysis_plots'] = self._maturity_analysis_plots_for_html()

        return elem
    
    def _maturity_analysis_plots_for_html(self) -> list:
        plots = []

        tab_name = {'qtr': 'Duration (Quarters)',
                    'yr': 'Duration (Years)',
                    'md': 'Modified Duration',
                    'ecai': 'ECAI'}

        for i in ['qtr', 'yr', 'md', 'ecai']:
            fig_mv = self.plot_maturity(by=i, value='mv')
            fig_count = self.plot_maturity(by=i, value='count')
            plots.append((
                fig_mv.to_html(include_plotlyjs=False,
                               full_html=False,
                               default_width=900),
                fig_count.to_html(include_plotlyjs=False,
                                  full_html=False,
                                  default_width=900),
                i,
                tab_name[i]
            ))

        return plots

    def generate_reports(self, excel: bool = False, html: bool = False):
        if excel:
            self.generate_excel_report()
        
        if html:
            self.generate_html_report()
