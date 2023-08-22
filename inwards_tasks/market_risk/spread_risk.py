from typing import List, Union
import webbrowser
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline
from markdown2 import Markdown

try:
    from .utils import CORPORATE_VALUE_COLOR
    from .utils import PLOT_TITLE_FONT_FORMAT
    from .utils import CORPORATE_COLOR
    from .utils import produce_unique_rows
    from .utils import get_report_period

    from .market_risk import MarketRisk
    from .risk_data import RiskData
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from utils import CORPORATE_VALUE_COLOR
    from utils import PLOT_TITLE_FONT_FORMAT
    from utils import CORPORATE_COLOR
    from utils import produce_unique_rows
    from utils import get_report_period

    from fc_computed_amounts import import_fc_computed_amounts
    from market_risk import MarketRisk
    from risk_data import RiskData


def import_shock_parameters() -> dict:
    shock_parameters = {}
    filepath = os.path.join(os.path.dirname(__file__),
                            'aux_files',
                            'spread_shock_parameters.txt')

    df = pd.read_csv(filepath, sep='\t')

    shock_parameters['a'] = df.loc[df['fupItem'] == 'Absolute']
    shock_parameters['b'] = df.loc[df['fupItem'] == 'D_Relative']
    shock_parameters['subtract'] = df.loc[df['fupItem'] == 'D_Subtract']

    return shock_parameters


class SpreadData(RiskData):
    def __init__(self, folder_path: str) -> None:

        super().__init__(folder_path=folder_path)

        self._DATA_NAME = 'SPREAD'
        self._data = None

        self.import_data()
        
    
    def __repr__(self) -> str:
        return super().__repr__()
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()
    
    @data.setter
    def data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # First, we process the data
        df = self.process_data(df)

        # Then assing processed DF to model
        self._data = df
    
    def import_data(self) -> None:
        dtypes = {'MARKET_VALUE_EUR': np.float64,
                  'MARKET_VALUE': np.float64}
        try:
            filepath = os.path.join(self.folder_path, 'BOND_DATA.xlsx')
            df = pd.read_excel(filepath, engine='openpyxl', dtype=dtypes)
        except FileNotFoundError:
            filepath = os.path.join(self.folder_path, 'BOND_DATA.txt')
            df = pd.read_csv(filepath, sep='\t', dtype=dtypes)

        # In case file has extra lines, and they are imported with NaN
        df.dropna(axis='index', how='all', inplace=True)

        self.data = df
        self.fc_computed = self.import_fc_computed()

        return None
    
    def insert_shock_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        # Modifying class directly
        shock_parameters = import_shock_parameters()

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
        
        return df
    
    def _calculate_shock(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
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

        return np.select(cond_list, choice_list)
    
    def _calculate_scr(self, df: pd.DataFrame) -> pd.Series:
        return df['SHOCK_FACTOR'] * df['MARKET_VALUE_EUR']
    
    def _calulate_duration_to_maturity(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Convert everything to pandas datetime
        df['MATURITY_DAT'] = pd.to_datetime(df['MATURITY_DAT'], dayfirst=True)

        if 'DURATION_TO_MATURITY' not in df.columns:
            df['DURATION_TO_MATURITY'] = (
                (df['MATURITY_DAT'] - df['REPORTING_DAT']) / np.timedelta64(1, 'Y')
            )
            df['DURATION_QTR'] = np.floor(df['DURATION_TO_MATURITY'] * 4).astype(np.int64)
            df['DURATION_YR'] = np.floor(df['DURATION_TO_MATURITY']).astype(np.int64)
        
        return df
    
    def _normalize_duration_qtr_yr(self, df: pd.DataFrame) -> pd.DataFrame:
        "Transform duration columns into sections for table and graphs."        
        df = df.copy()

        df['DURATION_QTR'] = df['DURATION_QTR'].apply(
            lambda x: f'{int(x):0>2}-{int((x + 1)):0>2} quarters')
        
        df['DURATION_YR'] = df['DURATION_YR'].apply(
            lambda x: f'{int(x)}-{int(x + 1)} years')
        
        return df
    
    def _normalize_modified_duration(self, df: pd.DataFrame) -> pd.Series:
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

        return np.select(cond, choice)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:

        # Convert column to Timestamp
        df['REPORTING_DAT'] = pd.to_datetime(df['REPORTING_DAT'], dayfirst=True)

        df_shocks = self.insert_shock_parameters(df)
        df_shocks['SHOCK_FACTOR'] = self._calculate_shock(df_shocks)
        df_shocks['SCR'] = self._calculate_scr(df_shocks)
        df_maturity = self._calulate_duration_to_maturity(df_shocks)
        df_normalized = self._normalize_duration_qtr_yr(df_maturity)
        df_normalized['MODIFIED_DURATION_RANGE'] = self._normalize_modified_duration(df_normalized)

        return df_normalized
    
    def _filter_data(
        self,
        filter_issuer_ids: Union[None, List[str]]
    ) -> pd.DataFrame:

        df = self.data
        if filter_issuer_ids is not None:
            # If there's a list of ISSUER_IDS, then filter data
            df = df.loc[df['ISSUER_ID'].isin(filter_issuer_ids)]

        return df
    
    def get_weighted_average(self,
                             value_col: str,
                             weight_col: str,
                             filter_issuer_ids: List[str] = None) -> float:
        
        df = self._filter_data(filter_issuer_ids=filter_issuer_ids)

        if value_col == 'ECAI':
            # ECAI is loaded as str, so we convert to int so we can calculate
            df[value_col] = df[value_col].astype(int)

        df['WA_HELPER'] = df[value_col] * df[weight_col]

        return df['WA_HELPER'].sum() / df[weight_col].sum()
    
    def get_scr_sum(self):
        return self.data['SCR'].sum()
    
    # def get_market_value_sum(self):
    #     return self.data['MARKET_VALUE_EUR'].sum()
    


class SpreadRisk(MarketRisk):
    QTR_COMPARISON_USE_COLS = ['BOND_IDENTIFIER',
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

    ID_COLUMNS = ['BOND_IDENTIFIER']
    
    ISSUER_DATA_COLS = ['REPORTING_DAT',
                        'BOND_IDENTIFIER',
                        'ISSUER_ID',
                        'ISSUER_DES',
                        'ISSUER_GROUP_DES']

    def __init__(self) -> None:
        
        super().__init__()

        self._issuer_data = None

        self.market_risk_type = 'spread'
        self.market_value_col = 'MARKET_VALUE'
        self.market_value_eur_col = 'MARKET_VALUE_EUR'
    
    @property
    def data(self) -> Union[None, pd.DataFrame]:
         # Make copy to not alter original
        df = self._data.copy()

        # Drop all ISSUER_DATA columns except issuer ID and reporting date
        df.drop(self.ISSUER_DATA_COLS[2:], axis=1, inplace=True)

        # Merge in ALIAS_NAME back to self._data (main DataFrame), removing
        # reporting date from issuer data information
        return df.merge(self.issuer_data,
                        how='left',
                        on=self.ID_COLUMNS)

    @data.setter
    def data(self, risk_data: SpreadData) -> None:
        # Store individual data in model
        self.datasets = risk_data

        # Concatenate with data already in Model, as to have single DataFrame
        # with all values
        df = pd.concat([self._data, risk_data.data])

        self._issuer_data = df

        # Drop any duplicates, in case we add twice same data
        df.drop_duplicates(inplace=True)

        self._data = produce_unique_rows(data=df, keys=self.ID_COLUMNS)

        print(f"Data added for {risk_data.report_period}.")
    
    def import_data(self, folder_path: str) -> None:
        self.data = SpreadData(folder_path=folder_path)

    def calculate_movements(
        self,
        only_latest_n: Union[None, int] = None,
        mv_diff: bool = False
    ) -> pd.DataFrame:

        df = self.data

        if isinstance(only_latest_n, int):
            # Filter latest n periods. For example. If user wants only 
            # latest 2 periods, then we slice dates_added with [-2:]
            # This gives us the latest 2 dates added
            df = df.loc[
                df['REPORTING_DAT'].isin(self.dates_added[-only_latest_n:])
            ].copy()

        # To help with calculations, fillna(0)
        df['SCR'].fillna(0, inplace=True)
        df['MARKET_VALUE_EUR'].fillna(0, inplace=True)

        if mv_diff is True:
            df['Diff-MV'] = self.market_value_diff(df)
        
        df['BOND_STATUS'] = self.add_bond_status(df)
        df['Diff'] = self.calculate_scr_movement(df)
        df['Diff-Purchased/Sold'] = self.calculate_scr_purch_sold_movements(df)
        df['Diff-Org Growth'] = self.calculate_scr_organic_movements(df)
        df['Diff-Shock'] = self.calculate_scr_shock_movements(df)
        df['Diff-FX'] = self.calculate_scr_fx_movements(df)

        return df

    
    @staticmethod
    def add_bond_status(df: pd.DataFrame) -> pd.Series:
        prev_row_is_same_eq_id = df['BOND_IDENTIFIER'].eq(df['BOND_IDENTIFIER'].shift(1))
        #prev_row_is_same_purch_dat = df['PURCHASE_DAT'].eq(df['PURCHASE_DAT'].shift(1))
        #prev_row_is_same_issuer_id = df['ISSUER_ID'].eq(df['ISSUER_ID'].shift(1))
        # prev_row_is_same = ((prev_row_is_same_eq_id)
        #                     & (prev_row_is_same_purch_dat)
        #                     & (prev_row_is_same_issuer_id))
        prev_row_is_same = (prev_row_is_same_eq_id)

        market_val_col = 'MARKET_VALUE'
        
        is_nothing = (
            (df[market_val_col].shift(1).isna()) 
            & (df[market_val_col].isna())
        )

        bond_sold = ((df[market_val_col].shift(1).notna()) 
                    & df[market_val_col].isna())
        
        bond_purchased = ((df[market_val_col].shift(1).isna()) 
                          & df[market_val_col].notna())
        
        bond_unchanged = ((df[market_val_col].shift(1).notna()) 
                          & df[market_val_col].notna())
        


        condlist = [~prev_row_is_same,
                    is_nothing,
                    bond_sold,
                    bond_purchased,
                    bond_unchanged]

        choicelist = ['',
                      '',
                      'SOLD',
                      'PURCHASED',
                      'UNCHANGED']
        
        return np.select(condlist, choicelist, '')
    
    def calculate_scr_organic_movements(self,
                                        df: pd.DataFrame) -> pd.Series:
        """Return SCR movement due to Organic movements only.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing entire data, with one unique entry for each
            reporting period.

        Returns
        -------
        pd.Series
            Return SCR Movement due to Organic as Series, to be inserted
            directly to original DataFrame.
        """        

        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        return np.where(filter_unchanged,
                        # ( MV(new) - MV(old) ) * Shock(old)
                        # * Diversifcation_Benefit(old)
                        # ------------------------------------------
                        # FX_rate(old)
                        (
                            (df['MARKET_VALUE'] - df['MARKET_VALUE'].shift(1))
                            * df['SHOCK_FACTOR'].shift(1)
                            / df['FX_RATE'].shift(1)
                        ),
                        0)
    
    def calculate_scr_shock_movements(self, df: pd.DataFrame) -> pd.Series:
        """Return SCR movement due to Shocks only.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing entire data, with one unique entry for each
            reporting period.

        Returns
        -------
        pd.Series
            Return SCR Movement due to Shocks as Series, to be inserted
            directly to original DataFrame.
        """        
        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        return np.where(filter_unchanged,
                        # ( Shock(new) - Shock(old) ) * MV(new)
                        # * Diversifcation_Benefit(old)
                        # -------------------------------------
                        # FX_rate(old)
                        (
                            (df['SHOCK_FACTOR'] - df['SHOCK_FACTOR'].shift(1))
                            * df['MARKET_VALUE']
                            / df['FX_RATE'].shift(1)
                        ),
                        0)

    def calculate_scr_fx_movements(self, df: pd.DataFrame) -> pd.Series:
        """Return SCR movement due to FX Rates only.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing entire data, with one unique entry for each
            reporting period.

        Returns
        -------
        pd.Series
            Return SCR Movement due to FX Rates as Series, to be inserted
            directly to original DataFrame.
        """
        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        return np.where(filter_unchanged,
                        # ( Shock(new) - Shock(old) ) * MV(new)
                        # * Diversifcation_Benefit(old)
                        # -------------------------------------
                        # 1 / FX_rate(new) - 1 / FX_rate(old)
                        (
                            df['SHOCK_FACTOR']
                            * df['MARKET_VALUE']
                            * (1 / df['FX_RATE'] - 1 / df['FX_RATE'].shift(1))
                        ),
                        0)
    
    # def get_scr_sum(self, period: str) -> str:
    #     df = self.calculate_quarter_movements()
    #     return df['SCR-'+self.periods[period]].sum()

    def market_value_diff(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(
            by=self.ID_COLUMNS)[self.market_value_eur_col].diff().fillna(0)

    def filter_calculated_movements(
        self,
        dates: Union[None, List[pd.Timestamp]] = None,
        bond_status: Union[None, List[str]] = None,
        issuer_ids: Union[None, List[str]] = None) -> pd.DataFrame:
        """Return filtered DataFrame from self.data.

        Parameters
        ----------
        bond_status : Union[None, List[str]], optional
            if not None, then will return a DataFrame containing only
            the bond status passed, by default None
        issuer_ids : Union[None, List[str]], optional
            if not None, then will return a DataFrame containing only
            the issuer ids passed, by default None

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        df = self.calculate_movements(mv_diff=True)

        if isinstance(dates, list):
            # Filter for specific dates
            df = df.loc[df['REPORTING_DAT'].isin(dates)]

        if isinstance(bond_status, list):
            # Filter for specific bond status
            df = df.loc[df['BOND_STATUS'].isin(bond_status)]
        
        if isinstance(issuer_ids, list):
            # Filter for specific bond status
            df = df.loc[df['ISSUER_ID'].isin(issuer_ids)]
        
        return df

    def get_scr_summary(self, reset_index: bool = False) -> pd.DataFrame:
        df = self.calculate_movements()
        df_grp = df.groupby('REPORTING_DAT').sum()
        df_grp['Diff %'] = df_grp['SCR'].pct_change()
        df_grp['SCR : MV'] = df_grp['SCR'] /  df_grp[self.market_value_eur_col]

        # Dropping cols because we are summing all rows, so the individual 
        # values lose any meaning
        cols_to_keep = [
            col for col in df_grp.columns if 'Diff' in col or 'SCR' in col or 'MARKET_VALUE_EUR' in col
        ]

        if reset_index:
            return df_grp[cols_to_keep].reset_index()

        return df_grp[cols_to_keep]
    
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

        text = self.get_overall_movement_text(to_html=to_html)

        text += self.get_purchased_sold_movement_text(to_html=to_html)

        text += self._get_organic_movement_text(to_html=to_html)

        text += self._get_shock_movement_text(to_html=to_html)

        text += self._report_obs(to_html=to_html)

        if to_html:
            text = text.replace('(I)', '<sup>(I)</sup>')
            text = text.replace('(II)', '<sup>(II)</sup>')
            text = text.replace('<sup>(II)</sup> Shock change',
                                '</p><p><sup>(II)</sup> Shock change')
        return text
    
    def get_overall_movement_text(self, to_html: bool = False) -> str:
        prev_data_idx = -2
        curr_data_idx = -1
        
        # Overall Calculations
        prev_scr = self.datasets[prev_data_idx].scr
        curr_scr = self.datasets[curr_data_idx].scr
        scr_diff = curr_scr - prev_scr
        scr_pct_ch = (scr_diff) / prev_scr

        text = '#### Total Movement:\n'
        text += '- {:+.1%} (or €{:+,.1f}m) from last quarter\n'.format(
            scr_pct_ch,
            scr_diff / 1e6
        )
        text += '- From €{:,.1f}m to €{:,.1f}m\n\n'.format(prev_scr / 1e6,
                                                           curr_scr / 1e6)

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def get_purchased_sold_movement_text(self, to_html: bool = False) -> str:
        # Using Diff column because if Purchased this quarter, then previous
        # quarter would have MV of Zero. The different will be the total 
        # Market Value. Same logic applies to sold.
        market_val_diff_col = 'Diff-MV'
        
        mv_purch_val = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],
            bond_status=['PURCHASED']
        )[market_val_diff_col].sum()

        # Absolute value because the if sold, the different will be negative
        mv_sold_val = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],
            bond_status=['SOLD']
        )[market_val_diff_col].abs().sum()

        
        text = '#### Purchased / Sold:\n'

        text += '- Total Market Value Purchased: €{:,.2f}m\n'.format(
            mv_purch_val / 1e6)
        
        text += '- Total Market Value Sold: €{:,.2f}m\n'.format(
            mv_sold_val / 1e6)
        
        text += '- Net Purchased / Sold in Quarter: €{:+,.1f}m\n'.format(
            (mv_purch_val - mv_sold_val) / 1e6)
        
        text += self._get_weighted_average_text()
        
        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def _get_organic_movement_text(self, to_html: bool = False) -> str:

        # Only bonds that were not sold or purchased
        mv_eur_current = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],
            bond_status=['UNCHANGED']
        )[self.market_value_eur_col].sum()

        mv_eur_previous = self.filter_calculated_movements(
            dates=[self.dates_added[-2]],
            bond_status=['UNCHANGED']
        )[self.market_value_eur_col].sum()

        text = '#### Organic Growth<sup>(I)</sup>:\n'
        text += '- Market Value from €{:,.1f}m to €{:,.1f}m\n'.format(
            mv_eur_previous / 1e6,
            mv_eur_current / 1e6,
        )

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def _get_weighted_average_text(self) -> str:
        text = ''

        df = self.calculate_movements()

        df_only_purch = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],  # most recent quarter
            bond_status=['PURCHASED'],     # only unchanged bond status
        )

        # Filtering latest quarter with only bonds that were sold.
        # Then we'll use these bonds to calculate the weighted average 
        # Shock and ECAI last quarter
        df_only_sold = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],  # most recent quarter
            bond_status=['SOLD'],     # only unchanged bond status
        )
        
        for val in ['ECAI', 'SHOCK_FACTOR']:
            wa_purch = self.datasets[-1].get_weighted_average(
                value_col=val,
                weight_col=self.market_value_eur_col,
                filter_issuer_ids=df_only_purch['ISSUER_ID']
            )

            wa_sold = self.datasets[-2].get_weighted_average(
                value_col=val,
                weight_col=self.market_value_eur_col,
                filter_issuer_ids=df_only_sold['ISSUER_ID']
            )

            if val == 'ECAI':
                text += '  - {}: from {:.2f} (Sold) to {:.2f} (Purchased)\n'.format(
                    val,
                    wa_sold,  # previous quarter
                    wa_purch,  # current quarter
                )
            elif val == 'SHOCK_FACTOR':
                text += '  - {}: from {:.1%} (Sold) to {:.1%} (Purchased)\n'.format(
                    val,
                    wa_sold,  # previous quarter
                    wa_purch,  # current quarter
                )
        
        text = text.replace('SHOCK_FACTOR', 'Shock Factor')
        
        return text
    
    # def _helper_weighted_average(self, by: str,  bond_status: str = None):
    #     """Return Weighted Average for specific column (by parameter).

    #     Parameters
    #     ----------
    #     by : str
    #         Column to calculate Weighted Average
    #     bond_status : str, optional
    #         Bond status for latest REPORTING_DAT, by default None

    #     Notes
    #     -----
    #     bond_status is used to filter the bond status in latest reporting
    #     period. Those bonds that match the criteria are then matched 
    #     in the previous quarter.

    #     Returns
    #     -------
    #     pandas.Series
    #     """        
    #     df = self.calculate_movements()

    #     # # Rename ECAI rating of 9
    #     # df_raw.loc[df_raw['ECAI'] == 'Unrated', 'ECAI'] = 9
        
    #     if bond_status is not None:
    #         # If not None, then we get only the Bond IDs that were
    #         # SOLD, PURCHASED or UNCHANGED.
    #         df = df.loc[df['BOND_STATUS'] == bond_status].copy()

    #     df['HELPER_WA'] = df[by] * df[self.market_value_eur_col]
    #     df['HELPER_WA'] = df['HELPER_WA'].fillna(0)
    #     s = df.sum()
        
    #     return s['HELPER_WA'] / s[self.market_value_eur_col]
    
    def _get_shock_movement_text(self, to_html: bool = False) -> str:
        # Filter calculated movements using only latest date and unchanged
        df_new_unchanged = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],  # most recent quarter
            bond_status=['UNCHANGED'],     # only unchanged bond status
        )

        # Find Weighted Average Shocks
        wa_shock_old = self.datasets[-2].get_weighted_average(
            value_col='SHOCK_FACTOR',
            weight_col=self.market_value_eur_col,
            filter_issuer_ids=df_new_unchanged['ISSUER_ID']
        )
        
        wa_shock_new = self.datasets[-1].get_weighted_average(
            value_col='SHOCK_FACTOR',
            weight_col=self.market_value_eur_col,
            filter_issuer_ids=df_new_unchanged['ISSUER_ID']
        )

        text = '#### Shock (I)(II):\n'
        text += '- Portfolio Shock from '
        text += f'{wa_shock_old:.1%} to {wa_shock_new:.1%}'

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
                                 exclude_cr1: bool = True) -> pd.DataFrame:
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
        date : str
            date corresponding to the reporting date

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
        df = self.calculate_movements()

        if exclude_cr1:
            df = df.loc[df['COUNTERPARTY_RISK_TYPE_CODE'] != 'CR1'].copy()
        
        # Include column of 1, to count bonds by "by" argument.
        df['BOND_COUNT'] = np.where(df['SECTOR'].isna(), 0, 1)

        return df.groupby(by=['REPORTING_DAT', by_col], as_index=False).sum()

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
        # We only want to count bonds that are not CR1 since CR1 bonds
        # do not have any SCR penalties
        df = self.get_quarter_movements_by(by=by, exclude_cr1=True)

        by_col = df.columns[1]
        
        # Create filter for previous quarter
        filter_prev = df['REPORTING_DAT'] == self.datasets[-2].report_date
        
        # Create filter for current quarter
        filter_curr = df['REPORTING_DAT'] == self.datasets[-1].report_date
        
        df_old = df.loc[filter_prev].copy()
        df_new = df.loc[filter_curr].copy()

        # Merge everything together so we can have an Excel graph
        df_mrg = df_old.merge(df_new,
                              left_on=by_col,
                              right_on=by_col,
                              how='outer',
                              suffixes=("-"+self.datasets[-2].report_period,
                                        "-"+self.datasets[-1].report_period))

        # Keeping only MV and Count
        cols_to_keep = [
            col for col in df_mrg.columns if 'MARKET_VALUE_EUR' in col or 'BOND_COUNT' in col
        ]

        # Reverse as to have Market Value EUR at the front
        cols_to_keep.sort(reverse=True)

        # Insert by column at the beginning
        # get_quarter_movements_by will always aggregate data by REPORTING_DAT
        # and by_col, so by_col will always be second column. We use [1] to
        # access it.
        cols_to_keep.insert(0, by_col)

        return df_mrg[cols_to_keep].fillna(0)
    
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
            col = 'BOND_COUNT-'
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

        col_prev = col+self.datasets[-2].report_period
        col_curr = col+self.datasets[-1].report_period

        df['Diff'] = df[col_curr] - df[col_prev]

        # Create Figure object
        fig = go.Figure()

        # Add Previous Quarter Data
        #df[df.columns[0]] = first column of aggregated data.
        fig.add_trace(go.Scatter(x=df[df.columns[0]],
                                 y=df[col_prev],
                                 mode='lines+markers',
                                 name=col_prev,
                                 line={'color': CORPORATE_COLOR[0]}))
        
        # Add Current Quarter Data
        fig.add_trace(go.Scatter(x=df[df.columns[0]],
                                 y=df[col_curr],
                                 mode='lines+markers',
                                 name=col_curr,
                                 line={'color': CORPORATE_COLOR[1]}))
        
        # Add Difference QoQ
        fig.add_trace(go.Bar(x=df[df.columns[0]],
                             y=df['Diff'],
                             name='Difference QoQ',
                             marker_color=CORPORATE_COLOR[-1]))

        fig.update_layout(title=plot_title,
                          title_font=PLOT_TITLE_FONT_FORMAT,
                          hovermode='x',
                          template='plotly_white')
        if value == 'mv':
            fig.update_layout(yaxis_tickformat='.2s')
        
        if df.columns[0] == 'ECAI':
            fig.update_xaxes(type='category')

        return fig
    
    def _waterfall_helper(self) -> pd.DataFrame:
        df = self.get_scr_summary(reset_index=True)
        
        # Most recent 2 dates from DataFrame
        most_recent_data = df.nlargest(1, columns='REPORTING_DAT')

        scr_idx_prev = "SCR-"+self.datasets[-2].report_period
        scr_idx_curr = "SCR-"+self.datasets[-1].report_period

        cols_to_use = [
            col for col in most_recent_data.columns if 'Diff-' in col
        ]

        # Insert index for previous SCR at the beginning
        cols_to_use.insert(0, scr_idx_prev)
        
        # Insert index for current SCR at the end
        cols_to_use.append(scr_idx_curr)
        
        # Create empty DataFrame to append data for Graph
        df_wf = pd.DataFrame(columns=['Amounts', 'Measure'],
                             index=cols_to_use)

        # Append first value, which is SCR that we're coming from
        # Use values[0] to extract just number
        df_wf.loc[scr_idx_prev, :] = [
            self.datasets[-2].get_scr_sum(),
            'absolute']
        
        df_wf.loc[scr_idx_curr, :] = [
            self.datasets[-1].get_scr_sum(),
            'absolute']

        # Append all Diff- columns, representing each moving part of the SCR
        for diff_col in most_recent_data.columns:
            if 'Diff-' in diff_col:
                df_wf.loc[diff_col, :] = [
                    most_recent_data[diff_col].values[0],
                    'relative'
                ]

        return df_wf
    
    def plot_waterfall_movement_summary(self, to_html: bool = False):
        df = self._waterfall_helper()

        plot_val_text = []
        for val in df['Amounts'].values:
            # To show values in millions
            plot_val_text.append("{:,.2f}m".format(val / 1e6))

        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=df['Measure'],
            x=df.index,
            textposition="outside",
            textfont={'size': 16},
            text=plot_val_text,
            y=df['Amounts'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {
                "color": CORPORATE_VALUE_COLOR['negative']}},
            decreasing={"marker": {
                "color": CORPORATE_VALUE_COLOR['positive']}},
            totals={"marker": {"color": CORPORATE_VALUE_COLOR['neutral']}},
            showlegend=False,
        ))

        wf.update_layout(title_text="Equity SCR (QoQ)",
                         title_font=PLOT_TITLE_FONT_FORMAT,
                         waterfallgap=0.5,
                         template='plotly_white')
        
        # Create plot Range for better visibility, instead of starting plot
        # Y axis from zero 
        # 0.95 and 1.05 were chosen just for formatting purposes.
        y_axis_range = [df[df.index.str.contains('SCR')].min()[0] * 0.9,
                        df[df.index.str.contains('SCR')].max()[0] * 1.15]

        # Remove Y axis and set range to y_axis_range
        wf.update_yaxes(visible=False, range=y_axis_range)

        if to_html:
            return plotly.offline.plot(wf,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return wf
    
    def plot_market_value_and_scr(self, to_html: bool = False):

        df = self.get_scr_summary()

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index,
                       y=df[self.market_value_eur_col],
                       name="Market Value (EUR)",
                       line={'color': CORPORATE_COLOR[-1]}),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index,
                       y=df['SCR'],
                       name="SCR",
                       line={'color': CORPORATE_COLOR[-3]},
                       visible='legendonly'),
            secondary_y=False)

        # Add figure title
        fig.update_layout(title_text="Market Value, SCR and SCR:MV (ratio)",
                          title_font=PLOT_TITLE_FONT_FORMAT,
                          yaxis_tickformat=',.0f',
                          hovermode="x",
                          template='plotly_white')

        # Set x-axis title
        fig.update_xaxes(title_text="",
                         showgrid=False,
                         zeroline=False,
                         showline=False)

        # Set y-axes titles
        fig.update_yaxes(title_text="EUR (m)",
                         secondary_y=False,
                         tickformat='.4s')

        fig.add_trace(
            go.Scatter(x=df.index,
                        y=df['SCR : MV'],
                        name="SCR : MV",
                        line={'color': CORPORATE_COLOR[-2]}),
            secondary_y=True)

        fig.update_yaxes(title_text="Market Value : SCR",
                            tickformat='.2%',
                            showgrid=False,
                            secondary_y=True)

        if to_html:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return fig
    
    def get_report_name(self, file_extension: str) -> str:
        if self.is_scr_matching_fc_computed():
            report_type = '(Final)'
        else:
            report_type = '(Preliminary)'

        file_name = '{} Risk-{} {}.{}'.format(
            'Spread',
            get_report_period(self.dates_added[-1]),
            report_type,
            file_extension
        )

        return os.path.join(file_name)

    def generate_excel_report(self):
        filename = self.get_report_name('xlsx')
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
        df = self.calculate_movements()
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
        for i in self.datasets:
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
        df = self.get_scr_summary()
        
        (max_row, max_col) = df.shape

        cols_to_use = ['SCR',
                       'Diff',
                       'Diff-Purchased/Sold',
                       'Diff-Org Growth',
                       'Diff-Shock',
                       'Diff-FX',
                       'Diff %']

        df[cols_to_use].to_excel(excel_writer,
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
        rows_from_df = 3
        rows_merged = 21
        summary_text_rng = f"B{max_row + rows_from_df}:E{max_row + rows_merged}"

        ws.merge_range(
            summary_text_rng,
            self.get_movement_summary_text(),
            merge_format
        )

        # Formatting
        # SCR Summary Worksheet
        ws.set_column('B:B', 15)
        ws.set_column('C:H', 13, num_format)
        ws.set_column('I:I', 13, pct_format)
    
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
                        startrow=row,
                        index=False)
            
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

        templates_dir = os.path.join(os.path.dirname(__file__),
                                     'templates')
                                     
        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('spread_risk.html')

        filename = os.path.join(os.getcwd(),
                                self.get_report_name(file_extension='html'))
        
        # Dict with all elements to complete HTML report
        elem = self._get_html_elements()
        
        with open(filename, 'w') as fh:
            fh.write(template.render(
                current_period=self.datasets[-1].report_period,
                previous_period=self.datasets[-2].report_period,
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
        current_scr = self.datasets[-1].get_scr_sum()
        previous_scr = self.datasets[-2].get_scr_sum()
        scr_diff = current_scr - previous_scr
        scr_pct_ch = scr_diff / previous_scr
        
        elem = {}
        # First Box
        
        elem['current_scr'] = f"EUR {current_scr:,.0f}m"
        
        # Second Box
        
        elem['previous_scr'] = f"EUR {previous_scr:,.0f}m"
        elem['previous_scr_in_m'] = "EUR {:,.0f}m".format(previous_scr / 1e6)
        elem['total_diff_qoq'] = "EUR {:+,.2f}m".format(
            (current_scr - previous_scr) / 1e6
        )

        elem['scr_diff_pct'] = scr_pct_ch
        elem['scr_diff_pct_formatted'] = "{:+,.1%}".format(scr_pct_ch)
        elem['scr_increase'] = scr_pct_ch > 0

        elem['scr_movement_summary_text'] = self.get_movement_summary_text(
            to_html=True)
        
        elem['waterfall_movement_summary'] = self.plot_waterfall_movement_summary(
            to_html=True)
        
        elem['mv_scr_plot'] = self.plot_market_value_and_scr(to_html=True)
        
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
