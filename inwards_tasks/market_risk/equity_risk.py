from typing import List, Union
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline
import webbrowser
from markdown2 import Markdown

try:
    from .utils import CORPORATE_VALUE_COLOR
    from .utils import PLOT_TITLE_FONT_FORMAT
    from .utils import CORPORATE_COLOR
    from .utils import add_attributes_to_html_table
    from .utils import get_report_period
    from .utils import produce_unique_rows

    from .market_risk import MarketRisk
    from .risk_data import RiskData
    from .fc_computed_amounts import import_fc_computed_amounts
    from .eiopa_shocks import EIOPAShocks
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from utils import CORPORATE_VALUE_COLOR
    from utils import PLOT_TITLE_FONT_FORMAT
    from utils import CORPORATE_COLOR
    from utils import add_attributes_to_html_table
    from utils import get_report_period
    from utils import produce_unique_rows

    from market_risk import MarketRisk
    from fc_computed_amounts import import_fc_computed_amounts
    from eiopa_shocks import EIOPAShocks


class EquityRiskData(RiskData):
    """Class holding Equity Risk Data."""
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self._data = data
        self.DATA_NAME = 'Equity'
    
    @property
    def market_value(self) -> int:
        return self.data['SOL_II_VAL'].sum()
    
    @property
    def market_value_eur(self) -> int:
        return self.data['SOL_II_VAL_EUR'].sum()
    
    @property
    def overall_shock(self) -> float:
        return self.data['SCR'].sum() / self.data['SOL_II_VAL_EUR'].sum()
    
    def _filter_data(
        self,
        filter_issuer_ids: Union[None, List[str]]
    ) -> pd.DataFrame:

        df = self.data
        if filter_issuer_ids is not None:
            # If there's a list of ISSUER_IDS, then filter data
            df = df.loc[df['ISSUER_ID'].isin(filter_issuer_ids)]

        return df
    
    def get_market_value_sum(
        self,
        eur: bool = False,
        filter_issuer_ids: Union[None, List[str]] = None
    ) -> int:

        if eur:
            mv_col = 'SOL_II_VAL_EUR'
        else:
            mv_col = 'SOL_II_VAL'

        df = self._filter_data(filter_issuer_ids=filter_issuer_ids)

        return df[mv_col].sum()
    
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

class EquityRisk(MarketRisk):
    DATE_COLS = ['REPORTING_DAT',
                 'PURCHASE_DAT']

    UNUSED_COLS = ['SUBPORTFOLIO',
                   'U_ECAI']

    ISSUER_DATA_COLS = ['REPORTING_DAT',
                        'ISSUER_ID',
                        'ALIAS_NAME',
                        'COUNTRY_CODE',
                        'SECTOR']

    ID_COLUMNS = ['EQUITY_ID',
                  'PURCHASE_DAT',
                  'ISSUER_ID']

    SORTING_COLS = ['EQUITY_ID',
                    'PURCHASE_DAT',
                    'ISSUER_ID',
                    'REPORTING_DAT']

    def __init__(self) -> None:
        super().__init__()
        
        self._fc_computed = None
        self._issuer_data = None

        self.shocks = EIOPAShocks()
    
    @property
    def data(self) -> Union[None, pd.DataFrame]:
        # Make copy to not alter original
        df = self._data.copy()

        # Drop all ISSUER_DATA columns except issuer ID and reporting date
        df.drop(self.ISSUER_DATA_COLS[2:], axis=1, inplace=True)

        # Merge in ALIAS_NAME back to self._data (main DataFrame)
        return df.merge(self.issuer_data,
                        how='left',
                        on='ISSUER_ID')
    
    @data.setter
    def data(self, raw_data: pd.DataFrame) -> None:

        # Process DF to calculate SCR
        df_processed = self._process_equity_data(raw_data)

        # add processed data to self.datasets, which is an attribute from
        # Parent Class MarketRisk
        self.datasets = EquityRiskData(df_processed)

        # Concatenate with data already in Model
        df = pd.concat([self._data, df_processed])

        # Drop any duplicates, in case we add twice same data
        df.drop_duplicates(inplace=True)

        data_ts = df_processed['REPORTING_DAT'].unique()[0]

        report_period = get_report_period(data_ts)
        
        # This makes sure we have a unique entry (based on IDENTIFIER_COLS)
        # per reporting date
        self._data = produce_unique_rows(df, self.ID_COLUMNS)

        print("Data added for ", report_period)

    @property
    def fc_computed(self) -> Union[None, pd.DataFrame]:
        return self._fc_computed
    
    @fc_computed.setter
    def fc_computed(self, df: pd.DataFrame) -> None:
        # Concat all quarters together
        self._fc_computed = pd.concat([self._fc_computed, df])

        self._fc_computed.drop_duplicates(inplace=True)
        
        # Sort from oldest to newest to make things easier
        self._fc_computed.sort_values(by='REPORTING_DAT', inplace=True)

    @property
    def periods(self) -> dict:
        dates = self.data['REPORTING_DAT'].unique()
        dates.sort()
        dates

        dates_dict = {}
        # -1 is the most recent dates, and -2 the second most recent
        dates_dict['new'] = get_report_period(dates[-1])
        dates_dict['old'] = get_report_period(dates[-2])

        return dates_dict
    
    def import_data(self, folderpath: str) -> None:
        try:
            eq_filepath = os.path.join(folderpath, 'TBSL_EQUITIES.xlsx')

            # load excel file into model
            df = pd.read_excel(eq_filepath,
                               dtype={'EQUITY_TYP': str,
                                      'PURCHASE_DAT': str,
                                      'ISSUER_ID': str,
                                      'ECAI': str},
                               engine='openpyxl')
        except FileNotFoundError:
            eq_filepath = os.path.join(folderpath, 'TBSL_EQUITIES.txt')

            # load excel file into model
            df = pd.read_csv(eq_filepath, sep='\t')

            # Convert columns to string, if file is txt
            cols_to_str = ['EQUITY_TYP', 'ISSUER_ID', 'ECAI']
            for col in cols_to_str:
                df[col] = df[col].astype(str)

        # Assing new data to self.data. The setter method will take care of
        # processing the data
        self.data = df
        self.unique_identifiers = df

        fc_comp_filepath = os.path.join(folderpath, 'FC_COMPUTED_AMOUNTS.txt')
        self.fc_computed = import_fc_computed_amounts(fc_comp_filepath)

    def get_shocks(self, date: str, shock_type: str = None) -> pd.Series:
        
        return self.shocks.get_shock_factors(date=date,
                                             shock_type=shock_type)
    
    def _process_equity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # In case data has extra lines, and they are imported with NaN
        df.dropna(axis='index', how='all', inplace=True)
        
        # Timestamp data
        if 'REPORTING_DAT' not in df.columns:
            msg_err = "REPORTING_DAT column has to be in equity data. "
            msg_err += "Please include this column with the correspoding "
            msg_err += "reporting date."
            raise ValueError(msg_err)

        # Normalize Date columns
        for col in self.DATE_COLS:
            df[col] = pd.to_datetime(df[col], dayfirst=True)

        if 'EQUITY_CATEGORY' in df.columns and 'EQUITY_TYP' in df.columns:
            # Add Equity Category to DataFrame.
            purchase_after_2016 = (
                df['PURCHASE_DAT'] > pd.to_datetime('1/1/2016', dayfirst=True)
            )

            df['equityCategory'] = np.where(
                purchase_after_2016,
                # If purchased AFTER 2016
                df['EQUITY_CATEGORY'] + df['EQUITY_TYP'],
                # Else
                'TRANSITIONAL' + df['EQUITY_TYP']
            )
        
        # Remove unneeded columns
        df.drop(columns=self.UNUSED_COLS, inplace=True)

        # insert shocks
        df = df.merge(self.get_shocks(df['REPORTING_DAT'].unique()[0]),
                      left_on='EQUITYCATEGORY',
                      right_index=True,
                      how='left')
        
        # Calculate raw SCR by applying Shock to Market Value in EUR
        df['SHOCKED_VAL'] = df['SHOCK_FACTOR'] * df['SOL_II_VAL_EUR']

        df['DIVERS_BENFT'] = self._calculate_diversification_benefit(df)

        df['SCR'] = df['SHOCKED_VAL'] * df['DIVERS_BENFT']

        return df
    
    @staticmethod
    def _calculate_diversification_benefit(df: pd.DataFrame):
        df_pvt = df.pivot_table(values='SHOCKED_VAL',
                                columns='EQUITY_TYP',
                                aggfunc=np.sum).rename_axis(None)
        
        # ** 0.5 -> Square Root of the Sums
        df_pvt['SCR_RAW'] = df_pvt['1'] + df_pvt['2']
        df_pvt['SCR'] = (
            df_pvt['1'] ** 2
            + df_pvt['2'] ** 2
            + 1.5 * df_pvt['1'] * df_pvt['2']
        ) ** 0.5

        # Comparing SCR without diversification benefits (SCR_RAW)
        # and SCR with diversification benefits
        result = df_pvt['SCR'] / df_pvt['SCR_RAW']

        return result[0]
    
    def calculate_movements(
        self,
        only_latest_n: Union[None, int] = None,
        mv_diff: bool = False
    ) -> pd.DataFrame:
        """Calculate movements from Quarter to Quarter.   

        Parameters
        ----------
        only_latest_n : Union[None, int], optional
            if int, then filter latest filter_n_periods, by default None.
        mv_diff : bool, optional
            if True, then calculate diff() using Market Value column.

        Returns
        -------
        pd.DataFrame
            DataFrame with all the movements from Quarter to Quarter.
        """        
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
        df['SOL_II_VAL_EUR'].fillna(0, inplace=True)
        
        if mv_diff is True:
            df['Diff-MV'] = self.market_value_diff(df)

        df['BOND_STATUS'] = self.add_bond_status(df)
        df['Diff'] = self.calculate_scr_movement(df)
        df['Diff-Purchased/Sold'] = self.calculate_scr_purch_sold_movements(df)
        df['Diff-Org Growth'] = self.calculate_scr_organic_movements(df)
        df['Diff-Shock'] = self.calculate_scr_shock_movements(df)
        df['Diff-FX'] = self.calculate_scr_fx_movements(df)
        df['Diff-Div Benefit'] = self.calculate_scr_diversification_movements(df)
        return df
    
    def market_value_diff(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(by=self.ID_COLUMNS)['SOL_II_VAL_EUR'].diff().fillna(0)
    
    @staticmethod
    def add_bond_status(df: pd.DataFrame) -> pd.Series:
        prev_row_is_same_eq_id = df['EQUITY_ID'].eq(df['EQUITY_ID'].shift(1))
        prev_row_is_same_purch_dat = df['PURCHASE_DAT'].eq(df['PURCHASE_DAT'].shift(1))
        prev_row_is_same_issuer_id = df['ISSUER_ID'].eq(df['ISSUER_ID'].shift(1))
        prev_row_is_same = ((prev_row_is_same_eq_id)
                            & (prev_row_is_same_purch_dat)
                            & (prev_row_is_same_issuer_id))
        
        is_nothing = (
            (df['SOL_II_VAL'].shift(1).isna()) 
            & (df['SOL_II_VAL'].isna())
        )

        bond_sold = ((df['SOL_II_VAL'].shift(1).notna()) 
                    & df['SOL_II_VAL'].isna())
        
        bond_purchased = ((df['SOL_II_VAL'].shift(1).isna()) 
                          & df['SOL_II_VAL'].notna())
        
        bond_unchanged = ((df['SOL_II_VAL'].shift(1).notna()) 
                          & df['SOL_II_VAL'].notna())
        


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
                            (df['SOL_II_VAL'] - df['SOL_II_VAL'].shift(1))
                            * df['SHOCK_FACTOR'].shift(1)
                            * df['DIVERS_BENFT'].shift(1)
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
                            * df['SOL_II_VAL']
                            * df['DIVERS_BENFT'].shift(1)
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
                            * df['SOL_II_VAL']
                            * df['DIVERS_BENFT'].shift(1)
                            * (1 / df['FX_RATE'] - 1 / df['FX_RATE'].shift(1))
                        ),
                        0)
    
    def calculate_scr_diversification_movements(self,
                                                df: pd.DataFrame) -> pd.Series:
        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        return np.where(filter_unchanged,
                         # ( DivBenefit(new) - DivBenefit(old) ) * MV(new)
                        # * Shock Factor(old)
                        # -------------------------------------
                        # FX_rate(old)
                        (
                            (df['DIVERS_BENFT'] - df['DIVERS_BENFT'].shift(1))
                            * df['SOL_II_VAL']
                            * df['SHOCK_FACTOR']
                            / df['FX_RATE']
                        ),
                        0)
    
    def get_scr_summary(self, reset_index: bool = False) -> pd.DataFrame:
        df = self.calculate_movements()
        df_grp = df.groupby('REPORTING_DAT').sum()
        df_grp['Diff %'] = df_grp['SCR'].pct_change()
        df_grp['SCR : MV'] = df_grp['SCR'] /  df_grp['SOL_II_VAL_EUR']

        # Dropping cols because we are summing all rows, so the individual 
        # values lose any meaning
        cols_to_drop = ['FX_RATE',
                        'SHOCK_FACTOR',
                        'DIVERS_BENFT']
        df_grp.drop(columns=cols_to_drop, inplace=True)

        if reset_index:
            df_grp.reset_index(inplace=True)

        return df_grp
    
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
    
    def _waterfall_helper(self) -> pd.DataFrame:
        df = self.calculate_movements()
        
        # Aggregate all that for Waterfall graph
        df_grp = df.groupby('REPORTING_DAT', as_index=False).sum()
        
        # Most recent 2 dates from DataFrame
        recent_dates = df_grp['REPORTING_DAT'].nlargest(2)
        
        # Assign current (most recent) date
        curr_date = pd.to_datetime(recent_dates.values[0])
        
        # Assign previous (2nd most recent) date
        prev_date = pd.to_datetime(recent_dates.values[1])

        # Create empty DataFrame to append data for Graph
        df_wf = pd.DataFrame(columns=['Amounts', 'Measure'])

        # Append first value, which is SCR that we're coming from
        # Use values[0] to extract just number
        df_wf.loc[f'SCR-{get_report_period(prev_date)}', :] = [
            df_grp.loc[df_grp['REPORTING_DAT'] == prev_date, 'SCR'].values[0],
            'absolute'
        ]

        filter_curr_date = df_grp['REPORTING_DAT'] == curr_date
        
        # Append all Diff- columns, representing each moving part of the SCR
        for diff_col in df_grp.columns:
            if 'Diff-' in diff_col:
                df_wf.loc[diff_col, :] = [
                    df_grp.loc[filter_curr_date, diff_col].values[0],
                    'relative'
                ]

        df_wf.loc[f'SCR-{get_report_period(curr_date)}', :] = [
            df_grp.loc[filter_curr_date, 'SCR'].values[0],
            'total'
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
    
    def plot_market_value_and_scr(self,
                                  ratio: bool = False,
                                  to_html: bool = False):

        df = self.get_scr_summary()

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index,
                       y=df['SOL_II_VAL_EUR'],
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

        if ratio:
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
    
    def plot_market_value_by_ecai(self, to_html: bool = False):
        df_raw = self.data.groupby(
            ['REPORTING_DAT', 'ECAI']).sum().reset_index(level=1)

        # dates_added[-2:] represent the latest 2 dates added to model
        df = df_raw.loc[self.dates_added[-2:]].copy()

        # ECAI 9 should be named as Unrated, as per request
        df.loc[df['ECAI'] == '9', 'ECAI'] = 'Unrated'

        df.rename({'SOL_II_VAL_EUR': 'Market Value (EUR)'},
                  axis=1, inplace=True)

        # Applied to index to make legend readable in plot
        df.index = df.index.map(get_report_period)

        fig = px.bar(
            df,
            x='ECAI',
            y='Market Value (EUR)',
            color=df.index,
            barmode='group',
            template='plotly_white',
            labels={'REPORTING_DAT': ''},
            color_discrete_sequence=[CORPORATE_COLOR[0], CORPORATE_COLOR[1]],
        )

        fig.update_yaxes(title_text='Market Value (EUR)')

        fig.update_xaxes(type='category',
                         categoryorder='category ascending',
                         title_text='ECAI Rating')

        fig.update_layout(title_text='Market Value (EUR) by ECAI',
                          title_font=PLOT_TITLE_FONT_FORMAT)

        if to_html:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return fig

    def plot_movement_by_issuer(self, to_html: bool = False):
        # Get calculated movements for only latest 2 quarters
        df_mov = self.calculate_movements(only_latest_n=2)

        # Group that data by ISSUER_ID and REPORTING_DAT so we can plot
        # data more easily
        df = df_mov.groupby(by=['ISSUER_ID',
                                'ALIAS_NAME',
                                'REPORTING_DAT'], as_index=False).sum()

        # Sort values so that plot has largest movements to the left
        df.sort_values('Diff', ascending=False, inplace=True)

        # Renaming for plot reason
        df.rename(columns={'SOL_II_VAL_EUR': 'Market Value (EUR)'},
                  inplace=True)
        
        # Filter out small SCR movements between 20k and -20k
        df = df.loc[(df['Diff'] > 20000) | (df['Diff'] < -20000)]

        # Remove deltas that are too small, in particular FX
        # and Diversification Benefit
        df.drop(columns=['Diff-FX', 'Diff-Div Benefit'], inplace=True)

        fig = px.bar(
            df,
            x="ISSUER_ID",
            y=[col for col in df.columns if 'Diff-' in col],
            template='plotly_white',
            labels={'value': 'Amount (EUR)',
                    'ALIAS_NAME': ''},
            hover_data={'Market Value (EUR)': ':.2s',
                        'ALIAS_NAME': True},
            color_discrete_sequence=[CORPORATE_COLOR[-1],
                                     CORPORATE_COLOR[-2],
                                     CORPORATE_COLOR[-3],
                                     CORPORATE_COLOR[-4],
                                     CORPORATE_COLOR[-5]]
        )

        # Remove Diff-, as per request
        for dat in fig.data:
            dat.name = dat.name.replace("Diff-", "")

        plot_title = 'Top Positive and Negative Movements (SCR) '
        plot_title += "by Issuer"
        plot_title += "<br><sup>"
        plot_title += "Only values with movement greated than 20k "
        plot_title += "(positive and negative)"
        plot_title += "</sup>"

        fig.update_layout(title_text=plot_title,
                          title_font=PLOT_TITLE_FONT_FORMAT,
                          height=700,
                          xaxis_tickangle=-45,
                          legend_title_text='Deltas')

        if to_html:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return fig
    
    def plot_top_issuers_by_market_value(self, n: int, to_html: bool = False):
        # First, calculate movements using only latest 2 quarters
        df_mov = self.calculate_movements(only_latest_n=2)

        # Group by alias name to simplify plot
        df = df_mov.groupby(
            by=['ALIAS_NAME', 'REPORTING_DAT'], as_index=False).sum()

        # Reset index for diff function
        df.reset_index(drop=True, inplace=True)

        # Create filter to check that code is applying diff to correct rows
        # meaning row above is the same as current row
        filter_same_as_above = df['ALIAS_NAME'] == df['ALIAS_NAME'].shift(1)
        df['MV Diff'] = np.where(filter_same_as_above,
                                 df['SOL_II_VAL_EUR'].diff(),
                                 0)

        # Add PERIOD column for plot legend
        df['PERIOD'] = df['REPORTING_DAT'].apply(
            lambda x: get_report_period(x))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Slice DataFrame to keep only data for current quarter
        df_new = df.loc[df['REPORTING_DAT'] == self.dates_added[-1]].copy()
        df_new.sort_values('SOL_II_VAL_EUR', ascending=False, inplace=True)
        # Keep only top 10 in Market Value
        df_new = df_new.nlargest(n, 'SOL_II_VAL_EUR')

        # Slice DataFrame to keep only data for previous quarter
        df_old = df.loc[df['REPORTING_DAT'] == self.dates_added[-2]].copy()
        df_old.sort_values('SOL_II_VAL_EUR', ascending=False, inplace=True)

        # Keep only top 10 in Market Value from current quarter
        df_old = df_old.loc[df_old['ALIAS_NAME'].isin(df_new['ALIAS_NAME'])]

        # Add traces for previous Quarter
        fig.add_trace(
            go.Bar(x=df_old['ALIAS_NAME'],
                   y=df_old['SOL_II_VAL_EUR'],
                   name=df_old['PERIOD'].unique()[0],
                   marker_color=CORPORATE_COLOR[0]),
            secondary_y=False)

        # Add traces for current Quarter
        fig.add_trace(
            go.Bar(x=df_new['ALIAS_NAME'],
                   y=df_new['SOL_II_VAL_EUR'],
                   name=df_new['PERIOD'].unique()[0],
                   marker_color=CORPORATE_COLOR[1]),
            secondary_y=False)
        
        fig.update_layout(title='Top Issuers by Market Value (EUR)',
                          title_font=PLOT_TITLE_FONT_FORMAT,
                          template='plotly_white')

        if to_html:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return fig
    
    def plot_largest_market_value_movements(self,
                                            position: str,
                                            n: int,
                                            to_html: bool = False):
        """Plot top or bottom n Issuers by Market Value diff.

        Parameters
        ----------
        position : str
            'top' (pd.nlargest) or 'bottom' (pd.nsmallest)
        n : int
            top or bottom n companies by Market Value difference
            between quarter will be selected.
        """
        df_raw = self.calculate_movements(only_latest_n=2, mv_diff=True)
        df_sum = df_raw.groupby(['ISSUER_ID', 'ALIAS_NAME'], as_index=False).sum()

        mv_diff_col = 'Market Value QoQ (EUR)'
        df_sum.rename(columns={'Diff-MV': mv_diff_col}, inplace=True)

        if position == 'top':
            df = df_sum.nlargest(n=n, columns=mv_diff_col)
        elif position == 'bottom':
            # it is visually better to reverse the dataframe when showing
            # negative movements (used [::-1])
            df = df_sum.nsmallest(n=n, columns=mv_diff_col).iloc[::-1]
        
        plot_title = "{} {} in Market Value Holding Difference (QoQ)".format(
            position.capitalize(),
            n
        )

        fig = px.bar(df,
                     x="ALIAS_NAME",
                     y=mv_diff_col,
                     title=plot_title,
                     color_discrete_sequence=[CORPORATE_COLOR[-1]],
                     hover_data={mv_diff_col: ':.3s'})

        fig.update_layout(title_font=PLOT_TITLE_FONT_FORMAT,
                          template='plotly_white')

        fig.update_xaxes(title_text='')

        if to_html:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return fig

    
    def get_holding_by_issuer_df(self, to_html: bool = False) -> pd.DataFrame:
        # Keep only latest 2 datasets
        df = self.calculate_movements(only_latest_n=2)
        
        # fillna() so that when grouping, code doesn't exclude NaN values
        df['SOL_II_VAL_EUR'].fillna(0, inplace=True)

        df = df.groupby(by=['ALIAS_NAME', 'REPORTING_DAT']).sum()
        
        # Just so we create an empty column
        df['MV-QoQ'] = 0

        # Column order
        cols_to_keep = ['SOL_II_VAL_EUR', 'MV-QoQ', 'SCR', 'Diff']
        
        df_final = df[cols_to_keep].copy()

        # Calculate Market Value QoQ movement
        df_final['MV-QoQ'] = df_final['SOL_II_VAL_EUR'].groupby('ALIAS_NAME').diff()
        
        # # fillna below is producing an error because it converts some values
        # # to strings, and strings can't have the format "{:,.0f}"
        # df_final['MV-QoQ'].fillna('', inplace=True)

        # Reset index so that we can format table for HTML
        df_final.reset_index(inplace=True)

        # Rename columns to make user friendly
        df_final.rename(columns={'REPORTING_DAT': 'Report Date',
                                 'SOL_II_VAL_EUR': 'Market Value (EUR)',
                                 'SCR': 'SCR (EUR)',
                                 'Diff': 'SCR-QoQ'},
                        inplace=True)

        if to_html:
            format_dict = {}
            for col in df_final.columns:
                if ('QoQ' in col) or ('(EUR)' in col):
                    format_dict[col] = "{:,.0f}"
                elif col == 'Report Date':
                    format_dict[col] = "{:%Y-%m-%d}"

            df_formatted = df_final.style.format(format_dict).to_html()

            # Columns that contain numbers, for sorting purposes
            num_cols = [
                # If QoQ or (EUR) are in column headers, it's a number's column
                col for col in df_final.columns if 'QoQ' in col or '(EUR)' in col
            ]

            return add_attributes_to_html_table(html=df_formatted,
                                                data_search=True,
                                                data_pagination=True,
                                                data_sorter_cols=num_cols,
                                                remove_nan=True)
        else:
            return df_final
        
    def get_movement_summary_text(self, to_html: bool = False) -> str:

        text = self._get_overall_movement_text(to_html=to_html)

        text += self._get_purchased_sold_movement_text(to_html=to_html)

        text += self._get_organic_movement_text(to_html=to_html)

        text += self._get_shock_movement_text(to_html=to_html)

        text += self._report_obs(to_html=to_html)

        return text
    
    def _get_overall_movement_text(self, to_html: bool = False) -> str:
        
        df = self.get_scr_summary()
        df.loc[self.dates_added[-1], 'SCR']
        
        # Overall Calculations
        prev_scr = df.loc[self.dates_added[-2], 'SCR']
        curr_scr = df.loc[self.dates_added[-1], 'SCR']
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
    
    def _get_purchased_sold_movement_text(self, to_html: bool = False) -> str:
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
                weight_col='SOL_II_VAL_EUR',
                filter_issuer_ids=df_only_purch['ISSUER_ID']
            )

            wa_sold = self.datasets[-2].get_weighted_average(
                value_col=val,
                weight_col='SOL_II_VAL_EUR',
                filter_issuer_ids=df_only_sold['ISSUER_ID']
            )

            text += '  - {}: from {:.2f} (Sold) to {:.2f} (Purchased)\n'.format(
                val,
                wa_sold,  # previous quarter
                wa_purch,  # current quarter
            )
        
        text = text.replace('SHOCK_FACTOR', 'Shock Factor')
        
        return text
    
    def _get_organic_movement_text(self, to_html: bool = False) -> str:

        # Only bonds that were not sold or purchased
        mv_eur_current = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],
            bond_status=['UNCHANGED']
        )['SOL_II_VAL_EUR'].sum()

        mv_eur_previous = self.filter_calculated_movements(
            dates=[self.dates_added[-2]],
            bond_status=['UNCHANGED']
        )['SOL_II_VAL_EUR'].sum()

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
        
    def _get_shock_movement_text(self, to_html: bool = False) -> str:
        # Filter calculated movements using only latest date and unchanged
        df_new_unchanged = self.filter_calculated_movements(
            dates=[self.dates_added[-1]],  # most recent quarter
            bond_status=['UNCHANGED'],     # only unchanged bond status
        )

        # Find Weighted Average Shocks
        wa_shock_old = self.datasets[-2].get_weighted_average(
            value_col='SHOCK_FACTOR',
            weight_col='SOL_II_VAL_EUR',
            filter_issuer_ids=df_new_unchanged['ISSUER_ID']
        )
        
        wa_shock_new = self.datasets[-1].get_weighted_average(
            value_col='SHOCK_FACTOR',
            weight_col='SOL_II_VAL_EUR',
            filter_issuer_ids=df_new_unchanged['ISSUER_ID']
        )

        text = '#### Shock:\n'
        text += '- Portfolio Shock from '
        text += f'{wa_shock_old:.1%} to {wa_shock_new:.1%}'

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    @staticmethod
    def _report_obs(to_html: bool = False) -> str:
        text = '<br>\n<sup>(I)</sup> Only Bonds that were not purchased or sold.\n'
        text += '\n<sup>(II)</sup> SCR amounts shown are after Diversification Benefits.\n'

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
        
    def is_matching_fc_computed(self):
        # Produces a DF with Reporting Date as Index, with only SCR column
        df_data_grp = self.data.groupby('REPORTING_DAT').sum()['SCR']

        # Merge in Grouped data above into FC_COMPUTED DataFrame.
        # how=outer so that we get NaN if there's no available FC_COMPUTED
        # data for specific period.
        df = self.fc_computed.merge(df_data_grp,
                                    left_index=True,
                                    right_index=True,
                                    how='outer')
        
        df['Match'] = df['COMPUTED_AMT_EUR'].round(1) == df['SCR'].round(1)
        df['Match'].all()
        
        if df['COMPUTED_AMT_EUR'].isna().any():
            df_with_na = df.loc[df['COMPUTED_AMT_EUR'].isna()]
            # If any value in column is NaN, then return False
            dates_not_validated = [
                dat.strftime('%Y-%B') for dat in df_with_na.index]

            print(f"The following dates don't have FC_COMPUTED amounts: {dates_not_validated}")
            return False
        elif not df['Match'].all():
            # If not all values in columm Match are True, then something is
            # not matching
            vals_not_matching = df.loc[~df['Match']]

            print("The following amounts are not matching with FC_COMPUTED:")
            for _, v in vals_not_matching.iterrows():
                print("REPORTING_DAT: ", v['REPORTING_DAT'])
                print("- FC_COMPUTED:", v['COMPUTED_AMT_EUR'])
                print("- Manual Calculated SCR:", v['SCR'])
            return False
        return True
        
    
    def get_report_name(self, file_extension: str) -> str:
        if self.is_matching_fc_computed():
            report_type = '(Final)'
        else:
            report_type = '(Preliminary)'

        file_name = '{} Risk-{} {}.{}'.format(
            'Equity Risk',
            self.periods['new'],
            report_type,
            file_extension
        )

        return os.path.join(file_name)

    def generate_html_report(self):
        from jinja2 import Environment, FileSystemLoader

        templates_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'templates'
        )

        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('equity_risk.html')

        filename = os.path.join(os.getcwd(),
                                self.get_report_name(file_extension='html'))

        # Dict with all elements to complete HTML report
        elem = self._get_html_elements()

        with open(filename, 'w') as fh:
            fh.write(template.render(
                checked_with_group_data=self.is_matching_fc_computed(),
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
                mv_by_ecai_plot=elem['mv_by_ecai_plot'],
                movement_by_issuer_plot=elem['movement_by_issuer_plot'],
                top_issuers_by_mv=elem['top_issuers_by_mv'],
                top_mv_movements_top=elem['top_mv_movements_top'],
                top_mv_movements_bottom=elem['top_mv_movements_bottom'],
                # Holding Section
                df_holding=elem['df_holding'],
                # Shock Section
                plot_shock_over_time=elem['plot_shock_over_time'],
                df_shock_over_time=elem['df_shock_over_time']
            ))

        print(f"HTML report created at {filename}.")
        webbrowser.open(filename, new=2)

    def _get_html_elements(self) -> dict:
        df_scr_summ = self.get_scr_summary()
        elem = {}
        
        # First Box
        elem['current_scr'] = "EUR {:,.0f}".format(
            df_scr_summ.loc[self.dates_added[-1], 'SCR'])

        # Second Box
        elem['previous_scr'] = "EUR {:,.0f}m".format(
            df_scr_summ.loc[self.dates_added[-2], 'SCR'])

        elem['previous_scr_in_m'] = "EUR {:,.1f}m".format(
            df_scr_summ.loc[self.dates_added[-2], 'SCR'] / 1e6)

        elem['total_diff_qoq'] = "EUR {:+,.1f}m".format(
            df_scr_summ.loc[self.dates_added[-1], 'Diff'] / 1e6
        )

        elem['scr_diff_pct'] = df_scr_summ.loc[self.dates_added[-1], 'Diff %']

        elem['scr_diff_pct_formatted'] = "{:+,.1%}".format(
            elem['scr_diff_pct'])
        
        elem['scr_increase'] = elem['scr_diff_pct'] > 0

        elem['scr_movement_summary_text'] = self.get_movement_summary_text(
            to_html=True)
        
        elem['waterfall_movement_summary'] = self.plot_waterfall_movement_summary(
            to_html=True)

        elem['mv_scr_plot'] = self.plot_market_value_and_scr(ratio=True,
                                                             to_html=True)

        elem['mv_by_ecai_plot'] = self.plot_market_value_by_ecai(
            to_html=True)

        elem['movement_by_issuer_plot'] = self.plot_movement_by_issuer(
            to_html=True)

        elem['top_issuers_by_mv'] = self.plot_top_issuers_by_market_value(
            10, to_html=True)

        elem['top_mv_movements_top'] = self.plot_largest_market_value_movements(
            position='top', n=10, to_html=True)

        elem['top_mv_movements_bottom'] = self.plot_largest_market_value_movements(
            position='bottom', n=10, to_html=True)

        # ---------------------------------------------------------------------
        # Holding section
        elem['df_holding'] = self.get_holding_by_issuer_df(to_html=True)

        # ---------------------------------------------------------------------
        # Shock section
        elem['plot_shock_over_time'] = self.shocks.plot_shocks(
            to_html=True)
        elem['df_shock_over_time'] = self.shocks.get_shocks_by_date(
            to_html=True)

        #elem['maturity_analysis_plots'] = self._maturity_analysis_plots_for_html()

        return elem
    
    def generate_excel_report(self):
        filename = self.get_report_name(file_extension='xlsx')
        writer = pd.ExcelWriter(filename,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')
        
        # Create Sheet with SCR Summary
        self._scr_summary_to_excel(excel_writer=writer)

        # Create Sheet with Quarter Movements
        self._quarter_movements_to_excel(excel_writer=writer)

        # Create Sheets with Raw Bond Data
        self._raw_data_to_excel(excel_writer=writer)

        writer.save()
    
    def _scr_summary_to_excel(self, excel_writer):
        sheetname = 'SCR Summary'

        scr_summary_df = self.get_scr_summary(reset_index=True)
        scr_summary_df.to_excel(excel_writer,
                                sheet_name=sheetname,
                                startcol=0,
                                startrow=0,
                                index=False)

        # DF to Excel Table
        (max_row, max_col) = scr_summary_df.shape

        column_settings = [
            {'header': column} for column in scr_summary_df.columns]

        excel_writer.sheets[sheetname].add_table(
            0, 0, max_row, max_col - 1, {'columns': column_settings}
        )

        # Get the xlsxwriter workbook and worksheet objects.
        wb = excel_writer.book
        num_format = wb.add_format({'num_format': '#,##0'})
        pct_format = wb.add_format({'num_format': '0.00%'})
        merge_format = wb.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'top'
        })

        ws = excel_writer.sheets[sheetname]

        rows_from_df = 3
        rows_merged = 21
        summary_text_rng = f"B{max_row + rows_from_df}:E{max_row + rows_merged}"
        
        # Mini text processing because we have HTML elements in the text
        summary_text = self.get_movement_summary_text()
        summary_text = summary_text.replace('<sup>', '')
        summary_text = summary_text.replace('</sup>', '')
        
        # Write explanation to cell below
        ws.merge_range(summary_text_rng, summary_text, merge_format)

        # Formatting
        # SCR Summary Worksheet
        ws.set_column('A:A', 17)              # Reporting Date
        ws.set_column('B:D', 17, num_format)
        ws.set_column('E:F', 13, num_format)  # SCR, Diff
        ws.set_column('G:G', 20, num_format)  # Diff- Purchased/Sold
        ws.set_column('H:H', 16, num_format)  # Diff-Org Growth
        ws.set_column('I:J', 12, num_format)  # Diff-FX
        ws.set_column('K:K', 16, num_format)  # Diff-Div Benefit
        ws.set_column('L:M', 12, pct_format)  # DIff %, SCR : MV
    
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
        ws.set_column('A:E', 17)
        ws.set_column('F:F', 13)  # EQUITY_TYP
        ws.set_column('G:G', 13)  # ECAI
        ws.set_column('H:H', 12, num_format)  # SOL_II_VAL (market value)
        ws.set_column('I:I', 12)  # CURRENCY
        ws.set_column('J:J', 10)  # FX-Rate
        ws.set_column('K:K', 16, num_format)  # SOL_II_VAL_EUR (market value)
        ws.set_column('L:L', 16)  # SHOCK_FACTOR
        ws.set_column('M:M', 15, num_format)  # SHOCKED_VAL
        ws.set_column('N:N', 15)  # DIVERS_BENFT
        ws.set_column('O:O', 10, num_format)  # SCR
        ws.set_column('P:R', 17)  # ALIAS_NAME, COUNTRY_CODE, SECTOR
        ws.set_column('S:S', 15)  # BOND_STATUS
        ws.set_column('T:T', 10, num_format)  # 
        ws.set_column('U:V', 16, num_format)  # 
        ws.set_column('W:Y', 12, num_format)  # 

    
    def _raw_data_to_excel(self, excel_writer):
        
        # Export raw BOND data
        for i in self.datasets[-3:]:
            sheetname = f'EQUITY_DATA {i.report_period}'
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
            ws.set_column('A:G', 17)
            ws.set_column('H:H', 7)
            ws.set_column('I:J', 17)
            ws.set_column('K:K', 12, num_format)
            ws.set_column('L:L', 10)
            ws.set_column('M:M', 11, curr_format)
            ws.set_column('N:N', 17, num_format)
            ws.set_column('O:O', 16, curr_format)
            ws.set_column('P:P', 15, num_format)
            ws.set_column('Q:Q', 15, curr_format)
            ws.set_column('R:R', 10, num_format)
    
    def generate_reports(self, excel: bool = False, html: bool = False):
        if excel:
            self.generate_excel_report()

        if html:
            self.generate_html_report()
