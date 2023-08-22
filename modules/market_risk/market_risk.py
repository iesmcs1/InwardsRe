from typing import Union
from abc import abstractmethod
import pandas as pd
import os
from markdown2 import Markdown
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline

try:
    from ..utils import get_report_period, get_date_from_period
    from ..utils import CORPORATE_VALUE_COLOR, CORPORATE_COLOR, PLOT_TITLE_FONT_FORMAT
except ImportError:
    import sys
    if os.getcwd().endswith('InwardsRe'):
        sys.path.insert(0, os.path.abspath('./modules'))
    from utils import get_report_period, get_date_from_period
    from utils import CORPORATE_VALUE_COLOR, CORPORATE_COLOR, PLOT_TITLE_FONT_FORMAT

class MarketRisk:
    def __init__(self) -> None:
        self._market_risk_data = list()

        self._current_report_date = None

        self._report_dates_added = list()

        self._periods = {'new': None, 'old': None}

        self._comparison_data = {'new': None, 'old': None}

        

        # 3 attributes below will be updated for each 
        # Market Risk type, if needed.
        self.market_risk_type = None
        self.market_value_col = None
        self.market_value_eur_col = None
    
    @abstractmethod
    def calculate_quarter_movements(self):
        """Abstract Method for calculating quarter movements.

        Raises
        ------
        NotImplementedError
            Child class has to overwrite this method, otherwise raise error.
        """        
        raise NotImplementedError
    
    @property
    def market_risk_data(self):
        return self._market_risk_data
    
    @market_risk_data.setter
    def market_risk_data(self, data):
        # Put data timestamp to variable
        ts = data.report_date

        if data in self.market_risk_data:
            # If Period already has data, remove data
            self.market_risk_data.remove(data)
            print(f"Data for {ts} was overwritten by new data.")
        
        # Add data to list
        self.market_risk_data.append(data)

        # Sort list from oldest to newest
        self.market_risk_data.sort()

        # Add date to report_dates_added
        self.report_dates_added = ts

        # Add date to current_report_date. It will only keep the latest date
        self.current_report_date = ts

        # Update periods
        self.periods = ts
    
    @property
    def current_report_date(self):
        return self._current_report_date
    
    @current_report_date.setter
    def current_report_date(self, date: pd.Timestamp):
        if self._current_report_date is None:
            self._current_report_date = date
        elif self._current_report_date < date:
            # If loaded date is newer than previous loaded date
            self._current_report_date = date
        else:
            pass
    
    @property
    def report_dates_added(self):
        return self._report_dates_added
    
    @report_dates_added.setter
    def report_dates_added(self, date: Union[str, pd.Timestamp]):
        ts = self.convert_to_timestamp(date)

        if ts not in self.report_dates_added:
            # Append date added to model if data is not already in
            # This avoids creating duplicated dates when adding
            # twice same data
            self._report_dates_added.append(ts)
        
        # Sort from oldest to newest
        self._report_dates_added.sort()

    
    @property
    def periods(self):
        return self._periods
    
    @periods.setter
    def periods(self, date: Union[str, pd.Timestamp]):
        # make sure it's pd.Timestamp
        ts = self.convert_to_timestamp(date)
        if ts in self.report_dates_added:
            if self._periods['new'] is None:
                # If first time adding data to model
                self._periods['new'] = get_report_period(ts)
            elif get_date_from_period(self._periods['new']) > ts:
                # if periods['new'] is newer than date, then ts is old
                self._periods['old'] = get_report_period(ts)
            elif get_date_from_period(self._periods['new']) < ts:
                # if new period date is older than date, then it's new period
                self._periods['old'] = self._periods['new']
                self._periods['new'] = get_report_period(ts)
            
            # Lastly, use date index to find exact position of corresponding
            # data. Then add the data to comparison data
            # The self.periods will have been updated, so we just need 
            # to add comparison data

            for data in self.market_risk_data:
                # to make sure data is added to comparison data
                self.comparison_data = data
        else:
            # date_index will produce ValueError if date is not in 
            # report_dates_added
            print(f"Data for date {ts} is not in model.")
            return None
    
    @property
    def comparison_data(self):
        return self._comparison_data
    
    @comparison_data.setter
    def comparison_data(self, data):
        if data.report_period == self.periods['new']:
            self._comparison_data['new'] = data
        elif data.report_period == self.periods['old']:
            self._comparison_data['old'] = data
    
    @staticmethod
    def convert_to_timestamp(date: Union[str, pd.Timestamp]):
        if isinstance(date, pd.Timestamp):
            return date
        else:
            return pd.to_datetime(date, dayfirst=True)
        
    def update_periods_for_comparison(
        self,
        old_date: Union[str, None] = None,
        new_date: Union[str, None] = None) -> None:
        """Update either old or new date for comparison.

        Parameters
        ----------
        old_date : Union[str, None], optional
            oldest date to be used in comparison. New date will
            remain the same, by default None
        new_date : Union[str, None], optional
            Newest date to be used in comparison. Old date will
            remain the same, by default None
        """        
        if len(self.market_risk_data) < 2:
            print("None enough data points for comparison.")
            return None

        if old_date is not None:
            self.periods = self.convert_to_timestamp(old_date)

        if new_date is not None:
            self.periods = self.convert_to_timestamp(old_date)

        print(
            f'Model will compare {self.periods["old"]} to {self.periods["new"]}.')

        return None

    def get_scr_summary_table(self) -> pd.DataFrame:
        """Return SCR Summary table, grouped by date.

        Returns
        -------
        DataFrame
            DataFrame containing SCR for each quarter, with absolute and
            relative change.
        """
        # For SCR Summary, we only need 2 columns
        cols = ['REPORTING_DAT', 'SCR']

        # create empty DF to concatenate later
        df_raw = pd.DataFrame()

        # Iterate over data
        for i in self.market_risk_data:
            if self.market_risk_type == 'currency':
                # For currency Risk, we have multiple tables to calculate
                # the SCR. Therefore, it is stored differently.
                df_temp = i.data['SCR'].copy()
                df_temp['REPORTING_DAT'] = i.report_date
                df_raw = pd.concat([df_raw, df_temp[cols]])
            else:
                df_raw = pd.concat([df_raw, i.data[cols]])
        
        # Group by REPORTING_DAT, as to calculate movement between qtr
        df = df_raw.groupby(by='REPORTING_DAT').sum()

        df['Diff'] = df['SCR'].diff()
        df['Diff %'] = df['SCR'].pct_change()

        df['MATCH_GROUP_CALC'] = self._match_fc_computed()

        return df
    
    def _match_fc_computed(self):
        for i in self.market_risk_data:
            if i.match_fc_computed_amount() is False:
                return False
        
        return True
    
    def _get_overall_movement_text(self, to_html) -> str:
        
        df = self.get_scr_summary_table()
        
        # Overall Calculations
        prev_scr = df.loc[df.index[-2], 'SCR']
        curr_scr = df.loc[df.index[-1], 'SCR']
        scr_var = (curr_scr - prev_scr) / prev_scr

        text = '#### Total Movement:\n'
        text += '- {:+.1%} (or €{:+,.1f}m) from last quarter\n'.format(
            scr_var,
            (curr_scr - prev_scr) / 1e6
        )
        text += '- From €{:,.1f}m to €{:,.1f}m\n\n'.format(prev_scr / 1e6,
                                                           curr_scr / 1e6)

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def _get_purchased_sold_movement_text(self, to_html):
        
        df = self.calculate_quarter_movements()
        df = df.groupby('BOND_STATUS').sum()

        mv_sold = df.loc['SOLD', f'{self.market_value_col}-'+self.periods['old']]
        mv_purc = df.loc['PURCHASED', f'{self.market_value_col}-'+self.periods['new']]

        text = '#### Purchased / Sold:\n'

        text += '- Total Market Value Purchased: €{:,.2f}m\n'.format(
            mv_purc / 1e6)
        
        text += '- Total Market Value Sold: €{:,.2f}m\n'.format(
            mv_sold / 1e6)
        
        text += '- Net Purchased / Sold in Quarter: €{:+,.1f}m\n'.format(
            (mv_purc - mv_sold) / 1e6)
        
        text += self._get_weighted_average_text(
            scr_section='Purchased/Sold')

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
        
    def _get_organic_movement_text(self, to_html:bool) -> str:
        df = self.calculate_quarter_movements()
        df = df.groupby('BOND_STATUS').sum()

        # -2 is previous reporting date
        mv_old = df.loc['UNCHANGED', f'{self.market_value_col}-'+self.periods['old']]
        mv_new = df.loc['UNCHANGED', f'{self.market_value_col}-'+self.periods['new']]

        text = '#### Organic Growth(I):\n'
        text += '- Market Value from €{:,.1f}m to €{:,.1f}m\n'.format(
            mv_old / 1e6,
            mv_new / 1e6,
        )

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def _helper_weighted_average(self,
                                 by: str,
                                 bond_status: str = None):
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
            mv_col = f'{self.market_value_col}-'+self.periods['old']
        elif self.periods['new'] in by:
            mv_col = f'{self.market_value_col}-'+self.periods['new']

        if 'ECAI' in by:
            df[by] = df[by].astype(int)

        df['HELPER_WA'] = df[by] * df[mv_col]
        df['HELPER_WA'] = df['HELPER_WA'].fillna(0)
        s = df.sum()
        
        return s['HELPER_WA'] / s[mv_col]
    
    def _get_weighted_average_text(self, scr_section: str = None) -> str:
        """[summary]

        Parameters
        ----------
        scr_section : str, optional
            if it is necessary to calculate the weighted average of a 
            certain section of the SCR. Values can be:
            - 'purchased-sold', to calculate only for that particular section
            
            by default None
        """
        text = ''

        if self.market_risk_type == 'spread':
            cols_for_wa_text = ['MODIFIED_DURATION-', 'ECAI-', 'SHOCK_FACTOR-']
        elif self.market_risk_type == 'equity':
            cols_for_wa_text = ['SHOCK_FACTOR-', 'ECAI-']

        for col in cols_for_wa_text:
            if scr_section == 'Purchased/Sold':
                sold_weighted_av = self._helper_weighted_average(
                    by=col+self.periods['old'], bond_status='SOLD')

                purc_weighted_av = self._helper_weighted_average(
                    by=col+self.periods['new'], bond_status='PURCHASED')
                
                if 'SHOCK_FACTOR' in col:
                    text += '  - {}: from {:.1%} ({}) to {:.1%} ({})\n'.format(
                        col.replace('-', ''),
                        sold_weighted_av,
                        'Sold',
                        purc_weighted_av,
                        'Purchased'
                    )
                else:
                    text += '  - {}: from {:.2f} (Sold) to {:.2f} (Purchased)\n\n'.format(
                        col.replace('-', ''),
                        sold_weighted_av,  # previous quarter
                        purc_weighted_av,  # current quarter
                    )
            elif scr_section == 'SHOCK_FACTOR':
                pass
            else:
                err_msg = 'Weighted Average can only be calculated for'
                err_msg += 'Purchased/Sold section of the SCR.'
                raise NotImplementedError(err_msg)
        
        text = text.replace('MODIFIED_DURATION', 'Modified Duration')

        return text
    
    def _get_report_name(self, file_extension: str):
        if self._match_fc_computed():
            report_type = '(Final)'
        else:
            report_type = '(Preliminary)'

        file_name = '{} Risk-{} {}.{}'.format(
            self.market_risk_type.capitalize(),
            self.periods['new'],
            report_type,
            file_extension
        )

        return os.path.join(file_name)
    
    def plot_waterfall_movement_summary(self, for_html_report: bool = False):
        """Plot a Waterfall, showing movements from one quarter to the next.

        Parameters
        ----------
        quarters_to_plot : int, optional
            How many quarter to appear in the Waterfall Plot, by default 2
        """        
        df = self.calculate_quarter_movements()

        cols_to_use = [f'SCR-{self.periods["old"]}',
                       'Diff-Purchased/Sold',
                       'Diff-Org Growth',
                       'Diff-Shock',
                       'Diff-FX',
                       f'SCR-{self.periods["new"]}',]

        df_sum = df[cols_to_use].sum()

        text_values = []
        for i in df_sum:
            # this is to show values in millions
            text_values.append("{:,.2f}m".format(i / 1e6))
        
        measure = ['absolute',
                   'relative',
                   'relative',
                   'relative',
                   'relative',
                   'total']

        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=measure,
            x=cols_to_use,
            textposition="outside",
            textfont={'size': 16},
            text=text_values,
            y=df_sum,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": CORPORATE_VALUE_COLOR['negative']}},
            decreasing={"marker": {"color": CORPORATE_VALUE_COLOR['positive']}},
            totals={"marker": {"color": CORPORATE_VALUE_COLOR['neutral']}},
            showlegend=False,
        ))

        wf.update_layout(title_text="{} SCR (QoQ)".format(self.market_risk_type.capitalize()),
                         title_font=PLOT_TITLE_FONT_FORMAT,
                         waterfallgap=0.5,
                         template='plotly_white')

        # 0.95 and 1.05 were chosen just for formatting purposes.
        # This ensures that the numbers will be visible with any values
        scr_range = [df_sum[f'SCR-{self.periods["old"]}'],
                     df_sum[f'SCR-{self.periods["new"]}']]
        
        y_axis_range = [min(scr_range) * 0.9,
                        max(scr_range) * 1.15]

        # Remove Y axis and set range to y_axis_range
        wf.update_yaxes(visible=False, range=y_axis_range)

        if for_html_report:
            return plotly.offline.plot(wf,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return wf

    def plot_market_value_and_scr(self,
                                  add_mv_scr_ratio: bool,
                                  for_html_report: bool = False):

        df = pd.DataFrame()
        for i in self.market_risk_data:
            df = pd.concat([
                pd.DataFrame([[i.get_market_value_sum(), i.get_scr_sum()]],
                             index=[i.report_date],
                             columns=[f'{self.market_value_eur_col}', 'SCR']),
                df
            ])
        df.sort_index(inplace=True)

        # Convert everything into millions
        # Doing this way because the D3 formatting ('.2s') was showing G
        # for values in billions, because it's Giga. This is a workaround.
        df = df[[f'{self.market_value_eur_col}', 'SCR']] / 1e6
        df['MV : SCR'] = df['SCR'] / df[f'{self.market_value_eur_col}']

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index,
                       y=df[f'{self.market_value_eur_col}'],
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
                         tickformat=',.0f',
                         ticksuffix='M')

        if add_mv_scr_ratio:
            fig.add_trace(
                go.Scatter(x=df.index,
                           y=df['MV : SCR'],
                           name="MV : SCR",
                           line={'color': CORPORATE_COLOR[-2]}),
                secondary_y=True)

            fig.update_yaxes(title_text="Market Value : SCR",
                             tickformat='.2%',
                             showgrid=False,
                             secondary_y=True)

        if for_html_report:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')

        return fig
    
    def plot_market_value_by_ecai(self, for_html_report: bool = False):
        """Bar plot with Market Value by ECAI

        Parameters
        ----------
        for_html_report : bool, optional
            If True return html element to be inserted in report,
            by default False.

        Returns
        -------
        Ploty plot or HTML element.
        """        
        # Make copy so we don't alter original DataFrame
        df = self.calculate_quarter_movements()

        ecai_col_old = f"ECAI-{self.periods['old']}"
        ecai_col_new = f"ECAI-{self.periods['new']}"

        mv_col_old = f"{self.market_value_eur_col}-{self.periods['old']}"
        mv_col_new = f"{self.market_value_eur_col}-{self.periods['new']}"

        # ECAI 9 should be named as Unrated, as per request
        df.loc[df[ecai_col_old] == '9', ecai_col_old] = 'Unrated'
        df.loc[df[ecai_col_new] == '9', ecai_col_new] = 'Unrated'

        # Grouping data as to avoid showing each equity in plot
        df = df.groupby(by=[ecai_col_old, ecai_col_new], as_index=False).sum()

        # Rename columns for Plot
        df.rename(columns={mv_col_old: f"Market Value-{self.periods['old']}",
                           mv_col_new: f"Market Value-{self.periods['new']}"},
                  inplace=True)

        # Aggregate all ECAI values and leave only uniques for X axis
        x_vals = pd.concat([df[ecai_col_new], df[ecai_col_old]]).unique()

        y_vals = [f"Market Value-{self.periods['old']}",
                  f"Market Value-{self.periods['new']}"]

        fig = px.bar(
            df,
            x=x_vals,
            y=y_vals,
            barmode='group',
            template='plotly_white',
            #labels={'value': 'Market Value (EUR)'},
            color_discrete_sequence=[CORPORATE_COLOR[0], CORPORATE_COLOR[1]]
        )

        fig.update_yaxes(title_text='Market Value (EUR)')

        fig.update_xaxes(type='category',
                         categoryorder='category ascending',
                         title_text='ECAI Rating')

        fig.update_layout(title_text='Market Value (EUR) by ECAI',
                          title_font=PLOT_TITLE_FONT_FORMAT)

        if for_html_report:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')

        return fig
