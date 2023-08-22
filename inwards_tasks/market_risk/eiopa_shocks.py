from os import path

import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import QuarterEnd
import numpy as np
import plotly.express as px
import plotly.offline

try:
    from .utils import CORPORATE_COLOR, PLOT_TITLE_FONT_FORMAT
    from .utils import add_attributes_to_html_table
except ImportError:
    import sys
    sys.path.append(path.dirname(path.abspath(__file__)))
    from utils import CORPORATE_COLOR, PLOT_TITLE_FONT_FORMAT
    from utils import add_attributes_to_html_table

SHOCK_FILE_PATH = path.join(path.dirname(path.abspath(__file__)),
                            'aux_files',
                            'eiopa_shocks.txt')


class EIOPAShocks:
    def __init__(self) -> None:
        """
        Class with EIOPA Shocks.

        Class can be used to compare Shocks used in the Equity SCR.

        Notes
        -----
        The main functions from this class are:
        - `get_shocks(report_date)`
        - `plot_shocks()`
        - `self.shocks
        """
        self.shocks = None

        # Import and calculate shocks
        self._load_eiopa_shocks()

        # To make sure we have the latest Quarter date.
        self.check_latest_shock_date()

        # Calculate all shocks
        self._calculate_equity(cat_type=1)
        self._calculate_equity(cat_type=2)
        self._calculate_other_equity()
        self._calculate_strategic_shock()
        self._calculate_transitional(cat_type=1)
        self._calculate_transitional(cat_type=2)

    def _load_eiopa_shocks(self) -> None:
        """
        Import Shocks directly from EIOPA.
        """
        try:
            # Here I decided that I should save the EIOPA shocks into the
            # aux_files folder, so that it speeds up the code, and only
            # download from the website if necessary
            df = pd.read_csv(SHOCK_FILE_PATH, index_col=0, sep='\t')

            # Convert index to datetime to avoid calculation errors
            df.index = pd.to_datetime(df.index)

            self.shocks = df
            print(f'Shocks loaded from:\n{SHOCK_FILE_PATH}')
        except FileNotFoundError:
            self.import_and_process_eiopa_shocks_from_web()

    def import_and_process_eiopa_shocks_from_web(self):
        today = pd.to_datetime('now', utc=True)
        # Every quarter end, we will always use the previous month's shocks
        # So, if we are checking Q4 runs, we will run this code in January
        # and EIOPA will have released the December shocks
        subtract_months = -1
        if today.day < 3:
            # EIOPA takes a few days to update the latest
            # symmetric adjustments
            # This takes care of that.
            subtract_months -= 1
        eiopa_date = today + DateOffset(months=subtract_months)
        eiopa_month = eiopa_date.month_name().lower()
        eiopa_year = eiopa_date.year

        link = 'https://www.eiopa.europa.eu/sites/default/files/symmetric_adjustment/eiopa_symmetric_adjustment_equity_capital_charge_{}_{}.xlsx'.format(
            eiopa_month,
            eiopa_year
        )

        print('Loading EIOPA Shocks from\n{}'.format(link))
        print('...')

        # EIOPA Shocks are always within an Excel file
        # that follows the same pattern
        df = pd.read_excel(link,
                           sheet_name='Calculations',
                           usecols='B, G',
                           header=8,
                           index_col=0,
                           engine='openpyxl')

        df.index.name = None
        # Eliminate empty rows
        df = df.dropna(subset=['Dampener final'])

        ## Filters below are just to reduce the amount of rows
        # Shocks at quarter end
        only_qrt_end = df.index.is_quarter_end
        # This date is specified in the SCR Risk Manual
        only_after_date = df.index >= pd.to_datetime('2016/03/31')

        print('EIOPA Shocks loaded.')

        df.loc[(only_qrt_end) & (only_after_date)
               ].to_csv(SHOCK_FILE_PATH, sep='\t')

        self.shocks = df.loc[(only_qrt_end) & (only_after_date)]

    def check_latest_shock_date(self):
        # to avoid passing seconds and minutes
        today = pd.Timestamp.today().normalize()
        subtract_quarter_end = -1
        if today.day < 3:
            subtract_quarter_end -= 1
        latest_quarter_end = today + QuarterEnd(subtract_quarter_end)

        try:
            self.shocks.loc[latest_quarter_end]
        except KeyError:
            self.import_and_process_eiopa_shocks_from_web()

    def _calculate_equity(self, cat_type: int) -> None:
        """
        Calculate Category Equity Shocks.
        
        Parameters
        ----------
        cat_type : int
            Shock Category Type can be 1 (`0.39`) or 2 (`0.49`)
            Shocks are added to Symmetric Adjustment from EIOPA.
        """
        if cat_type == 1:
            eq_shock = 0.39
        elif cat_type == 2:
            eq_shock = 0.49

        equity_col = 'EQUITY{}'.format(cat_type)

        self.shocks[equity_col] = self.shocks['Dampener final'] + eq_shock

        return None

    def _calculate_strategic_shock(self) -> None:
        """
        Calculate Category Strategic 1 & 2 Shocks.
        
        Notes
        -----
        Strategic 1 & 2 Shock is 22%.
        """
        self.shocks['STRATEGIC1'] = 0.22
        self.shocks['STRATEGIC2'] = 0.22
        return None

    def _calculate_other_equity(self) -> None:
        """
        Calculate Category Other Equity Shock.
        
        Notes
        -----
        Other Equity Shock is calculated by adding 49% to EIOPA Shocks.
        """
        eq2_sh = 0.49
        self.shocks['OTHER EQUITY2'] = self.shocks['Dampener final'] + eq2_sh
        return None

    def _calculate_transitional(self, cat_type: int) -> None:
        """
        Calculate Category Transitional 1 Shock.
        
        Notes
        -----
        Transitional 1 Shock is calculated with the following formula:

        `(1 - w) * 0.22 + w * (X + SA)`

        Where:
        ```
        w = number of reporting quarters after 1/1/2016, including current
            ----------------------------------------------------------------
            Total number of reporting quarters after 1/1/2016, incl. current
        
        SA = Symmetric Adjustment (EIOPA)
        X = Shock Category 1 (0.39) or 2 (0.49).
        ```

        Paramaters
        ----------
        cat_type : int
            1 (`X = 0.39`) or 2 (`X = 0.49`)
            X is the variable in Notes
        """
        equity_col = 'EQUITY{}'.format(cat_type)
        trans_helper_col = 'TRANS{}-qrt_elapsed'.format(cat_type)
        transitional_col = 'TRANSITIONAL{}'.format(cat_type)

        # Calculate after 31/12/2015 to include current quarter
        self.shocks[trans_helper_col] = (self.shocks.index
                                         - pd.to_datetime('2015/12/31'))

        # Converts Date Difference into months
        self.shocks[trans_helper_col] = (
            self.shocks[trans_helper_col] / np.timedelta64(1, 'M')
        )

        # Round and divide by 3 to calculate how many quarters elapsed
        self.shocks[trans_helper_col] = (
            self.shocks[trans_helper_col].round(decimals=1) / (3)
        )

        self.shocks[transitional_col] = (
            0.22 * (1 - self.shocks[trans_helper_col] / 28)
            + self.shocks[equity_col] * self.shocks[trans_helper_col] / 28
        )

        self.shocks.drop(columns=[trans_helper_col], inplace=True)

        return None

    def get_shock_factors(self, date: str, shock_type: str = None) -> pd.Series:
        """Return Shocks correspoding to a certain report period.

        Parameters
        ----------
        report_period : str
            Reporting period being analysed in format `dd/mm/yyyy`
        """

        # Create Timestamp of report_period string
        date_ts = pd.to_datetime(date, dayfirst=True)

        # Dampener final is the SA. There's no need to include that
        sh = self.shocks.loc[date_ts].drop('Dampener final')

        # rename to facilitate merger with main data
        sh.rename('SHOCK_FACTOR', inplace=True)

        if shock_type is not None:
            return sh[shock_type]
        else:
            return sh

    def plot_shocks(self, to_html: bool = False) -> None:
        """Generate plot with shocks across time.
        """
        if self.shocks is None:
            # to avoid error if shocks are not loaded from EIOPA website
            return px.line()

        fig = px.line(
            self.shocks[['EQUITY1', 'TRANSITIONAL1']],
            title='Shock Variation',
            labels={'value': 'Shock Factor',
                    'index': '',
                    'variable': 'Equity Category'},
            color_discrete_sequence=[CORPORATE_COLOR[0], CORPORATE_COLOR[1]]
        )

        fig.update_traces(mode="markers+lines", hovertemplate=None)
        fig.update_layout(title_font=PLOT_TITLE_FONT_FORMAT,
                          template='plotly_white',
                          hovermode='x unified',
                          yaxis_tickformat='%')

        if to_html:
            return plotly.offline.plot(fig,
                                       include_plotlyjs=False,
                                       output_type='div')
        else:
            return fig

    def get_shocks_by_date(self, to_html: bool = False):
        df = self.shocks.copy()
        df.index = df.index.strftime('%Y-%m-%d')

        df.rename(columns={'Dampener final': 'Symmetric Adjustment'},
                  inplace=True)

        df_formatted = df.style.format("{:.1%}")

        if to_html:

            df_to_html = df_formatted.to_html()

            return add_attributes_to_html_table(html=df_to_html,
                                                data_search=True,
                                                data_pagination=True,
                                                # Using df.columns becase all
                                                # columns are numbers, except
                                                # the index
                                                data_sorter_cols=df.columns)
        else:
            return df_formatted
