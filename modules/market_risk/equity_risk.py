from .market_risk_data import MarketRiskData
from .market_risk import MarketRisk
import os
import pandas as pd
import numpy as np
from ..utils import get_report_period, get_inwards_folder, add_attributes_to_html_table
from ..utils import PLOT_TITLE_FONT_FORMAT, CORPORATE_COLOR
from .eiopa_shocks import EIOPAShocks
import plotly.express as px
from markdown2 import Markdown


class EquityData(MarketRiskData):
    def __init__(self, folder_path: str, report_date: str) -> None:

        MarketRiskData.__init__(self,
                                folder_path=folder_path,
                                report_date=report_date)

        self.shocks = EIOPAShocks()

        self.DATA_NAME = 'EQUITY'

        self.import_data()
        self.process_data()

    def get_shocks(self, shock_type: str = None):
        return self.shocks.get_shock_factors(
            date=self.report_date, shock_type=shock_type)

    def import_data(self):
        filepath = os.path.join(self.folder_path, 'TBSL_EQUITIES.xlsx')

        df = pd.read_excel(filepath,
                           dtype={'EQUITY_TYP': str,
                                  'PURCHASE_DAT': str,
                                  'ISSUER_IDENTIFIER': str,
                                  'ECAI': str},
                           engine='openpyxl')

        # In case Excel file has extra lines, and they are imported with NaN
        df.dropna(axis='index', how='all', inplace=True)

        self.data = df

        return None

    def process_data(self):

        self._timestamp_data()
        self._normalize_purchase_dat_column()
        self._get_equity_category()
        self._remove_cols()
        self._insert_shocks()
        self._calculate_raw_scr()
        self._calculate_real_scr()

        return None

    def _timestamp_data(self):
        df = self.data
        df['REPORTING_DAT'] = self.report_date
        return None

    def _normalize_purchase_dat_column(self):
        df = self.data
        df['PURCHASE_DAT'] = pd.to_datetime(df['PURCHASE_DAT'], dayfirst=True)
        return None

    def _get_equity_category(self):
        df = self.data
        if 'EQUITY_CATEGORY' in df.columns and 'EQUITY_TYP' in df.columns:
            # Add Equity Category to DataFrame.

            purchase_after_2016 = (df['PURCHASE_DAT'] >
                                   pd.to_datetime('1/1/2016', dayfirst=True))

            df['equityCategory'] = np.where(
                purchase_after_2016,
                # purchased AFTER 2016
                df['EQUITY_CATEGORY'] + df['EQUITY_TYP'],
                # purchased BEFORE 2016
                'TRANSITIONAL' + df['EQUITY_TYP']
            )
        return None

    def _remove_cols(self):
        df = self.data

        cols_to_remove = ['SUBPORTFOLIO', 'U_ECAI']
        df.drop(columns=cols_to_remove, inplace=True)

        return None

    def _insert_shocks(self):
        df = self.data

        df = df.merge(self.get_shocks(),
                      left_on='EQUITYCATEGORY',
                      right_index=True,
                      how='left')

        # Assigning again because merge creates a new copy
        self.data = df

        return None

    def _calculate_raw_scr(self):
        df = self.data

        # Calculate raw SCR by applying Shock to Market Value in EUR
        df['SHOCKED_VAL'] = df['SHOCK_FACTOR'] * df['SOL_II_VAL_EUR']

    def _calculate_diversification_benefit_var(self):
        """Calculate diversification benefit variable.
        
        This variable will be applied to SCR_RAW.

        Returns
        -------
        float
            Variable to be used in calculating real SCR (var * raw SCR).
        """
        df_raw = self.data
        df = df_raw.pivot_table(values='SHOCKED_VAL',
                                columns='EQUITY_TYP',
                                aggfunc=np.sum).rename_axis(None)

        # ** 0.5 -> Square Root of the Sums
        df['SCR_RAW'] = df['1'] + df['2']
        df['SCR'] = (
            df['1'] ** 2 + df['2'] ** 2 + 1.5 * df['1'] * df['2']
        ) ** 0.5

        # Comparing SCR without diversification benefits (SCR_RAW)
        # and SCR with diversification benefits
        result = df['SCR'] / df['SCR_RAW']

        return result[0]

    def _calculate_real_scr(self):
        df = self.data

        df['DIVERS_BENFT'] = self._calculate_diversification_benefit_var()

        df['SCR'] = df['SHOCKED_VAL'] * df['DIVERS_BENFT']

        return None

    def get_market_value_sum(self):
        return self.data['SOL_II_VAL_EUR'].sum()


class EquityRisk(MarketRisk):
    QTR_COMPARISON_USE_COLS = ['EQUITY_ID',
                               'PURCHASE_DAT',
                               'ISSUER_ID',
                               'SOL_II_VAL',
                               'SOL_II_VAL_EUR',
                               'FX_RATE',
                               'ECAI',
                               'SHOCK_FACTOR',
                               'SCR']

    QTR_COMPARISON_MERGE_ON_COLS = ['EQUITY_ID',
                                    'PURCHASE_DAT',
                                    'ISSUER_ID']

    def __init__(self) -> None:
        """This is an Equity Risk class

        Notes
        -----
        the EIOPA Shocks will be used to check Equity Shock.xlsx.
        If there is no need to check this, then set to False.

        Parameters
        ----------
        import_eiopa_shocks : bool, optional
            If True, then download EIOPA Shocks.
            If False, do not download.

        """

        MarketRisk.__init__(self)

        self.market_risk_type = 'equity'
        self.market_value_col = 'SOL_II_VAL'
        self.market_value_eur_col = 'SOL_II_VAL_EUR'

        self.issuer_data = pd.DataFrame()

    def import_and_process_sourcedata_files(self,
                                            report_date: str,
                                            sourcedata_folder=None):

        self.check_folder_path_date(sourcedata_folder, report_date)

        # import and process Equity Data
        df = EquityData(folder_path=sourcedata_folder, report_date=report_date)

        # Append Equity Data Class to Equity Risk Class
        self.market_risk_data.append(df)

        self.market_risk_data.sort()

        print("{} data was added.".format(get_report_period(report_date)))

        # Update dates
        self._update_current_report_date_and_periods()

        self._update_issuer_data()

        return None

    def _update_issuer_data(self) -> None:
        "Update issuer data attribute."

        issuer_data_cols = ['ISSUER_ID',
                            'ALIAS_NAME',
                            'COUNTRY_CODE',
                            'SECTOR']

        df = self.issuer_data

        for i in self.market_risk_data:
            df = pd.concat([df, i.data[issuer_data_cols]])

        df.drop_duplicates(subset='ISSUER_ID', inplace=True, ignore_index=True)

        self.issuer_data = df

        return None

    def calculate_quarter_movements(self, issuer_data: bool = False):
        """Calculate SCR movement between the 2 latest quarters.

        Parameters
        ----------
        issuer_data : bool, optional
            If True, adds issuer data to DataFrame, by default False

        Returns
        -------
        pandas.DataFrame
            [description]
        """
        # List with all data added to model
        dates_in_data = [i.report_date for i in self.market_risk_data]

        # Find latest 2 dates
        dates_for_comparison = pd.Series(dates_in_data).nlargest(2)

        # Set these dates to old and new, to compare Quarter over Quarter (QoQ)
        old_date = dates_for_comparison.min()
        new_date = dates_for_comparison.max()

        data_old = None
        data_new = None

        for equity_data in self.market_risk_data:
            if equity_data.report_date == old_date:
                data_old = equity_data
            elif equity_data.report_date == new_date:
                data_new = equity_data

        # Merge data as to have everything side by side to see what
        # changed over the Quarter
        df = pd.merge(left=data_old.data[self.QTR_COMPARISON_USE_COLS],
                      right=data_new.data[self.QTR_COMPARISON_USE_COLS],
                      how='outer',
                      on=self.QTR_COMPARISON_MERGE_ON_COLS,
                      suffixes=('-'+data_old.report_period,
                                '-'+data_new.report_period))

        df_bond_status = self._get_bond_status(df)
        df_scr_movements = self._calculate_scr_movement(df_bond_status)
        df_scr_purch_sold = self._calculate_scr_purchased_sold_movements(
            df_scr_movements)
        df_scr_org_growth = self._calculate_scr_org_growth_movements(
            df_scr_purch_sold)
        df_scr_shock = self._calculate_scr_shock_movements(df_scr_org_growth)
        df_scr_fx = self._calculate_scr_fx_movements(df_scr_shock)

        if issuer_data:
            return df_scr_fx.merge(self.issuer_data,
                                   how='left',
                                   on='ISSUER_ID')

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
            ((df['SOL_II_VAL-'+self.periods['new']]
             - df['SOL_II_VAL-'+self.periods['old']])
             * df['SHOCK_FACTOR-'+self.periods['old']]
             / df['FX_RATE-'+self.periods['old']]),
            0
        )

        return df

    def _calculate_scr_shock_movements(self, dataframe):
        df = dataframe

        filter_unchanged = df['BOND_STATUS'] == 'UNCHANGED'

        df['Diff-Shock'] = np.where(
            filter_unchanged,
            # ( Shock(new) - Shock(old) ) * MV(new) / FX_rate(old)
            ((df['SHOCK_FACTOR-'+self.periods['new']]
             - df['SHOCK_FACTOR-'+self.periods['old']])
             * df['SOL_II_VAL-'+self.periods['new']]
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
             * df['SOL_II_VAL-'+self.periods['new']]
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
    
    def get_holding_by_issuer(self, to_html: bool = False):
        df_raw = self.calculate_quarter_movements(issuer_data=True)

        mv_col_old = f"{self.market_value_eur_col}-{self.periods['old']}"
        mv_col_new = f"{self.market_value_eur_col}-{self.periods['new']}"

        issuer_data_cols = [col for col in self.issuer_data.columns]
        # Holding by issuer will contain all issuer data columns

        df_grp = df_raw.groupby(by=issuer_data_cols, as_index=False).sum()

        holding_cols = issuer_data_cols.copy()
        for i in ['SOL_II_VAL_EUR-', 'SCR-']:
            for period in ['new', 'old']:
                holding_cols.append(i+self.periods[period])

        df = df_grp[holding_cols].copy()

        df['Diff-MV'] = df[mv_col_new] - df[mv_col_old]

        df['Diff-SCR'] = (
            df[f'SCR-{self.periods["new"]}'] - df[f'SCR-{self.periods["old"]}']
        )

        df.rename(columns={mv_col_new: f'MV-{self.periods["new"]}',
                           mv_col_old: f'MV-{self.periods["old"]}'},
                  inplace=True)

        if to_html:
            format_dict = {}
            for col in df.columns:
                if 'SCR' in col or 'MV' in col:
                    format_dict[col] = "{:,.0f}"

            df_formatted = df.style.format(format_dict).to_html()

            # Columns that contain numbers, for sorting purposes
            num_cols = [
                # If Diff, MV, or SCR are in column headers,
                # it's a number's column
                col for col in df.columns if 'Diff' in col or 'MV' in col or 'SCR' in col
            ]

            return add_attributes_to_html_table(html=df_formatted,
                                                data_search=True,
                                                data_pagination=True,
                                                data_sorter_cols=num_cols)
        else:
            return df

    def get_movement_summary_text(self, to_html: bool = False) -> str:

        text = self._get_overall_movement_text(to_html=to_html)

        text += self._get_purchased_sold_movement_text(to_html=to_html)

        text += self._get_organic_movement_text(to_html=to_html)

        text += self._get_shock_movement_text(to_html=to_html)

        return text

    def _get_shock_movement_text(self, to_html: bool) -> str:
        text = '#### Shock:\n'

        old_shock = None
        new_shock = None

        for i in self.market_risk_data:
            if i.report_period == self.periods['new']:
                new_shock = i.get_shocks(shock_type='EQUITY1')
            elif i.report_period == self.periods['old']:
                old_shock = i.get_shocks(shock_type='EQUITY1')

        text += f'- Standard Shock from {old_shock:.1%} to {new_shock:.1%}'

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text

    @staticmethod
    def _report_obs(to_html: bool) -> str:
        text = '\n(I) Only Bonds that were not purchased or sold.\n'

        if to_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text

    def plot_movement_by_issuer(self, for_html_report: bool = False):
        df = self.calculate_quarter_movements().sort_values(
            by='Diff', ascending=False)

        df_issuer = df.merge(self.issuer_data[['ISSUER_ID', 'ALIAS_NAME']],
                             how='left',
                             on='ISSUER_ID')

        mv_col_old = f"{self.market_value_eur_col}-{self.periods['old']}"
        mv_col_new = f"{self.market_value_eur_col}-{self.periods['new']}"

        # Market Value Difference
        # The MV Diff had to be added to hover data since it is not in
        # dataframe. I thought this would be easier to do like this.
        mv_diff = df[mv_col_new].fillna(0) - df[mv_col_old].fillna(0)

        # Rename Market Value columns
        df_issuer.rename(
            columns={mv_col_old: f"Market Value-{self.periods['old']}",
                     mv_col_new: f"Market Value-{self.periods['new']}"},
            inplace=True)

        hover_data = {}

        # fillna to avoid ugly text in tooltip -> "%{customdata[1]:.4s}"
        df_issuer.fillna(0, inplace=True)

        hover_data['Market Value Diff'] = (':.4s', mv_diff)

        fig = px.bar(
            df_issuer,
            x="ALIAS_NAME",
            y=["Diff-Purchased/Sold", "Diff-Org Growth", "Diff-Shock"],
            template='plotly_white',
            labels={'value': 'Amount (EUR)', 'ALIAS_NAME': ''},
            hover_data=hover_data,
            color_discrete_sequence=[CORPORATE_COLOR[-1],
                                     CORPORATE_COLOR[-2],
                                     CORPORATE_COLOR[-3]]
        )

        for dat in fig.data:
            dat.name = dat.name.replace("Diff-", "")

        fig.update_layout(title_text='Top Positive and Negative Movements (SCR)',
                          title_font=PLOT_TITLE_FONT_FORMAT,
                          height=700,
                          xaxis_tickangle=-45)

        if for_html_report:
            return fig.to_html(include_plotlyjs=False, full_html=False)
        return fig

    def plot_top_issuers_by_market_value(self,
                                         n: int,
                                         for_html_report: bool = False):
        """Plot top n Issuers by Market Value in dataframe or plot.

        Parameters
        ----------
        n : int
            largest n companies by Market Value will be selected.
        """
        mv_col_old = f"{self.market_value_eur_col}-{self.periods['old']}"
        mv_col_new = f"{self.market_value_eur_col}-{self.periods['new']}"

        df = self.calculate_quarter_movements(issuer_data=True)
        df_sum = df.groupby(
            by=['ISSUER_ID', 'ALIAS_NAME'], as_index=False).sum()
        df_top_n = df_sum.nlargest(n, columns=[mv_col_new])

        # Rename columns for Plot
        df_top_n.rename(
            columns={mv_col_old: f"Market Value-{self.periods['old']}",
                     mv_col_new: f"Market Value-{self.periods['new']}"},
            inplace=True
        )

        fig = px.bar(
            df_top_n,
            x="ALIAS_NAME",
            # we only need Market Value for this plot
            y=[i for i in df_top_n.columns if 'Market Value' in i],
            title=f"Top {n} Holding",
            barmode='group',
            hover_data={'value': ':.3s'},
            color_discrete_sequence=[CORPORATE_COLOR[0], CORPORATE_COLOR[1]])

        fig.update_layout(title_font=PLOT_TITLE_FONT_FORMAT,
                          template='plotly_white',
                          legend_traceorder='reversed')

        if for_html_report:
            return fig.to_html(include_plotlyjs=False, full_html=False)
        return fig

    def plot_top_market_value_movements(self,
                                        position: str,
                                        n: int,
                                        for_html_report: bool = False):
        """Plot top or bottom n Issuers by Market Value diff.

        Parameters
        ----------
        position : str
            'top' (pd.nlargest) or 'bottom' (pd.nsmallest)
        n : int
            top or bottom n companies by Market Value difference
            between quarter will be selected.
        """

        mv_diff_col = 'Difference QoQ (EUR)'

        df = self.calculate_quarter_movements(issuer_data=True)

        mv_col_old = f"{self.market_value_eur_col}-{self.periods['old']}"
        mv_col_new = f"{self.market_value_eur_col}-{self.periods['new']}"

        df[mv_diff_col] = df[mv_col_new].fillna(0) - df[mv_col_old].fillna(0)

        df_sum = df.groupby(
            by=['ISSUER_ID', 'ALIAS_NAME'], as_index=False).sum()

        if position == 'top':
            df_final = df_sum.nlargest(n=n, columns=mv_diff_col)
        elif position == 'bottom':
            # it is visually better to reverse the dataframe when showing
            # negative movements (used [::-1])
            df_final = df_sum.nsmallest(n=n, columns=mv_diff_col).iloc[::-1]

        fig = px.bar(df_final,
                     x="ALIAS_NAME",
                     # we only need Market Value for this plot
                     y=mv_diff_col,
                     title="{} {} in Market Value Holding Difference (QoQ)".format(
                         position.capitalize(),
                         n
                     ),
                     color_discrete_sequence=[CORPORATE_COLOR[-1]],
                     hover_data={mv_diff_col: ':.3s'})

        fig.update_layout(title_font=PLOT_TITLE_FONT_FORMAT,
                          template='plotly_white')

        fig.update_xaxes(title_text='')

        if for_html_report:
            return fig.to_html(include_plotlyjs=False, full_html=False)
        return fig

    def generate_html_report(self):
        from jinja2 import Environment, FileSystemLoader

        templates_dir = os.path.join(get_inwards_folder(),
                                     'modules',
                                     'market_risk',
                                     'templates')

        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('equity_risk.html')

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
        elem['scr_diff_pct_formatted'] = "{:+,.1%}".format(
            elem['scr_diff_pct'])
        elem['scr_increase'] = elem['scr_diff_pct'] > 0

        elem['scr_movement_summary_text'] = self.get_movement_summary_text(
            to_html=True)
        elem['waterfall_movement_summary'] = self.plot_waterfall_movement_summary(
            for_html_report=True)

        elem['mv_scr_plot'] = self.plot_market_value_and_scr(
            add_mv_scr_ratio=True, for_html_report=True)

        elem['mv_by_ecai_plot'] = self.plot_market_value_by_ecai(
            for_html_report=True)

        elem['movement_by_issuer_plot'] = self.plot_movement_by_issuer(
            for_html_report=True)

        elem['top_issuers_by_mv'] = self.plot_top_issuers_by_market_value(
            10, for_html_report=True)

        elem['top_mv_movements_top'] = self.plot_top_market_value_movements(
            position='top', n=10, for_html_report=True)

        elem['top_mv_movements_bottom'] = self.plot_top_market_value_movements(
            position='bottom', n=10, for_html_report=True)

        # ---------------------------------------------------------------------
        # Shock section
        elem['df_holding'] = self.get_holding_by_issuer(to_html=True)

        # ---------------------------------------------------------------------
        # Shock section
        self.market_risk_data.sort()
        data = self.market_risk_data[-1]

        elem['plot_shock_over_time'] = data.shocks.plot_shocks(
            for_html_report=True)
        elem['df_shock_over_time'] = data.shocks.get_shocks_by_date(
            for_html_report=True)

        #elem['maturity_analysis_plots'] = self._maturity_analysis_plots_for_html()

        return elem

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
        self._raw_data_to_excel(excel_writer=writer)

        # # Create Sheet with Duration to Maturity analysis
        # self._duration_to_maturity_to_excel(excel_writer=writer)

        writer.save()

        print(f'File "{filename}" created.')

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
        merge_format = wb.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'top'
        })

        ws = excel_writer.sheets[sheetname]

        # Write explanation to cell below
        ws.merge_range(
            'H2:H23', self.get_movement_summary_text(), merge_format)

        # Formatting
        # SCR Summary Worksheet
        ws.set_column('B:B', 18)
        ws.set_column('C:D', 13, num_format)
        ws.set_column('E:E', 13, pct_format)
        ws.set_column('F:F', 23, num_format)
        ws.set_column('H:H', 60, num_format)

    def _quarter_movements_to_excel(self, excel_writer):
        # Export Quarters comparison
        df = self.calculate_quarter_movements(issuer_data=True)
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
        ws.set_column('A:C', 19)

        # Old Quarter
        ws.set_column('D:E', 23, num_format)
        ws.set_column('F:F', 16, curr_format)
        ws.set_column('G:H', 13)
        ws.set_column('I:I', 12, num_format)

        # New Quarter
        ws.set_column('J:K', 23, num_format)
        ws.set_column('L:L', 16, curr_format)
        ws.set_column('M:N', 13)
        ws.set_column('O:O', 12, num_format)

        #BOND STATUS Column
        ws.set_column('P:P', 16)

        # Deltas
        ws.set_column('Q:U', 21, num_format)

    def _raw_data_to_excel(self, excel_writer):
        # Export raw BOND data
        for i in self.market_risk_data[-2:]:
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
            ws.set_column('A:G', 18)
            ws.set_column('H:H', 7)
            ws.set_column('I:I', 22)
            ws.set_column('J:J', 17)
            ws.set_column('K:K', 13, num_format)
            ws.set_column('L:L', 13)
            ws.set_column('M:M', 11, curr_format)
            ws.set_column('N:N', 18, num_format)
            ws.set_column('O:O', 12, curr_format)
            ws.set_column('P:P', 16, num_format)
            ws.set_column('Q:Q', 16, curr_format)
            ws.set_column('R:R', 10, num_format)

    def generate_reports(self, excel: bool = False, html: bool = False):
        if excel:
            self.generate_excel_report()

        if html:
            self.generate_html_report()
