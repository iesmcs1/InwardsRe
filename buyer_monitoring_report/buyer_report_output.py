import os
from typing import Dict, List, Union
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils import FORBIDDEN_STATUSES, CORPORATE_COLOR, CORPORATE_VALUE_COLOR
from buyer_rating_report import NamedExposure

from markdown2 import Markdown


def return_plot_fig(fig,
                    for_html: bool = False):
    fig.update_layout(template='plotly_white')

    if for_html:
        return fig.to_html(include_plotlyjs=False,
                           full_html=False,
                           config={'responsive': True})

    return fig


class BuyerReportHTML:
    PLOT_LABELS = {"RATING": "Rating",
                   "TPE_EUR": "TPE (EUR)",
                   "TOTAL_TPE": "Total TPE (EUR)",
                   "BUYER_ID": "BUS ID",
                   "BUYER_INDUSTRY": "Buyer Industry",
                   "CEDANT_NAME": "Cedant Name",
                   'BUYER_INDUSTRY': "Buyer Industry",
                   'Rating-Diff %': "Rating MoM %"}

    def __init__(self, named_exposure: NamedExposure) -> None:
        self.named_exposure = named_exposure

        # Initially, we will only be comparing the 2 most recent points
        # in time. If there's a need to compare different points in time,
        # we could implement it
        # Since named_data is in ascending order, -1 means it's the latest data
        self.current_date_idx = -1
        self.previous_date_idx = -2

        self.current_data = self.named_exposure.named_data[self.current_date_idx]
        self.previous_data = self.named_exposure.named_data[self.previous_date_idx]
    
    def plot_rating_composition_development(self, for_html: bool = False):
        fig = go.Figure()

        hovertemplate = "<b>Rating Range</b>: %{x}<br>"
        hovertemplate += "<b>TPE</b>: %{y:,.1f}M<br>"

        # In this plot, we'll always have the oldest dataset, and the
        # 2 most recent ones. Since named_data is from oldest to newest,
        # 0 is for the oldest one, and -2 and -1 for the 2 most recent ones.
        named_data_idx_to_add = [0, -2, -1]

        for idx in named_data_idx_to_add:
            data = self.named_exposure.named_data[idx]
            df = data.get_data(groupby=['BUYER_RATING_BAND'])

            fig.add_trace(
                    go.Bar(
                            name=data.date_str,
                            x=df['BUYER_RATING_BAND'],
                            y=df['TPE_EUR'] / 1e6,   #show value in millions
                            hovertemplate=hovertemplate
                    )
            )

        # Set title and apply corporate coloring
        fig.update_layout(title='Rating Composition Development',
                          colorway=CORPORATE_COLOR)
        
        fig.update_yaxes(title_text='TPE (EUR)',
                         tickformat=',.0f',
                         ticksuffix='m')

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_rating_by_tpe(self, for_html: bool = False):

        df_grp = self.current_data.get_data(groupby=['KEY', 'SHORT_NAME'])

        fig = px.scatter(df_grp,
                         x="RATING",
                         y="TPE_EUR",
                         hover_data=["KEY", "SHORT_NAME"],
                         labels=self.PLOT_LABELS,
                         color_discrete_sequence=CORPORATE_COLOR)
        
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(
                        args=[{'yaxis.type': 'scatter'}],
                        label="Linear Scale",
                        # relayout is used to change axis from Log to Scatter
                        # and vice-versa
                        method="relayout"
                    ),
                    dict(
                        args=[{'yaxis.type': 'log'}],
                        label="Log Scale",
                        method="relayout"
                    )
                ]),
                pad={"l": 10, "t": 10},
                showactive=True,
                # x=1 to position buttons at right-hand side
                x=1,
                xanchor="right",
                # y=1.15 to position buttons at the top
                y=1.15,
                yanchor="top"
            ),
        ]

        fig.update_layout(title='TPE x Rating, by Policy',
                          updatemenus=updatemenus)

        hovertemplate = "<b>Balloon ID</b>: %{customdata[0]}<br>"
        hovertemplate += "<b>Short Name</b>: %{customdata[1]}<br>"
        hovertemplate += "<b>Rating</b>: %{x}<br>"
        hovertemplate += "<b>TPE (EUR)</b>: %{y}"
        
        fig.update_traces(hovertemplate=hovertemplate)

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_largest_movements(self,
                               groupby: str,
                               n: Union[None, int] = None,
                               value: str = 'rating',
                               sort_by: str = 'rating',
                               position: str = 'top',
                               showgrid: bool = True,
                               for_html: bool = False):
        """Plot largest rating movements, by groupby argument.

        Parameters
        ----------
        groupby : str
            One column in DataFrame to use to group data.
            E.g.: 'CEDANT_NAME', 'BUYER_INDUSTRY', 'BUYER_COUNTRY', etc.
        n : Union[None, int], optional
            Top n elements to include in graph
        value : str, optional
            How to sort data for plot, by default None
            Useful when plotting TPE and Rating, and there's a need to
            maintain same order for both plots
        sort_by : str, optional
            [description], by default 'rating'
        position : str, optional
            If 'top', then find the largest movements.
            Else if 'bottom', then return smallest movements,
            by default 'top'
        width : int, optional
            Width to be applied to html string output, by default 800
        for_html : bool, optional
            If to export html object (str), by default False

        Returns
        -------
        Union[Plotly graph, str]
            If for_html is True, then return str, else Plotly object
        """        
        
        df = self.named_exposure.get_movements(groupby=groupby)
        
        sufx = (f"-{self.previous_data.date_str}",
                f"-{self.current_data.date_str}")
        
        if value == 'rating':
            value_col = 'Rating-Diff %'
            value_plot_title = 'Rating'
            bar_hovertemplate = "%{x}<br>Rating: %{y:.1f}"
        elif value == 'tpe':
            value = 'TPE_EUR'
            value_col = 'TPE-Diff %'
            value_plot_title = 'TPE'
            bar_hovertemplate = "%{x}<br>TPE: %{y}"
        
        #df.sort_values(by=sort_col, inplace=True)
        if n is None:
            # if not slicing top N in DataFrame, then return all of them.
            n = len(df)

        if position == 'top':
            position_plot_title = 'Deterioration'
            df_top = df.nlargest(n=n, columns=value_col)
        if position == 'bottom':
            position_plot_title = 'Improvement'
            df_top = df.nsmallest(n=n, columns=value_col)

            # Sort values from "largest" to smallest
            # (from -2 to -8, for example)
            df_top.sort_values(by=value_col, ascending=False, inplace=True)
        
        if sort_by == 'rating':
            sort_col = 'Rating-Diff %'
        elif sort_by == 'tpe':
            sort_col = 'TPE-Diff %'
        else:
            print('Invalid sort_by argument. No sorting applied.')
        
        # Sort values as to have largest to smallest (ascending=False)
        df_top.sort_values(by=sort_col, ascending=False, inplace=True)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add old data
        fig.add_trace(
            go.Bar(name=f'{self.previous_data.date_str}',
                   x=df_top[groupby],
                   y=df_top[f'{value.upper()}{sufx[0]}'],
                   hovertemplate=bar_hovertemplate,
                   marker_color=CORPORATE_COLOR[0])
        )

        fig.add_trace(
            go.Bar(name=f'{self.current_data.date_str}',
                   x=df_top[groupby],
                   y=df_top[f'{value.upper()}{sufx[1]}'],
                   hovertemplate=bar_hovertemplate,
                   marker_color=CORPORATE_COLOR[1])
        )

        fig.add_trace(go.Scatter(x=df_top[groupby],
                                 y=df_top[value_col],
                                 name="MoM % Change",
                                 hovertemplate="MoM : %{y:+.1%}<extra></extra>",
                                 marker_color=CORPORATE_COLOR[7]),
                      secondary_y=True)
        
        fig.update_yaxes(secondary_y=True, 
                         title_text='MoM % Change',
                         showgrid=False,
                         tickformat='.0%')
        
        fig.update_yaxes(secondary_y=False,
                         title_text=value_plot_title,
                         showgrid=showgrid)
        
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        by_plot_title = self.PLOT_LABELS[groupby]
        
        if n == len(df):
            # Checks if n was initially None, since it was changed before
            # slicing largest or smallest
            plot_title = f'{value_plot_title} by {by_plot_title}'
        else:
            # If n passed is not None
            plot_title = f'Top {n} Largest {value_plot_title} '
            plot_title += f'{position_plot_title} by {by_plot_title}'
        
        fig.update_layout(barmode='group',
                          title=plot_title)

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_top_adv_rated(self,
                           n: int,
                           rating_filter: str,
                           for_html: bool = False):

        df = self.current_data.get_data(
            filter_dict={'BUYER_IBR_RATING': rating_filter}
        )
        
        # Get grouped data, and select only top 5 Exposures
        df_top_adv_buyers = self.current_data.get_data(
            filter_dict={'BUYER_IBR_RATING': rating_filter},
            groupby=['BUYER_ID']
        ).nlargest(n=n, columns=['TPE_EUR'])
        
        # Create Series with only Buyer ID as Index, and TPE.
        # This will be used to map total TPE. squeeze() to create Series,
        # so it's easier to map below
        s_top_n = df_top_adv_buyers[['BUYER_ID', 'TPE_EUR']].set_index(
            'BUYER_ID').squeeze()

        # Filter filtered DF including only top buyers by TPE
        df_top_n_buyers = self.current_data.get_data(
            filter_dict={'BUYER_ID': s_top_n.index}
        )

        # Sort values to have an organized plot
        df_top_n_buyers.sort_values(['BUYER_ID', 'TPE_EUR'],
                                    ascending=[True, True],
                                    inplace=True)
        
        # Include column with Total TPE because this information will
        # be included in the plot, so user know total TPE for buyer.
        df_top_n_buyers["TOTAL_TPE"] = df_top_n_buyers['BUYER_ID'].map(s_top_n)

        fig = px.bar(df_top_n_buyers,
                 x="BUYER_ID",
                 y="TPE_EUR",
                 category_orders={'BUYER_ID': [i for i in s_top_n.index]},
                 color='BUYER_INDUSTRY',
                 hover_data={'BUYER_NAME': True,
                             "KEY": True,
                             "SHORT_NAME": True,
                             "TOTAL_TPE": ":.2s",
                             "TPE_EUR": ":.2s"},
                 color_discrete_sequence=CORPORATE_COLOR,
                 labels=self.PLOT_LABELS
                 )

        fig.update_layout(title='Top 5 Exposures by Buyer rated 100')

        fig.update_yaxes(showgrid=False)

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_na_to_adv_rating_buyers(self,
                                     adv_rating_threshold: int = 70,
                                     for_html: bool = False):
        # Get data from previous quarter
        df_old = self.previous_data.get_data()

        # Declare filter that will return only buyers not rated in prev month
        filter_rated_na = df_old['BUYER_IBR_RATING'].isna()

        # Apply filter to previous dataset
        s_old_not_rated = df_old.loc[filter_rated_na, 'BUYER_ID'].unique()

        # Variable that will hold only buyers that were not rated and now
        # are rated above 70
        df_new = self.current_data.get_data(filter_dict={
            'BUYER_IBR_RATING': F"> {adv_rating_threshold}",
            'BUYER_ID': s_old_not_rated
        })

        if df_new.shape[0] == 0:
            # If dataframe is empty, then return empty string
            return ''

        plot_title = f'Buyers rated over {adv_rating_threshold}'
        plot_title += ', previously unrated'

        if df_new.shape[0] == 0:
            # If dataframe has no rows, then return empty plot
            fig = px.bar(title=plot_title)
            return fig
        else:
            fig = px.bar(df_new,
                         x='BUYER_NAME',
                         y='TPE_EUR',
                         title=plot_title,
                         labels=self.PLOT_LABELS,
                         color_discrete_sequence=CORPORATE_COLOR)
        
        fig.update_layout(autosize=False)

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_treemap_forbidden_status(
        self,
        date: Union[None, str, pd.Timestamp] = None,
        for_html: bool = False
    ):
        named_exp_data = self.named_exposure.get_data_for_date(date=date)

        # Get data with buyers that have forbidden statuses only.
        df = named_exp_data.get_data(
            filter_dict={'BUYER_STATUS': FORBIDDEN_STATUSES}
        )

        df.to_csv('forbidden_status.txt', sep='\t')

        # Create Treemap figure
        fig = px.treemap(
            df,
            path=[px.Constant("Status"),
                  'BUYER_STATUS',
                  'SHORT_NAME',
                  'BUYER_NAME'],
            # Represents the size of rectangles
            values='TPE_EUR',
            # Using Corporate Colors when mapping discrete values (non-numeric)
            color_discrete_sequence=CORPORATE_COLOR,
            # this corresponds to customdata[0], [1] and [2] respectively
            hover_data=['BUYER_ID', 'KEY', 'SHORT_NAME']
        )

        hovertemplate = "%{label}<br>"
        hovertemplate += "<b>TPE (EUR)</b>: %{value:.4s}<br>"
        hovertemplate += "<b>BUS ID</b>: %{customdata[0]}<br>"
        hovertemplate += "<b>Balloon ID</b>: %{customdata[1]}<br>"
        hovertemplate += "<b>Treaty</b>: %{customdata[2]}<br>"

        fig.update_traces(root_color="lightgrey",
                          hovertemplate=hovertemplate)

        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),
                          title='TPE by Forbidden Statuses')

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_treemap_cedants_buyers_rated_over(
        self,
        rated_over: int,
        date: Union[None, str, pd.Timestamp] = None,
        for_html: bool = False
    ):
        # Get initial raw data class
        named_exp_data = self.named_exposure.get_data_for_date(date=date)

        # get_data() will first filter data with buyers rated over X, and then
        # aggregate data by TPE and Rating. Then, nlargest will return
        # largest 5 TPE_EUR's per Cedant
        cedant_with_large_tpe = named_exp_data.get_data(
            filter_dict={'BUYER_IBR_RATING': f"> {rated_over}"},
            groupby=['CEDANT_NAME']
        ).nlargest(5, columns=['TPE_EUR'])

        # Use get_data() again, applying filter to return only rated above 70
        # And with CEDANT_NAME in cedant_with_large_tpe
        df_over_70 = named_exp_data.get_data(
            filter_dict={'BUYER_IBR_RATING': f"> {rated_over}",
                         'CEDANT_NAME': cedant_with_large_tpe['CEDANT_NAME']},
        )

        color_scale = [CORPORATE_VALUE_COLOR['less_positive'],
                       CORPORATE_VALUE_COLOR['neutral'],
                       CORPORATE_VALUE_COLOR['less_negative'],
                       CORPORATE_VALUE_COLOR['negative']]

        treemap_path = [px.Constant("Cedants"),
                        'CEDANT_NAME',
                        'BUYER_INDUSTRY',
                        'BUYER_NAME']
        


        fig = px.treemap(df_over_70,
                         path=treemap_path,
                         values='TPE_EUR',
                         color='BUYER_IBR_RATING',
                         color_discrete_map={'(?)': 'lightgrey'},
                         color_continuous_scale=color_scale,
                         maxdepth=3,
                         hover_data=['BUYER_ID', 'KEY', 'SHORT_NAME'])

        hovertemplate = "%{label}<br>"
        hovertemplate += "<b>TPE (EUR)</b>: %{value:.4s}<br>"
        hovertemplate += "<b>IBR Rating</b>: %{color: .0f}<br>"
        hovertemplate += "<b>BUS ID</b>: %{customdata[0]}<br>"
        hovertemplate += "<b>Balloon ID</b>: %{customdata[1]}<br>"
        hovertemplate += "<b>Treaty</b>: %{customdata[2]}<br>"

        # Root is the first Section of the Treemap.
        # In this case, it's "Cedants"
        fig.update_traces(root_color="lightgrey",
                          hovertemplate=hovertemplate)

        plot_title = 'Top 5 Cedants with Largest Concentration'
        plot_title += f' of Exposure Rated Over {rated_over}'
        
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),
                          title=plot_title)

        return return_plot_fig(fig, for_html=for_html)
    
    def plot_sparkline(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        y_col_format: str = '',
        include_mom_pct_change: bool = False,
        include_ytd_pct_change: bool = False,
        for_html: bool = False
    ):
        """Plot sparkline graph type, with DF data.

        Notes
        -----
        DataFrame passed to function needs to be 3 columns wide.
        First column needs to be the amounts. The following 2
        columns need to be MoM pct change, and YTD pct change.

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        x_col : str
            _description_
        y_col : str
            _description_
        y_col_format : str, optional
            _description_, by default ''
        include_mom_pct_change : bool, optional
            _description_, by default False
        include_ytd_pct_change : bool, optional
            _description_, by default False
        for_html : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        # Set hover template
        hovertemplate = "<b>%{x}</b><br>"
        hovertemplate += "Rating: %{y:" + y_col_format + "}<br>"
        if include_mom_pct_change:
            hovertemplate += "MoM change: %{customdata[1]:.1%}<br>"
        
        if include_ytd_pct_change:
            hovertemplate += "YTD change: %{customdata[2]:.1%}<br>"
        
        hovertemplate += "<extra></extra>"

        try:
            # If X is index, then this will throw error
            x = df[x_col]
        except KeyError:
            # We access index with . (dot) notation
            x = df.x_col

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[y_col],
                customdata=df.fillna(0),  # NaN in DF were producing errors
                mode='lines+markers',
                name='lines',
                hovertemplate=hovertemplate
            )
        )

        # hide and lock down axes
        fig.update_xaxes(visible=False, fixedrange=True)
        fig.update_yaxes(visible=False, fixedrange=True)
        
        fig.update_layout(showlegend=False,
                          margin=dict(t=10,l=10,b=10,r=10))

        return return_plot_fig(fig, for_html=for_html)
        

    def get_summary_text(self, for_html: bool = False) -> str:
        text = "- Portfolio Weighted Average Rating from {:.2f} to {:.2f}\n".format(
            self.previous_data.get_weighted_average_rating(),
            self.current_data.get_weighted_average_rating()
        )

        text += "- % of Buyers rated above 70: {:.1%}\n".format(
            self.current_data.get_broad_rating_band_sum(pct_of_total=True)['Above 70'].values[0]
        )

        text += "- % of Buyers rated below 50: {:.1%}".format(
            self.current_data.get_broad_rating_band_sum(pct_of_total=True)['Below 50'].values[0]
        )

        if for_html:
            md = Markdown()
            return md.convert(text)
        else:
            return text
    
    def _helper_summary_table_headers(self) -> List[str]:
        # First element of list is empty string
        header_list = ['']

        # Insert current month
        header_list.append(self.named_exposure.named_data[-1].date_str)

        # Insert previous month
        header_list.append(self.named_exposure.named_data[-2].date_str)

        # Insert MoM % change
        header_list.append('MoM %')

        # Insert oldest month
        header_list.append(self.named_exposure.named_data[-0].date_str)

        # Insert MoM % change
        header_list.append('YTD %')

        return header_list
    
    def _helper_summary_table_rows(self) -> List[Dict[str, str]]:
        rows = []

        current_rating = self.current_data.get_weighted_average_rating()
        previous_rating = self.previous_data.get_weighted_average_rating()
        oldest_rating = self.named_exposure.named_data[0].get_weighted_average_rating()

        rows.append(
            {'title': 'Portfolio Rating',
             'current_val': current_rating,
             'previous_val': previous_rating,
             'month_pct_ch': (current_rating
                              - previous_rating) / previous_rating,
             'dec_val': oldest_rating,
             'ytd_pct_ch': (current_rating - oldest_rating) / oldest_rating}
        )

        current_above_70_pct = self.current_data.get_broad_rating_band_sum(
            pct_of_total=True)['Above 70'].values[0]
        
        previous_above_70_pct = self.previous_data.get_broad_rating_band_sum(
            pct_of_total=True)['Above 70'].values[0]
        
        oldest_above_70_pct = self.named_exposure.named_data[0].get_broad_rating_band_sum(
            pct_of_total=True)['Above 70'].values[0]

        rows.append(
            {'title': '% Buyers above 70',
             'current_val': current_above_70_pct,
             'previous_val': previous_above_70_pct,
             'month_pct_ch': (current_above_70_pct
                              - previous_above_70_pct) / previous_above_70_pct,
             'dec_val': oldest_above_70_pct,
             'ytd_pct_ch': (current_above_70_pct
                            - oldest_above_70_pct) / oldest_above_70_pct}
        )

        current_below_50_pct = self.current_data.get_broad_rating_band_sum(
            pct_of_total=True)['Below 50'].values[0]

        previous_below_50_pct = self.previous_data.get_broad_rating_band_sum(
            pct_of_total=True)['Below 50'].values[0]

        oldest_below_50_pct = self.named_exposure.named_data[0].get_broad_rating_band_sum(
            pct_of_total=True)['Below 50'].values[0]

        rows.append(
            {'title': '% Buyers below 50',
             'current_val': current_below_50_pct,
             'previous_val': previous_below_50_pct,
             'month_pct_ch': (current_below_50_pct
                              - previous_below_50_pct) / previous_below_50_pct,
             'dec_val': oldest_below_50_pct,
             'ytd_pct_ch': (current_below_50_pct
                            - oldest_below_50_pct) / oldest_below_50_pct}
        )

        return rows

    
    def to_html(self, open_file: bool = True, timestamp: bool = True) -> None:
        from jinja2 import Environment, FileSystemLoader
        import webbrowser

        # get absolute path for the template folder
        templates_dir = os.path.join(os.path.dirname(__file__),
                                     'templates')

        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('main_body.html')

        filename = self.named_exposure.get_report_name(file_extension='html',
                                                       timestamp=timestamp)

        # Dict with all elements to complete HTML report
        elem = self._get_html_elements()

        with open(filename, 'w') as fh:
            fh.write(template.render(
                # Summary Tab
                table_headers=elem['table_headers'],
                rows=elem['rows'],
                rating_composition_dev=elem['rating_composition_dev'],

                # Cedant Tab
                plot_rating_by_TPE=elem['plot_rating_by_TPE'],
                top_cedant_rating_deterioration=elem['top_cedant_rating_deterioration'],
                treemap_cedants_with_buyers_over_70=elem['treemap_cedants_with_buyers_over_70'],
                
                # Buyer Tab
                plot_na_to_adv_rating_buyers=elem['plot_na_to_adv_rating_buyers'],
                plot_treemap_forbidden_status=elem['plot_treemap_forbidden_status'],
                
                # Industry Tab
                plot_rating_movements_by_industry=elem['plot_rating_movements_by_industry'],
                plot_tpe_movements_by_industry=elem['plot_tpe_movements_by_industry']
            ))

        print(f"HTML report created at {filename}.")
        if open_file:
            webbrowser.open(filename, new=2)
    
    def _get_html_elements(self):
        elem = {}

        elem['table_headers'] = self._helper_summary_table_headers()

        elem['rows'] = self._helper_summary_table_rows()

        elem['rating_composition_dev'] = self.plot_rating_composition_development(
            for_html=True)
        
        elem['summary_text'] = self.get_summary_text(for_html=True)

        elem['plot_rating_by_TPE'] = self.plot_rating_by_tpe(for_html=True)

        elem['top_cedant_rating_deterioration'] = self.plot_largest_movements(
            groupby='CEDANT_NAME',
            n=10,
            for_html=True
        )

        # =====================================================================
        # Cedant Tab

        elem['treemap_cedants_with_buyers_over_70'] = self.plot_treemap_cedants_buyers_rated_over(
            rated_over=70,
            for_html=True
        )

        # =====================================================================
        # Buyer Tab

        elem['plot_na_to_adv_rating_buyers'] = self.plot_na_to_adv_rating_buyers(
            adv_rating_threshold=70,
            for_html=True
        )

        elem['plot_treemap_forbidden_status'] = self.plot_treemap_forbidden_status(
            for_html=True
        )

        # =====================================================================
        # Industry Tab

        elem['plot_rating_movements_by_industry'] = self.plot_largest_movements(
            groupby='BUYER_INDUSTRY',
            # since there aren't manny buyer industry categories, decided
            # to plot all of them, hence n=99 
            n=None,
            value='rating',
            position='bottom',
            sort_by='rating',
            for_html=True
        )

        elem['plot_tpe_movements_by_industry'] = self.plot_largest_movements(
            groupby='BUYER_INDUSTRY',
            # since there aren't manny buyer industry categories, decided
            # to plot all of them, hence n=99 
            n=None,
            value='tpe',
            position='bottom',
            sort_by='rating',
            for_html=True
        )

        return elem
