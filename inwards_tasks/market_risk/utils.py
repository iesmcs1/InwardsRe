from typing import List, Union
import pandas as pd

# Use as header in all Plotly plots
PLOT_TITLE_FONT_FORMAT = {'family': 'Segoe UI',
                          'size': 30,
                          'color': '#111827'}

# This is the sequence of coloring for Corporate material
CORPORATE_COLOR = ('#dc0028',
                   '#bebebe',
                   '#d99694',
                   '#ebebeb',
                   '#91827C',
                   '#953735',
                   '#414141')

CORPORATE_VALUE_COLOR = {'negative': '#dc0028',
                         'less_negative': '#d99694',
                         'neutral': '#9cb5c3',
                         'less_positive': '#9baa88',
                         'positive': '#598762'}


def produce_unique_rows(data: pd.DataFrame,
                        keys: List[str]) -> pd.DataFrame:

    report_date_col = 'REPORTING_DAT'
    
    # DataFrame with only unique Primary Keys
    pks = data[keys].drop_duplicates().copy()

    # Unique dates in DataFrame
    unique_dates = data[report_date_col].unique()

    df = None
    for date in unique_dates:
        # Assing single date to Primary Key rows, as to have multiple dates
        # for same unique row
        pks[report_date_col] = date
        
        # Concat everything
        df = pd.concat([df, pks])
    
    # Columns to use when sorting data. Also, we want these columns to
    # merge back in the original data
    sort_cols = keys + [report_date_col]

    df_all = df.merge(data, how='outer', on=sort_cols)

    df_all.sort_values(by=sort_cols, inplace=True)

    df_all.reset_index(drop=True, inplace=True)
    
    return df_all

def get_report_period(date: Union[str, pd.Timestamp], full_year_str=False) -> str:
    """
    Convert date to pattern YYQX.

    Notes
    -----
    YY is the last 2 digits of the year in `date`.

    X is the quarter that `date` falls into.

    Parameters
    ----------
    date : pd.Timestamp or str
    full_year_str : bool
        if True, return 4-digit year string, else 2-digit year string.

    Examples
    --------
    >>> get_report_period('31/12/2020')
    '20Q4'
    >>> get_report_period('30/09/2021', full_year_str=True)
    '2021Q3'
    """
    if not isinstance(date, pd.Timestamp):
        ts = pd.to_datetime(date, dayfirst=True)
    elif isinstance(date, pd.Timestamp):
        ts = date
    period = '{}Q{}'.format(ts.year, ts.quarter)
    if full_year_str:
        return period
    else:
        return period[2:]

def add_attributes_to_html_table(html: str,
                                 data_search: bool,
                                 data_pagination: bool,
                                 data_sorter_cols: List[str] = None,
                                 remove_nan: bool = False,
                                 remove_zeros: bool = False) -> str:
    """Add elements to HTML table to make it interactable.

    Parameters
    ----------
    html : str
        HTML table element as str.
    data_search : bool
        If True, add search box to table.
    data_pagination : bool
        If True, add data pagination to table.
    data_sorter_cols : List[str], optional
        List of columns with number data, to apply numSorter function
        This avoids applying number sorting to a str column.
    remove_nan : bool, optional
        If True, removes all "nan" text from html tables, to make 
        data clearer for user
    remove_zeros : bool, optional
        If True, removes all zeros from table. This applies only if cell
        in table contains a single zero (0).

    Returns
    -------
    str
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    # Import only when necessary
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')

    table_tag = soup.find('table')
    table_tag['class'] = 'table'
    table_tag['data-toggle'] = 'table'
    if data_search:
        table_tag['data-search'] = 'true'

    if data_pagination:
        table_tag['data-pagination'] = 'true'

    for th_elem in soup.findAll('th'):
        # all Tables are sortable
        th_elem['data-sortable'] = 'true'

        if (data_sorter_cols is not None) and (th_elem.text in data_sorter_cols):
            # if Table header is in list of columns passed,
            # add data-sorter attribute, which is always numSorter
            th_elem['data-sorter'] = 'numSorter'

    if remove_nan is True:
        # NaN appear in Table when using Pandas diff()
        for td in soup.findAll('td'):
            # Loop through all td's elements
            if td.text in ['nan', '0']:
                # If text is nan, then we replace it with empty strings
                td.string.replace_with('')


    return soup
