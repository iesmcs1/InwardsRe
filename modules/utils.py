from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from typing import List, Union

from pandas._libs.tslibs.timestamps import Timestamp

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



def get_inwards_folder():
    "Find InwardsRe folder and return absolute path to it."
    # start with working directory
    path_str = '.'
    abs_path = os.path.abspath(path_str)
    # just to make sure that if InwardsRe folder is not 5 parent folders above,
    # operation will raise error.
    flag_safety = 0
    while not abs_path.endswith('InwardsRe') and flag_safety < 5:
        path_str = '..\\' + path_str
        abs_path = os.path.abspath(path_str)
        flag_safety += 1

    if flag_safety == 5:
        raise NotADirectoryError(
            'InwardsRe folder does not exist. Please rename main folder to InwardsRe.')
    else:
        return abs_path


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

def get_date_from_period(period: str) -> pd.Timestamp:
    """Return Timestamp date from given period.

    Parameters
    ----------
    period : str
        str in format 'YYYYQX' or 'YYQX'

    Returns
    -------
    pd.Timestamp
        Timestamp representing given period.
    
    Examples
    --------
    >>> get_date_from_period('2021Q2')
    Timestamp('2021-06-30 00:00:00')
    >>> get_date_from_period('20Q1')
    Timestamp('2020-03-31 00:00:00')
    """    
    year = period[:-2]
    if len(year) == 2:
        year = '20' + year
    # Period uses Quarter, so we multiply by 3 to get month,
    # then convert back to str
    month = str(int(period[-1]) * 3)

    if month in ['3', '12']:
        day = 31
    else:
        day = 30

    return pd.to_datetime(f"{year}-{month}-{day}")


def get_market_risk_subfolder_path(risk_category: int, 
                                   year: int = None,
                                   quarter: int = None) -> str:
    """Return full path of Market Risk folder category.

    Parameters
    ----------
    risk : int
        Risk category number. The sequence of risk is:
        - 1 - Interest Rate
        - 2 - Equity
        - 3 - Property
        - 4 - Spread
        - 5 - Concentration
        - 6 - Currency

    year : int, optional
        year folder inside risk category, by default None
    quarter : int, optional
        quarter folder inside risk category, by default None

    Returns
    -------
    str
        full path to risk category.

    Raises
    ------
    ValueError
        year and quarter must be both int or None.
    """    

    
    mr_subfolders = {1: '1. Interest Rate Risk',
                     2: '2. Equity Risk',
                     3: '3. Property Risk',
                     4: '4. Spread Risk',
                     5: '5. Concentration Risk',
                     6: '6. Currency Risk'}
    
    folderpath = os.path.join(NON_GROUP_FOLDER_PATH,
                              'Actuarial',
                              'Gerard Coffey',
                              'Solvency Capital Requirements',
                              '1. Market Risk',
                              mr_subfolders[risk_category])
    if type(year) is type(quarter):
        folderpath = os.path.join(folderpath,
                                  str(year),
                                  str(year)+'Q'+str(quarter))
    else:
        raise ValueError(
            "'year and 'quarter' must both be none OR both contain an integer.")
    return folderpath


def add_attributes_to_html_table(html: str,
                                 data_search: bool,
                                 data_pagination: bool,
                                 data_sorter_cols: List[str] = None) -> str:
    """Add elements to HTML table to make it interactable.

    Parameters
    ----------
    html : str
        HTML table element as str.
    data_search : bool
        If True, add search box to table.
    data_pagination : bool
        If True, add data pagination to table.
    data_sortable : bool
        If True
    data_sorter : List[str], optional
        list of columns with number data, to apply numSorter function
        This avoids applying number sorting to a str column.

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

    return soup
