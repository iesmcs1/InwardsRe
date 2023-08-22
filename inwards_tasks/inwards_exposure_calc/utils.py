import pandas as pd

def get_report_period_str(date) -> str:
    """Returns string representing a date with Year and Quarter.

    Parameters
    ----------
    date : pandas.Timestamp or str

    Examples
    --------
    >>> get_report_period_str('31/12/2020')
    '20Q4'

    >>> date = pd.to_datetime('30/09/2020')
    >>> get_report_period_str(date)
    '20Q3'
    """

    if not type(date) is pd.Timestamp:
        date = pd.to_datetime(date, dayfirst=True)
    period = '{}Q{}'.format(date.year, date.quarter)
    return period[2:]
