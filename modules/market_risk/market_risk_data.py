import pandas as pd
import logging

try:
    from .fc_computed_amounts import FCComputedAmounts
    from ..utils import get_report_period
except ImportError:
    import sys, os
    if os.getcwd().endswith('InwardsRe'):
        sys.path.insert(0, os.path.abspath('./modules'))
        sys.path.insert(0, os.path.abspath('./modules/market_risk'))
    from utils import get_report_period
    from fc_computed_amounts import FCComputedAmounts

class MarketRiskData(FCComputedAmounts):
    def __init__(self, folder_path: str, report_date: str) -> None:
        FCComputedAmounts.__init__(self, folder_path=folder_path)

        self.folder_path = folder_path
        self.data = None
        self._DATA_NAME = None
        self.report_date = pd.to_datetime(report_date)
        self.report_period = get_report_period(self.report_date)

        self.check_folder_path_date()
    
    def __repr__(self):
        qq = self.report_date.quarter
        yyyy = self.report_date.year
        return f"<{self._DATA_NAME}_DATA Class-{yyyy}Q{qq}>"

    def __lt__(self, other):
        return self.report_date < other.report_date
    
    def __eq__(self, o: object) -> bool:
        return self.report_date == o.report_date
    
    def check_folder_path_date(self):
        """Check if period from report_date is in folder_path.

        Raises
        ------
        ValueError
            If period is not in folder_path.
        """        
        report_period = get_report_period(self.report_date, full_year_str=True)
        if report_period not in self.folder_path:
            err_msg = f"Report period {report_period} not in path {self.folder_path}."
            logging.warning(err_msg)
    
    def get_scr_sum(self):
        if self._DATA_NAME == 'CURRENCY':
            return self.data['SCR']['SCR'].sum()
        else:
            return self.data['SCR'].sum()

    def match_fc_computed_amount(self):
        return self.FC_COMPUTED.round(1) == self.get_scr_sum().round(1)