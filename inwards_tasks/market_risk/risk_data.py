from abc import abstractclassmethod
import logging
from typing import Union
import pandas as pd
import os

from .utils import get_report_period
from .fc_computed_amounts import import_fc_computed_amounts

class RiskData:
    def __init__(self, folder_path: str = None) -> None:
        self.folder_path = folder_path
        
        self.DATA_NAME = None
        self._data = None
        self._fc_computed = None

    def __repr__(self) -> str:
        return f"<class {self.DATA_NAME}RiskData-{self.report_period}>"
    
    def __eq__(self, o: object) -> bool:
        return self.report_date == o.report_date
    
    def __lt__(self, o: object) -> bool:
        return self.report_date < o.report_date
    
    @property
    def data(self):
        return self._data.copy()
    
    @property
    def scr(self):
        return self.data['SCR'].sum()

    @property
    def report_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.data['REPORTING_DAT'].unique()[0])
    
    @property
    def report_period(self):
        return get_report_period(self.report_date)
    
    @property
    def fc_computed(self) -> int:
        return self._fc_computed

    @fc_computed.setter
    def fc_computed(self, df: pd.DataFrame) -> None:
        df_date = pd.to_datetime(df.index.unique()[0])
        if df_date != self.report_date:
            err_msg = "FC_COMPUTED_AMOUNTS has different reporting date.\n"
            err_msg += f"Date should be as at {self.report_date}, but got {df_date}."
            raise ValueError(err_msg)
        else:
            # if dates match, then assign amount to fc_computed variable
            self._fc_computed = df['COMPUTED_AMT_EUR'].values[0]
    
    def is_scr_matching_fc_computed(self) -> bool:
        if self.fc_computed is None:
            # in case there is no FC_COMPUTED_AMOUNTS file loaded,
            # we return False
            logging.warning(f"No FC_COMPUTED for {self.DATA_NAME}/{self.report_period}")
            return False
        
        scr = self.scr.round(2)
        fc_comp = self.fc_computed.round(2)
        if scr == fc_comp:
            return True
        else:
            msg = f'SCR {scr} and FC_COMPUTED {fc_comp} do not match '
            msg += 'for {self.report_period}.'
            logging.warning(msg)
            return False

    @abstractclassmethod
    def import_data(self) -> None:
        raise NotImplementedError
    
    @abstractclassmethod
    def process_data(self) -> None:
        raise NotImplementedError
    
    def import_fc_computed(self) -> Union[None, pd.DataFrame]:
        filepath = os.path.join(self.folder_path, 'FC_COMPUTED_AMOUNTS.txt')
        return import_fc_computed_amounts(filepath)
