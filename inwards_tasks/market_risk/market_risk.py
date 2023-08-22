import pandas as pd
import numpy as np
from typing import List, Union


class MarketRisk:
    ID_COLUMNS = None
    ISSUER_DATA_COLS = None

    def __init__(self) -> None:
        # datasets hold each dataset individually
        self._datasets = list()

        # data has a single dataframe with all entries analysed together
        self._data = None

    @property
    def dates_added(self) -> Union[List[None], List[pd.Timestamp]]:
        dates = [data.report_date for data in self.datasets]
        dates.sort()
        
        return dates
    
    @property
    def datasets(self) -> List[object]:
        return self._datasets
    
    @datasets.setter
    def datasets(self, data):
        self._datasets.append(data)
        self._datasets.sort()
    
    @property
    def issuer_data(self) -> Union[None, pd.DataFrame]:
        #concatenate all datasets together
        df = pd.concat([
            dataset.data[self.ISSUER_DATA_COLS] for dataset in self.datasets
        ])

        # Sort by date, as to keep only latest
        df.sort_values(by='REPORTING_DAT', inplace=True)

        unique_id_col = self.ISSUER_DATA_COLS[1]

        df.drop_duplicates(subset=unique_id_col,
                           keep='last',
                           inplace=True,
                           ignore_index=True)

        return df.drop('REPORTING_DAT', axis=1)
    
    def is_scr_matching_fc_computed(self) -> bool:
        for data in self.datasets:
            # iterate through all datasets
            if not data.is_scr_matching_fc_computed():
                # if dataset has SCR not matching FC_COMPUTED, then it's 
                # preliminary
                return False
        # If all are matching, then return True, and it's final amounts.
        return True
    
    def calculate_scr_movement(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(by=self.ID_COLUMNS)['SCR'].diff().fillna(0)
    
    def calculate_scr_purch_sold_movements(self,
                                           df: pd.DataFrame) -> pd.Series:
        filter_sold = df['BOND_STATUS'] == 'SOLD'
        filter_purchased = df['BOND_STATUS'] == 'PURCHASED'

        return np.where((filter_sold) | (filter_purchased),
                        df['Diff'],
                        0)