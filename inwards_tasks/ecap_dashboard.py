from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
from .utils import get_report_period
import logging


class ReportedRun:
    COL_RENAME_MAPPING = {'EXP_GROSS_OF_RETRO': 'TPE',
                          'EC_CONSUMPTION_ND': 'ECAP',
                          'EXPECTED_LOSS': 'EL'}
    def __init__(self,
                 so_report_filepath: str,
                 epi_filepath: Union[None, str] = None,
                 cedant_info: Union[None, pd.DataFrame] = None) -> None:
        
        self._data = None
        self._epi = None
        
        # Importing done by ECap Dashboards class, so we assign directly
        self.cedant_info = cedant_info

        self.import_so_reporting(so_report_filepath)
        if epi_filepath is not None:
            self.import_epi(epi_filepath)
    
    def __repr__(self) -> str:
        return f"<class ReportedRun-{self.report_period}>"
    
    def __eq__(self, o: object) -> bool:
        if isinstance(o, pd.Timestamp):
            return self.report_date == o
        else:
            return self.report_date == o.report_date
    
    def __lt__(self, o: object) -> bool:
        return self.report_date < o.report_date
    
    @property
    def bond_data(self) -> pd.DataFrame:
        filter_bond = self.data['MODEL_TYPE'] == 'B'
        return self.data.loc[filter_bond].copy()
    
    @property
    def credit_data(self) -> pd.DataFrame:
        filter_cred = self.data['MODEL_TYPE'] == 'C'
        return self.data.loc[filter_cred].copy()
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()
    
    @data.setter
    def data(self, df: pd.DataFrame) -> None:
        # convert to timestamp
        df['REPORTING_PERIOD'] = pd.to_datetime(df['REPORTING_PERIOD'])

        # Add helper column to calculate any Weighted Average PD
        df['POD_WA_HELPER'] = df['ULTIMATE_POD'] * df['EXP_GROSS_OF_RETRO']

        df['RSQRD_WA_HELPER'] = (df['ULTIMATE_RSQUARED']
                                 * df['EXP_GROSS_OF_RETRO'])

        # Add Comp column to make mapping Cedants easier
        df['COMP'] = df['CUSTOMER_ID'].str[:5]

        # Add Bond or Credit column
        df['MODEL_TYPE'] = df['MODEL_SUB_TYPE'].str[0]

        # Add Exposure Type KN (Known) or UNK (Unknown)
        df['EXPOSURE_TYPE'] = np.where(
            df['MODEL_SUB_TYPE'].str.endswith('UNK'),
            'UNK',
            'KN'
        )

        self._data = df
    
    @property
    def report_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.data['REPORTING_PERIOD'].unique()[0])
    
    @property
    def report_date_str(self) -> str:
        return "{}/{}/{}".format(self.report_date.day,
                                 self.report_date.month,
                                 self.report_date.year)
    
    @property
    def report_period(self) -> str:
        return get_report_period(self.report_date)
    
    @property
    def epi(self) -> pd.DataFrame:
        if self._epi is None:
            # If no EPI file was lodaded, then do not proceed.
            return None
        
        return self._epi.copy()
    
    @epi.setter
    def epi(self, df: pd.DataFrame) -> None:

        df_model_type = self.data[['CONTRACT_ID', 'MODEL_TYPE']].drop_duplicates()

        # Inner used because we want only contracts that are in the IM
        df_mr = df.merge(df_model_type, on='CONTRACT_ID', how='inner')

        df_mr['CUSTOMER_ID'] = df_mr['CONTRACT_ID'].str[:10]

        epi_cols = ['CONTRACT_ID',
                    'CUSTOMER_ID',
                    'EPI is Rev EPI or EPI',
                    'YearFrac',
                    'Run-off',
                    'EPI',
                    'MODEL_TYPE']
        
        # Slicing full DF to make sure all columns are present
        self._epi = df_mr[epi_cols]
    
    @property
    def epi_by_cedant(self) -> pd.Series:
        if self.epi is None:
            # If no epi was lodaded, then do not proceed.
            return None

        df = self.epi.copy()
        # Create COMP column to aggregate by Company
        df['COMP'] = df['CONTRACT_ID'].str[:5]

        # Slice df to only 2 columns and group to aggregate
        # and squeeze to turn into Series
        return df[['COMP', 'EPI']].groupby('COMP').sum().squeeze()
    
    def import_so_reporting(self, filepath: str) -> None:
        self.data = pd.read_csv(filepath, sep='\t')
        return None
    
    def import_epi(self, filepath: str) -> None:
        self.epi = pd.read_csv(filepath, sep='\t')
        return None
    
    def _model_type_arg_validator(
        self,
        model_type: Union[None, str] = None
    ) -> pd.DataFrame:
        """Helper function to validate model_type argument.

        Parameters
        ----------
        model_type : None, 'bond', 'credit', optional
            If 'bond', then returns bond data only,
            Else if 'credit', then returns bond data only,
            Else if None, then return entire dataset,
            by default None

        Returns
        -------
        pd.DataFrame
            Dataset of specified model_type

        Raises
        ------
        ValueError
            if model_type is not 'bond', 'credit' or None, raise error
        """    
        if model_type == 'bond':
            return self.bond_data
        elif model_type == 'credit':
            return self.credit_data
        elif model_type == None:
            return self.data
        else:
            err_msg = "model_type argument can be None, 'credit' or 'bond'. "
            err_msg += f"'{model_type}' was passed."
            raise ValueError(err_msg)
        
    def get_data(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None,
        groupby: List[str] = None
    ) -> pd.DataFrame:

        df = self._model_type_arg_validator(model_type=model_type)

        if isinstance(column_filter, dict):
            # if there are columns to filter, then apply filter
            for col, val in column_filter.items():
                df = df.loc[df[col] == val].copy()

        if groupby is not None:
            # If groupby is not note, then we aggregate data
            df = df.groupby(by=groupby, as_index=False).sum()

            # Normalize Rating using TPE_EUR. This can be done in any grouping
            df['POD'] = df['POD_WA_HELPER'] / df['EXP_GROSS_OF_RETRO']
            df['RSQUARED'] = df['RSQRD_WA_HELPER'] / df['EXP_GROSS_OF_RETRO']

            df.drop(columns=['ULTIMATE_POD', 'ULTIMATE_RSQUARED', 'CALCRUN'],
                    inplace=True)

        return df

    def get_pod(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:
        # Use function to filter any kwargs used.
        # This also validates the model_type argument
        df = self.get_data(model_type=model_type,
                              column_filter=column_filter)

        wa_pod = (df['POD_WA_HELPER'].sum()
                  / df['EXP_GROSS_OF_RETRO'].sum())

        return pd.Series([wa_pod],
                         index=[self.report_date],
                         name='POD')
    
    def get_ecap(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:
        df = self.get_data(model_type=model_type,
                           column_filter=column_filter)
        
        return pd.Series([df['EC_CONSUMPTION_ND'].sum()],
                         index=[self.report_date],
                         name='ECAP')
    
    def get_tpe(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:
        df = self.get_data(model_type=model_type,
                           column_filter=column_filter)
        
        return pd.Series([df['EXP_GROSS_OF_RETRO'].sum()],
                         index=[self.report_date],
                         name='TPE')

    def get_rsquared(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:
        """Return RSquared from datasets.

        Parameters
        ----------
        model_type : Union[None, str], optional
            [description], by default None
        kwargs:
            if used, has to be of length == 1, and to be one of the
            columns in the dataset, being equal to some value in that
            respective column in the dataset.
            e.g.: CUSTOMER_ID='00794B1 01', CONTRACT_ID='36193B1 0121', etc.

        Returns
        -------
        pd.Series
            [description]
        """
        # Use function to filter any kwargs used.
        # This also validates the model_type argument
        df = self.get_data(model_type=model_type,
                           column_filter=column_filter)

        wa_rsquared = (df['RSQRD_WA_HELPER'].sum()
                       / df['EXP_GROSS_OF_RETRO'].sum())

        return pd.Series([wa_rsquared],
                         index=[self.report_date],
                         name='RSQUARED')
    
    def get_epi(
        self,
        model_type: Union[None, str] = None,
        customer_id: Union[None, str] = None
    ) -> pd.Series:
        
        df_epi = self.epi.copy()
        if model_type is not None:
            # Filter for Bond or Credit
            df_epi = df_epi.loc[
                df_epi['MODEL_TYPE'].str.startswith(model_type[0].upper())
            ].copy()
        
        if customer_id is not None:
            df_epi = df_epi[df_epi['CUSTOMER_ID'] == customer_id].copy()
        
        return pd.Series([df_epi['EPI'].sum()],
                         index=[self.report_date],
                         name='EPI')
    
    def group_data_by_contract(
            self,
            model_type: Union[None, str] = None,
            include_epi: bool = False,
            include_cedant_name: bool = False
        ) -> pd.DataFrame:
        """Return data grouped by Contract ID

        Parameters
        ----------
        model_type : Union[None, str], optional
            [description], by default None
        include_epi : bool, optional
            [description], by default False
        include_cedant_name : bool, optional
            [description], by default False

        Returns
        -------
        pd.DataFrame
            [description]
        """        
        df = self.get_data(model_type=model_type)

        # Group data to have sums
        df_gr = df.groupby('CONTRACT_ID', as_index=False).sum()

        # Calculate weighted average PD
        df_gr['PD'] = df_gr['POD_WA_HELPER'] / df_gr['EXP_GROSS_OF_RETRO']

        df_gr.drop(['ULTIMATE_POD', 'CALCRUN'], axis=1, inplace=True)

        if include_epi:
            epi_cols = ['CONTRACT_ID', 'EPI']
            df_gr = df_gr.merge(self.epi[epi_cols],
                                on='CONTRACT_ID',
                                how='left')
        
        if include_cedant_name:
            df_gr = df_gr.merge(
                self.cedant_info[['CONTRACT_ID', 'Cedant']],
                on='CONTRACT_id',
                how='left'
            )
        
        # Add Balloon Reference for when analysing Cedants
        df_gr['CUSTOMER_ID'] = df_gr['CONTRACT_ID'].str[:10]
        
        return df_gr
    
    def group_data_by_cedant(
        self,
        model_type: str,
        include_epi: bool = False,
        include_cedant_name: bool = False
    ) -> pd.DataFrame:
        df = self.group_data_by_contract(model_type=model_type,
                                         include_epi=include_epi)

        df['COMP'] = df['CONTRACT_ID'].str[:5]

        df_gr = df.groupby('COMP', as_index=False).sum()

        # Calculate weighted average PD per Cedant
        df_gr['PD'] = df_gr['POD_WA_HELPER'] / df_gr['EXP_GROSS_OF_RETRO']

        df_gr['ECAP / TPE'] = (df_gr['EC_CONSUMPTION_ND']
                               / df_gr['EXP_GROSS_OF_RETRO'])
        
        if include_epi:
            df_gr['ECAP / EPI'] = df_gr['EPI'] / df_gr['EXP_GROSS_OF_RETRO']

        if include_cedant_name:
            df_gr['CEDANT'] = df_gr['COMP'].map(self.cedants)

        df_gr.rename(columns=self.COL_RENAME_MAPPING, inplace=True)

        return df_gr


class ECapDashboard:
    def __init__(self) -> None:
        self._datasets = list()
        self._cedant_info = None

    @property
    def datasets(self) -> List[ReportedRun]:
        return self._datasets
    
    @datasets.setter
    def datasets(self, data: ReportedRun) -> None:
        if isinstance(data, ReportedRun):
            self._datasets.append(data)
            self._datasets.sort()
        else:
            warn_msg = f"Datasets only takes type 'ReportedRun'. {type(data)} was passed."
            logging.warning(warn_msg)
    
    @property
    def cedant_info(self) -> pd.Series:
        if self._cedant_info is not None:
            return self._cedant_info.copy()
        else:
            print("Cedant information not loaded.")
    
    @cedant_info.setter
    def cedant_info(self, df: pd.DataFrame) -> None:
        # Create Balloon ID column
        df['CUSTOMER_ID'] = df['Comp'] + df['Seq']

        # Create Contract ID column
        df['CONTRACT_ID'] = df['CUSTOMER_ID'] + df['UW Yr'].astype(str)

        # Sort values by UW Year, as to have only latest
        df.sort_values('UW Yr', ascending=False, inplace=True)

        df.drop_duplicates(subset='CUSTOMER_ID', keep='first', inplace=True)

        self._cedant_info = df
    
    def import_cedant_info(self, filepath: str) -> None:
        df = pd.read_csv(filepath, sep='\t', dtype={'Comp': str})

        self.cedant_info = df

        return None
    
    def import_run_data(self,
                        so_report_filepath: str,
                        epi_filepath: Union[None, str] = None) -> None:
        
        
        rr = ReportedRun(so_report_filepath=so_report_filepath,
                         epi_filepath=epi_filepath,
                         cedant_info=self.cedant_info)
        
        # Add class ReportedRun to ECapDashboard class
        self.datasets = rr
        print(f"Data added for {rr.report_period}")
        
        return None
    
    def get_tpe(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:
        # initialize empty Series
        s = pd.Series(dtype=str, name='TPE')

        for dataset in self.datasets:
            # for each dataset, concatenate rsquared values
            s = pd.concat([s, dataset.get_tpe(model_type=model_type,
                                              column_filter=column_filter)])

        return s
    
    def get_pod(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:

        # initialize empty Series
        s = pd.Series(dtype=str, name='POD')

        for dataset in self.datasets:
            # for each dataset, concatenate rsquared values
            s = pd.concat([s, dataset.get_pod(model_type=model_type,
                                              column_filter=column_filter)])

        return s

    def get_rsquared(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:

        # initialize empty Series
        s = pd.Series(dtype=str, name='RSQUARED')

        for dataset in self.datasets:
            # for each dataset, concatenate rsquared values
            s = pd.concat([
                s,
                dataset.get_rsquared(model_type=model_type,
                                     column_filter=column_filter)
            ])

        return s

    def get_ecap(
        self,
        model_type: Union[None, str] = None,
        column_filter: Union[None, Dict[str, str]] = None
    ) -> pd.Series:

        # initialize empty Series
        s = pd.Series(dtype=str, name='ECAP')

        for dataset in self.datasets:
            # for each dataset, concatenate rsquared values
            s = pd.concat([s, dataset.get_ecap(model_type=model_type,
                                               column_filter=column_filter)])

        return s
    
    def get_epi(self,
                model_type: Union[None, str] = None,
                customer_id: Union[None, str] = None) -> pd.Series:
        # initialize empty Series
        s = pd.Series(dtype=str, name='EPI')

        for dataset in self.datasets:
            # for each dataset, concatenate rsquared values
            s = pd.concat([s, dataset.get_epi(model_type=model_type,
                                              customer_id=customer_id)])

        return s
    
    def _dates_arg_validator(self,
                             dates: Union[None, List[str]]) -> Tuple[str]:
        if dates is None:
            # index of datasets, which is always ordered by oldest to newest
            # as to have newest at the end (index -1)
            idx_old = -2
            idx_new = -1
        elif len(dates) == 2:
            dates_ts = [pd.to_datetime(dt) for dt in dates]
            dates_ts.sort()
            # dates_ts will be a list of len 2, ordered from oldest to newest
            # index 0 is the oldest, and 1 is newest
            try:
                idx_old = self.datasets.index(dates_ts[0])
                idx_new = self.datasets.index(dates_ts[1])
            except ValueError:
                # if date passed is not in datasets list.
                err = f"Invalid date passed. "
                err += "These are dates available:\n"
                err += f"{[dt.report_date_str for dt in self.datasets]}"
                raise ValueError(err)
        else:
            err = "dates argument needs to be a list of 2 dates. "
            err += f"{type(dates)} of len {len(dates)} was passed."
            raise ValueError(err)
        
        return (idx_old, idx_new)
    
    def get_ecap_movement(
        self,
        model_type: str,
        dates: Union[None, List[str]] = None
    ) -> pd.DataFrame:

        idx_old, idx_new = self._dates_arg_validator(dates=dates)

        cols_to_keep = ['COMP', 'ECAP']

        # suffixes used when merging dataframes
        suffix_old = "-"+self.datasets[idx_old].report_period
        suffix_new = "-"+self.datasets[idx_new].report_period

        # Create dataframe for old data
        df_old = self.datasets[idx_old].group_data_by_cedant(
            model_type=model_type,
            include_epi=False,
        )[cols_to_keep]

        # Create dataframe for new data
        df_new = self.datasets[idx_new].group_data_by_cedant(
            model_type=model_type,
            include_epi=False,
        )[cols_to_keep]

        # Merge old into new, keeping all cedants from both (how="outer")
        df = df_old.merge(df_new,
                          on='COMP',
                          how='outer',
                          suffixes=(suffix_old, suffix_new))
        
        df_cedant = self.cedant_info[['Comp', 'Cedant']].drop_duplicates()
        s_cedant = df_cedant.set_index('Comp').squeeze().rename_axis(None)

        # Insert Cedant Name's column
        df.insert(loc=1,
                  column='Cedant',
                  value=df['COMP'].map(s_cedant))

        df['Diff'] = (df[f'ECAP{suffix_new}'].fillna(0)
                      - df[f'ECAP{suffix_old}'].fillna(0))

        df['Diff %'] = df['Diff'] / df[f'ECAP{suffix_old}'].fillna(0)

        # If ECap Diff is negative, then ordered by negative to positive
        # Else, by positive to negative
        is_ascending = df['Diff'].sum() > 0

        print(1)

        return df.sort_values('Diff', ascending=is_ascending)
