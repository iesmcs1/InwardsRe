# This file is not in the modules folder because it is supposed to be
# used as a standalone file.

from typing import Dict, Union, List
import pandas as pd
import numpy as np
import re  # regular expression library/module
import logging
import os
from utils import FORBIDDEN_STATUSES

from pandas.errors import ParserError


class NamedExposureData:
    def __init__(self, filepath: str) -> None:
        """Class to deal with Named Exposure data for a single point in time.

        Parameters
        ----------
        filepath : str
            path to file to added to model.
        """        
        self.filepath = filepath
        self._data = None

        self.import_data(filepath=filepath)
    
    def __repr__(self):
        date_str = pd.to_datetime(self.date).strftime('%Y-%b')
        return f"<NAMED_EXPOSURE_DATA Class-{date_str}>"
    
    def __lt__(self, other):
        return self.date < other.date
    
    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            # If object is string, then we compare date str
            return self.date_str == o
        else:
            # Else we're comparing NamedExposureData classes
            return self.date == o.date
    
    @property
    def data(self) -> pd.DataFrame:
        # Return copy as to not alter original data
        return self._data.copy()
    
    @data.setter
    def data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # When loading data by reading txt file (without read_csv),
            # last column is imported with "\n" at the end.
            # This line removes the "\n" at the end
            df.iloc[:, -1] = df.iloc[:, -1].str.replace('\n', "")
        except AttributeError:
            # If last column is not string, then an AttributeError will be
            # raised. Also, if it's not a string, then we know it doesn't
            # have '\n' at the end, so we pass
            pass

        # Two case where there was a blank space at the beginning of header
        df.columns = df.columns.str.strip()

        # Convert Column to datetime
        df['REFRESH_DATE'] = pd.to_datetime(df['REFRESH_DATE'],
                                                   dayfirst=True)
        
        # Normalize column to remove hours, minutes and seconds
        df['REFRESH_DATE'] = df['REFRESH_DATE'].dt.normalize()

        # renaming TPE header
        df.rename(columns={'CLD_TOTAL_AMT_EUR_VAR': 'TPE_EUR'},
                  inplace=True)

        to_numeric = ['BUYER_IBR_RATING',
                      'TPE_EUR']

        # Convert to int because source data has numbers as text
        for col in to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # removing rows with TPE 0
        filter_tpe_above_zero = df["TPE_EUR"] > 0
        filter_no_nas = df["TPE_EUR"].notna()

        df = df.loc[(filter_tpe_above_zero) & (filter_no_nas)].copy()

        # Add Rating Band Column
        df['BUYER_RATING_BAND'] = self.get_buyer_rating_band(df=df)

        df['BUYER_ID'] = df['BUYER_ID'].astype(str)

        # Add column to help calculating Weighted Average Rating
        df['WA_RATING_HELPER'] = df['TPE_EUR'] * df['BUYER_IBR_RATING']
        
        # Decided to include this column to check if Buyer has Rating or not
        # Groupins with many buyers without rating were being benefited due to
        # the TPE * RATING being null, but the TPE still being summed.
        df['TPE_EUR_FOR_WA_RATING'] = np.where(
            df['BUYER_IBR_RATING'].notna(),
            df['TPE_EUR'],
            np.nan
        )

        df['BROAD_RATING_BAND'] = self.get_broad_rating_band(df=df)

        self._data = df
    
    @property
    def date(self) -> pd.Timestamp:
        # Using to_datetime to return Timestamp type
        # Use unique() to return single value array, and [0] to return
        # the value, not the array
        ts = pd.to_datetime(self.data['REFRESH_DATE'].unique()[0])

        if ts.day <= 3:
            # Subtracting MonthEnd to bring dates that are in the beginning
            # of the month to the previous Month End.
            # e.g.: Refresh Date is 01/06/2021, ts should be 31/05/2021
            ts = ts - pd.tseries.offsets.MonthEnd(1)

        return ts
    
    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%b")
    
    def import_data(self, filepath: str) -> None:
        try:
            if filepath.endswith('.xlsx'):
                self.data = pd.read_excel(filepath)
            elif filepath.endswith('.txt'):
                self.data = pd.read_csv(filepath, sep='\t')
            else:
                err_msg = "Data needs to be in txt file or Excel (.xlsx)."
                err_msg += f"{filepath} was passed."
                print(err_msg)
                return None
        except ParserError:
            print(f'Error trying to read {filepath}.')
            print("Trying to read with Python's open() function.")
            with open(filepath, 'r', encoding='utf-8') as fh:
                # Read each line from file, and split using '\t'
                lines = [line.split('\t') for line in fh]

            # [1:] so we exclude Headers, and [0] so we use headers
            self.data = pd.DataFrame(lines[1:], columns=lines[0])
    
    @staticmethod
    def get_buyer_rating_band(df: pd.DataFrame) -> pd.Series:
        """Return Buyer Rating Band of buyer ratings.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing Buyer Ratings column named "BUYER_IBR_RATING".

        Returns
        -------
        pd.Series
            Buyer Rating Band series.
        """        

        df = df.copy()

        condlist = [(df["BUYER_IBR_RATING"].isna()),
                    (df["BUYER_IBR_RATING"] <= 10),
                    (df["BUYER_IBR_RATING"] <= 20),
                    (df["BUYER_IBR_RATING"] <= 30),
                    (df["BUYER_IBR_RATING"] <= 40),
                    (df["BUYER_IBR_RATING"] <= 50),
                    (df["BUYER_IBR_RATING"] <= 60),
                    (df["BUYER_IBR_RATING"] <= 70),
                    (df["BUYER_IBR_RATING"] <= 80),
                    (df["BUYER_IBR_RATING"] <= 90),
                    (df["BUYER_IBR_RATING"] <= 100)]

        choicelist = ["No Rating",
                      "1-10",
                      "11-20",
                      "21-30",
                      "31-40",
                      "41-50",
                      "51-60",
                      "61-70",
                      "71-80",
                      "81-90",
                      "91-100"]

        # if condition is fulfiled, value comes from the choicelist
        return np.select(condlist, choicelist, "No Rating")

    def extract_date_from_filename(self) -> pd.Timestamp:
        """Return Timestamp from filename

        Notes
        -----
        - ?P<year> is a named group
        - \d represents a digit from 0 to 9
        - {m, n} means there must be at least m repetitions, and at most n
        - The parenthesis "(" and ")" represent a grouping pattern

        Returns
        -------
        pd.Timestamp
            Timestamp from the filename used to load the data into class.
        """        

        # Define Regular Expression pattern to be found
        # ?P<...> is a key identifier when returning a dict with groupdict
        pattern = re.compile(
            r'_((?P<year>\d{4,4})(?P<month>\d{2,2})(?P<day>\d{2,2}))_'
        )
        
        # find the year, month and day
        match = re.search(pattern, self.filepath)
        
        # return dictionary with year, month and day
        match_dict = match.groupdict()

        # create a string to be used in pandas.to_datetime.
        # Year in the yyyy/mm/dd format to avoid confusion
        date_str = f"{match_dict['year']}-{match_dict['month']}-{match_dict['day']}"
        
        return pd.to_datetime(date_str)
    
    def get_data(
        self,
        filter_dict: Dict[str, Union[str, List[str]]] = None,
        remove_na_cols: List[str] = None,
        groupby: List[str] = None
    ) -> pd.DataFrame:
        """Return dataset, applying any parameters to data if passed.

        Parameters
        ----------
        filter_dict : Dict[str, Union[str, List[str]]], optional
            dict with key being column name and value being:
            - `str`:
                - if str, then code will evaluate string.
                E.g.: `'> 70'` will evaluate inside loc to `df[key] > 70`
            - `list`:
                - if list, then filter will use isin() function from pandas.
                E.g.: `[]` will evaluate to `df[key].isin(list)`
            , by default None

        remove_na_cols : `List[str]`, optional
            If not none, then will remove any rows that 
            contain NaN in columns from remove_na_cols argument,
            by default `None`
        groupby : List[str], optional
            If to aggregate the Data using certain columns, by default `None`.
            - This will return a dataset with TPE and 
              Weighted Average RATING

        Returns
        -------
        pd.DataFrame
            Return dataset with corresponding filters and groupby's
        """    
        df = self.data

        if filter_dict is not None:
            for col_name, cond in filter_dict.items():
                if isinstance(cond, str):
                    df = df.loc[eval(f"df['{col_name}'] {cond}")].copy()
                else:
                    df = df.loc[df[col_name].isin(cond)].copy()
        
        if remove_na_cols is not None:
            df = df.dropna(axis='rows', subset=remove_na_cols).copy()
        
        if groupby is not None:
            # If groupby is not note, then we aggregate data
            df = df.groupby(by=groupby, as_index=False).sum()

            # Normalize Rating using TPE_EUR. This can be done in any grouping
            df['RATING'] = df['WA_RATING_HELPER'] / df['TPE_EUR_FOR_WA_RATING']

            # Drop columns because they will have a sum, and are not
            # valid data
            cols_to_drop = ['BUYER_IBR_RATING',
                            'SYMPHONY_POLICY_ID',
                            'CREDIT_LIMIT_ID',
                            'CLD_AMT_ORIGINAL_CURR',
                            'FIXED_EUR_EXC_RATE',
                            'VAR_EUR_EXC_RATE',
                            'BUYER_NACE_CODE',
                            'BUYER_COUNTRY_STAR_RATING',
                            'PARENT_ID',
                            'PARENT_NACE_CODE',
                            'PARENT_COUNTRY_STAR_RATING',
                            'CUSTOMER_ID',
                            'COMMITMENT_CATEGORY',
                            'WA_RATING_HELPER']
            
            for col in cols_to_drop:
                try:
                    df.drop(columns=[col], inplace=True)
                except KeyError:
                    pass

        return df
    
    def get_weighted_average_rating(
        self,
        by: Union[None, list] = None
    ) -> Union[float, pd.DataFrame]:
        """Return the weighted average for entire data, or by columns.

        Parameters
        ----------
        by : Union[None, list]
            list of columns to aggregate the data.
            If none, then no aggregation is made
        
        Notes
        -----
        Aggregation performed on data is always sum.

        Returns
        -------
        Union[float, pd.DataFrame]
            If by is none, then return float.
            If by is list, then 
        """        
        df = self.get_data(remove_na_cols=['BUYER_IBR_RATING'],
                           groupby=by)

        if by is None:
            return (df['WA_RATING_HELPER'].sum()
                    / df['TPE_EUR_FOR_WA_RATING'].sum())
        else:
            return df

    def count_unique(
        self,
        count_col: str,
        filter_dict: Dict[str, Union[str, List[str]]] = None,
        remove_na_cols: List[str] = None
    ) -> int:
        """Return count of buyers based on criteria.

        Parameters
        ----------
        count_col : str
            column to be counted after filtering DataFrame using criteria.
        criteria_col : str, optional
            column to be filtered using conditon, by default None.
        criteria : str, optional
            The DataFrame stored in class instance will be filtered using
            df[criteria_col] + criteria, as when filtering a pd.DataFrame
            object, by default None.

        Returns
        -------
        int
            Count of buyers meeting criteria.
        
        Raises
        ------
        ValueError
            Raise error if criteria_col and criteria don't have the same type.
        """
        df = self.get_data(filter_dict=filter_dict,
                           remove_na_cols=remove_na_cols)

        return len(df[count_col].unique())
    
    def get_tpe(
        self,
        by: List[str] = None,
        filter_dict: Dict[str, Union[str, List[str]]] = None,
        remove_na_cols: List[str] = None
    ) -> int:
        """Return sum of TPE based on criteria.

        Parameters
        ----------
        sum_col : str
            Column to be summed after filtering DataFrame using criteria.
        criteria_col : str, optional
            column to be filtered using conditon, by default None.
        criteria : str, optional
            The DataFrame stored in class instance will be filtered using
            df[criteria_col] + criteria, as when filtering a pd.DataFrame
            object, by default None.
        
        Returns
        -------
        int
            Sum of TPE where rows meet the criteria.
        """        
        df = self.get_data(filter_dict=filter_dict,
                           remove_na_cols=remove_na_cols,
                           groupby=by)
        
        if by is None:
            return df['TPE_EUR'].sum()
        else:
            return df

    def get_broad_rating_band(self, df: pd.DataFrame) -> pd.Series:
        rating_col = 'BUYER_IBR_RATING'

        condlist = [df[rating_col].isna(),
                    df[rating_col] < 50,
                    df[rating_col] < 71,]

        choicelist = ['Not Rated',
                      'Below 50',
                      'Between 50 and 70']
        
        return np.select(condlist, choicelist, 'Above 70')

    # def group_data(self,
    #                by: List[str],
    #                func: str = 'sum',
    #                value_col: str = None) -> pd.DataFrame:
    #     col_list = []
    #     # Dealing with by argument
    #     if isinstance(by, str):
    #         # if "by" argument is only a string
    #         col_list.append(by)
    #     elif isinstance(by, list):
    #         # Append by as list to the empty list
    #         col_list += by

    #     # Dealing with value_col argument
    #     if isinstance(value_col, str):
    #         # if value_col is string, then append to col_list
    #         col_list.append(value_col)
    #     elif value_col is None:
    #         # I don't see how another column can be used as value_col
    #         # other than TPE_EUR, but I left the option to be used here.
    #         col_list.append('TPE_EUR')

    #     df = self.data[col_list].copy()

    #     return eval(f"df.groupby(by={by}).{func}()")
    
    def get_broad_rating_band_sum(
        self,
        groupby: Union[None, List[str]] = None,
        pct_of_total: bool = False
    ) -> pd.DataFrame:

        df = self.get_data(remove_na_cols=['BUYER_IBR_RATING'],
                           groupby=groupby)

        if groupby is None:
            rating_col = 'BUYER_IBR_RATING'
        else:
            rating_col = 'RATING'
        
        condlist = [df[rating_col] < 50,
                    df[rating_col] < 71,]

        choicelist = ['Below 50',
                      'Between 50 and 70']

        # Divide buyers into 
        df['BROAD_RATING_BAND'] = np.select(condlist, choicelist, 'Above 70')

        df_grp = df.groupby('BROAD_RATING_BAND').sum()['TPE_EUR']

        if pct_of_total:
            # In case we want % of total, instead of actual figures
            df_grp /= df['TPE_EUR'].sum()
        
        # Transpose to make plotting easier
        df_grp = pd.DataFrame(df_grp).T

        # Also, setting index to date to make plot easier
        df_grp.index = [self.date]
        
        return df_grp
    



class CompareNamedExposure:
    COLS_FOR_COMPARISON = ["BUYER_ID",
                           "KEY",
                           "BUYER_STATUS",
                           "BUYER_IBR_RATING",
                           "BUYER_RATING_BAND",
                           "TPE_EUR"]

    FINAL_OUTPUT_COLS = ["KEY",
                         "CEDANT_NAME",
                         "SHORT_NAME",
                         "BUYER_ID",
                         "BUYER_NAME",
                         "BUYER_VAT_NUMBER",
                         "BUYER_STATUS_new",
                         "BUYER_SRM_FLAG",
                         "BUYER_SECTOR_TYPE",  # priv or publ
                         "BUYER_INDUSTRY",
                         "BUYER_TRADE_GROUP",
                         "BUYER_TRADE_SECTOR",
                         "BUYER_IBR_RATING_new",
                         "BUYER_IBR_RATING_old",
                         "BUYER_RATING_NOTE",
                         "BUYER_RATING_DELTA",
                         "BUYER_RATING_BAND_new",
                         "BUYER_RATING_BAND_old",
                         "BUYER_RATING_DELTA_SCORE",
                         "TPE_EUR_new",
                         "TPE_EUR_old",
                         "TPE_EUR_delta",
                         "PARENT_ID",
                         "PARENT_NAME",
                         "PARENT_IBR_RATING"]

    BUYER_INFO_COLS = ["REFRESH_DATE",
                       "BUYER_ID",
                       "BUYER_NAME",
                       "BUYER_VAT_NUMBER",
                       "BUYER_SRM_FLAG",
                       "BUYER_SECTOR_TYPE",
                       "BUYER_INDUSTRY",
                       "BUYER_TRADE_GROUP",
                       "BUYER_TRADE_SECTOR",
                       "PARENT_ID",
                       "PARENT_NAME",
                       "PARENT_IBR_RATING", ]

    CEDANT_INFO_COLS = ["REFRESH_DATE",
                        "KEY",
                        "SHORT_NAME",
                        "CEDANT_NAME"]

    def __init__(self,
                 old_named_data: NamedExposureData,
                 new_named_data: NamedExposureData) -> None:
        if old_named_data.date >= new_named_data.date:
            # Check to make sure old data is older than new data
            # If not, then raise ValueError
            err_msg = "Old data needs to be older than new data. "
            err_msg += f"{old_named_data.date} and {new_named_data.date} "
            err_msg += "were passed respectively."
            raise ValueError(err_msg)

        self.old_named_data = old_named_data
        self.new_named_data = new_named_data

        self._buyer_info = None
        self._cedant_info = None

        self._compared_data = None

        self._update_buyer_and_cedant_info()

        self._generate_comparison()

    def __repr__(self):
        str_format = '%y-%b'
        old_date_str = pd.to_datetime(self.old_named_data.date).strftime(
            str_format)
        new_date_str = pd.to_datetime(self.new_named_data.date).strftime(
            str_format)
        return f"<CompareNamedExposure Class-{old_date_str} x {new_date_str}>"

    @property
    def buyer_info(self):
        return self._buyer_info

    @buyer_info.setter
    def buyer_info(self, df: pd.DataFrame) -> None:
        df = pd.concat([self._buyer_info, df])

        # Sort by REFRESH_DATE to keep latest date at the end to be used
        # when dropping duplicates
        df.sort_values(by='REFRESH_DATE', axis=0, inplace=True)

        # keep='last' to keep latest info
        df.drop_duplicates(subset="BUYER_ID", keep='last', inplace=True)

        self._buyer_info = df

    @property
    def cedant_info(self):
        return self._cedant_info

    @cedant_info.setter
    def cedant_info(self, df: pd.DataFrame) -> None:
        df = pd.concat([self._cedant_info, df])

        # Sort by REFRESH_DATE to keep latest date at the end to be used
        # when dropping duplicates
        df.sort_values(by='REFRESH_DATE', axis=0, inplace=True)

        # keep='last' to keep latest info
        df.drop_duplicates(subset="KEY", keep='last', inplace=True)

        self._cedant_info = df
    
    @property
    def compared_data(self) -> pd.DataFrame:
        try:
            return self._compared_data[self.FINAL_OUTPUT_COLS]
        except KeyError:
            return self._compared_data
    
    @compared_data.setter
    def compared_data(self, data: Union[None, pd.DataFrame]) -> pd.DataFrame:
        self._compared_data = data
        return None

    def _update_buyer_and_cedant_info(self) -> None:
        for data in [self.new_named_data.data, self.old_named_data.data]:
            self.buyer_info = data[self.BUYER_INFO_COLS]
            self.cedant_info = data[self.CEDANT_INFO_COLS]

        return None

    def _merge_comparison_data(self) -> None:
        # Assign variables to old and new dataset
        # Since it's NamedExposureData Class, we need to use .data to access
        # the underlying data
        df_old = self.old_named_data.data[self.COLS_FOR_COMPARISON]
        df_new = self.new_named_data.data[self.COLS_FOR_COMPARISON]

        df_merged = df_new.merge(right=df_old,
                                 how='outer',
                                 on=["BUYER_ID", "KEY"],
                                 suffixes=('_new', '_old'))

        # Assign df_merged straight to self.compared_data
        self.compared_data = df_merged

    def _fix_tpe_columns(self) -> None:
        df = self.compared_data.copy()

        # Set TPE NaN to zero
        df["TPE_EUR_new"].fillna(value=0, inplace=True)
        df["TPE_EUR_old"].fillna(value=0, inplace=True)

        # calculating TPE delta
        df["TPE_EUR_delta"] = df["TPE_EUR_new"] - df["TPE_EUR_old"]

        self.compared_data = df

        return None

    def _include_IBR_status_col(self) -> None:
        df = self.compared_data.copy()

        # Set up filters
        filter_TPE_old_isna = df['TPE_EUR_old'] == 0
        filter_TPE_new_isna = df['TPE_EUR_new'] == 0
        filter_B_IBR_new_UNRATED = df["BUYER_IBR_RATING_new"].isna()
        filter_B_IBR_old_UNRATED = df["BUYER_IBR_RATING_old"].isna()

        # pd.to_numeric needed to convert values to numbers and text to NaN
        filter_same_IBR = df["BUYER_IBR_RATING_new"] == df["BUYER_IBR_RATING_old"]
        filter_worse_IBR = df["BUYER_IBR_RATING_new"] > df["BUYER_IBR_RATING_old"]
        filter_improve_IBR = df["BUYER_IBR_RATING_new"] < df["BUYER_IBR_RATING_old"]

        # conditions:
        condlist = [
            filter_TPE_new_isna,
            filter_TPE_old_isna,
            (filter_B_IBR_new_UNRATED) & (filter_B_IBR_old_UNRATED),
            (filter_B_IBR_new_UNRATED) & (~filter_B_IBR_old_UNRATED),
            (~filter_B_IBR_new_UNRATED) & filter_B_IBR_old_UNRATED,
            (~filter_B_IBR_new_UNRATED) & (
                ~filter_B_IBR_old_UNRATED) & filter_same_IBR,
            (~filter_B_IBR_new_UNRATED) & (
                ~filter_B_IBR_old_UNRATED) & filter_worse_IBR,
            (~filter_B_IBR_new_UNRATED) & (
                ~filter_B_IBR_old_UNRATED) & filter_improve_IBR,
        ]
        # choicelist
        choicelist = [
            "TPE no longer recorded",
            "New TPE",
            "No Rating",
            "Lost rating",
            "Newly rated",
            "No change in rating",
            "Rating deteriorated",
            "Rating improved"
        ]
        # if condition is fulfiled, value comes from the choicelist -  first condition matches first choice etc.
        df["BUYER_RATING_NOTE"] = np.select(condlist, choicelist)

        self.compared_data = df

        return None

    def _include_IBR_delta_col(self) -> None:

        df = self.compared_data.copy()

        # Create BUYER_DELTA col and assigning values of (new IBR - old IBR)
        #df.loc[df["BUYER_RATING_STATUS"].isin(["Rating deteriorated", "Rating improved"]), "BUYER_DELTA"] = df["BUYER_IBR_RATING_new"] - df["BUYER_IBR_RATING_old"]
        df["BUYER_RATING_DELTA"] = df["BUYER_IBR_RATING_new"] - \
            df["BUYER_IBR_RATING_old"]

        # replacing NaN values with an empty string
        # df["BUYER_DELTA"].fillna('', inplace=True)

        self.compared_data = df

    def _include_buyer_info_cols(self) -> pd.DataFrame:
        df = self.compared_data.copy()

        buyer_info = self.buyer_info.copy()

        # We're not using REFRESH_DATE col, so we'll drop it
        buyer_info.drop(columns=['REFRESH_DATE'], inplace=True)

        self.compared_data = df.merge(right=buyer_info,
                                      how='left',
                                      on='BUYER_ID')

        return None

    def _include_cedant_info_cols(self) -> None:
        df = self.compared_data.copy()

        cedant_info = self.cedant_info.copy()

        # We're not using REFRESH_DATE col, so we'll drop it
        cedant_info.drop(columns=['REFRESH_DATE'], inplace=True)

        self.compared_data = df.merge(right=cedant_info,
                                      how='left',
                                      on='KEY')

        return None

    def _include_rating_score_col(self) -> None:
        df = self.compared_data.copy()

        # Include RATING_TPE_SCORE column
        # This takes into consideration the TPE amount over the total TP
        # and the rating delta, as to have the most significant movements
        # on top
        df['BUYER_RATING_DELTA_SCORE'] = (
            df['BUYER_RATING_DELTA']
            * df['TPE_EUR_new'] / df['TPE_EUR_new'].sum()
        )

        self.compared_data = df

        return None

    def _check_missing_columns(self) -> None:
        df_comp = self.compared_data
        if len(df_comp.columns) != len(self.FINAL_OUTPUT_COLS):
            # This warns user if there's a mismatch between the columns in
            # the final df, and the base columns we originally thought about
            missing_cols = [
                i for i in df_comp.columns if i not in self.FINAL_OUTPUT_COLS
            ]
            warn_msg = "The following columns are not in the final output:\n"
            warn_msg += f"{missing_cols}"
            logging.warning(warn_msg)
    
    def _normalize_buyer_status_new(self) -> None:
        df = self.compared_data.copy()

        # Create Series to map Buyer Status values
        #=========================================

        # Create DataFrame with all unique buyer status, only from new
        # dropna how='any' means it will drop any row that contains NaN
        df_buyer_status = df[[
            'BUYER_ID',
            'BUYER_STATUS_new'
        ]].dropna(axis=0, how='any').drop_duplicates().copy()

        # Set BUYER_ID as index, to help with the mapping later
        df_buyer_status.set_index('BUYER_ID', inplace=True)

        # Convert DataFrame to Series
        s_buyer_status = df_buyer_status['BUYER_STATUS_new']

        # Apply mapping to BUYER_ID column, as to have only most recent
        # buyer statuses
        df['BUYER_STATUS_new'] = df['BUYER_ID'].map(s_buyer_status)

        self.compared_data = df

        print('Normalizing BUYER_STATUS_new column.')

        return None

    def _generate_comparison(self) -> pd.DataFrame:
        self._merge_comparison_data()
        self._fix_tpe_columns()
        self._include_IBR_status_col()
        self._include_IBR_delta_col()
        self._include_buyer_info_cols()
        self._include_cedant_info_cols()
        self._include_rating_score_col()
        self._normalize_buyer_status_new()
        return None

    def get_buyer_rating_movement(self):
        old_wa_rat = self.old_named_data.get_weighted_average_buyer_rating()
        new_wa_rat = self.old_named_data.get_weighted_average_buyer_rating()

        var = (new_wa_rat - old_wa_rat) / old_wa_rat

        msg = "Weighted Average Buyer Rating is "
        msg += f"{new_wa_rat:.1f}, from {old_wa_rat:.1f} "
        msg += f"({var:+.2%})."
        return msg

    # def get_summary_buyers_rated(self, criteria: str) -> str:
    #     movement_verb = 'above' if '>' in criteria else 'below'
    #     movement_verb += ' or equal to' if '=' in criteria else ''

    #     # This returns the number inside criteria
    #     ref_number = re.search(r'\d+', criteria).group()

    #     # Only need newest dataset in comparison_data
    #     named_data_raw = self.new_named_data

    #     count_unique_buyers = named_data_raw.count_unique(count_col='BUYER_ID')
    #     total_tpe = named_data_raw.sum_tpe(sum_col='TPE_EUR')

    #     count_buyers_above = named_data_raw.count_unique(
    #         count_col='BUYER_ID',
    #         criteria_col='BUYER_IBR_RATING',
    #         criteria=criteria)

    #     sum_tpe_buyers_above = named_data_raw.sum_tpe(
    #         sum_col='TPE_EUR',
    #         criteria_col='BUYER_IBR_RATING',
    #         criteria=criteria)

    #     pct_above_unique_buyers = (
    #         count_buyers_above / count_unique_buyers)

    #     pct_tpe_above = (
    #         sum_tpe_buyers_above / total_tpe)

    #     msg = f"Count of unique buyers rated {movement_verb} {ref_number}:\n"
    #     msg += f"{count_buyers_above} ({pct_above_unique_buyers:.1%} of total)\n\n"
    #     msg += f"Sum of TPE of buyers rated {movement_verb} {ref_number}:\n"
    #     msg += f"EUR {sum_tpe_buyers_above:,.0f} ({pct_tpe_above:.1%} of total)\n\n"

    #     return msg

    def get_top_buyer_rating_movements_by(self, n: int = None) -> pd.DataFrame:
        score_col = 'BUYER_RATING_DELTA_SCORE'
        
        df = self.compared_data

        # since the function returns top buyers, we use BUYER_ID column only
        df_grp = df.groupby(by='BUYER_ID', as_index=False).sum()

        # Order from descending. This might change if we want to return the
        # largest improvements, not the worst deteriorations only.
        df_grp.sort_values(by=score_col,
                           ascending=False,
                           inplace=True)
        
        if isinstance(n, int):
            return df_grp.nlargest(n=n, columns=score_col)
        
        return df_grp

    def get_report_name(self, file_extension: str, timestamp: bool = False):
        old_date = self.old_named_data.date.strftime("%y-%b")
        new_date = self.new_named_data.date.strftime("%y-%b")

        # Timestamp
        if timestamp:
            ts = pd.to_datetime('now').strftime("%Y%m%d-%Hh%Mm")
            ts_str = f'-produced on {ts}'
        else:
            # If timestamp is false, then we set ts_str to empty string
            ts_str = ''

        return f'Buyer_Health_Report-{old_date} X {new_date}{ts_str}.{file_extension}'

    def to_excel(self, open_file: bool = True) -> None:
        # Create variable with filename
        filename = self.get_report_name(file_extension='xlsx', timestamp=True)

        writer = pd.ExcelWriter(filename,
                                # don't change engine
                                engine='xlsxwriter',
                                # don't change date_format
                                date_format='dd/mm/yyyy',
                                # don't change datetime_format
                                datetime_format='dd/mm/yyyy')

        sheetname = 'Details'

        df_to_excel = self.compared_data.copy()

        # This is only a request for the Excel output
        df_to_excel['BUYER_IBR_RATING_new'].fillna("No Rating", inplace=True)
        df_to_excel['BUYER_IBR_RATING_old'].fillna("No Rating", inplace=True)

        # Export to Excel
        df_to_excel.to_excel(
            writer,  # this is the variable defined above
            index=False,  # this is if we want to include the DataFrame index
            header=True,  # this is if we want to include the DataFrame header
            sheet_name=sheetname,  # this is the name of the sheet being created
            startrow=0,  # top row that the DataFrame will start. Row 1 is index 0
            startcol=0)  # most-left column that the DataFrame will start. col A is index 0

        # Get the xlsxwriter workbook and worksheet objects.
        wb = writer.book
        ws = writer.sheets[sheetname]

        # Get the dimensions of the dataframe.
        (max_row, max_col) = df_to_excel.shape

        # Create a list of column headers, to use in add_table().
        column_settings = [{'header': column}
                           for column in df_to_excel.columns]

        # Add the Excel table structure. Pandas will add the data.
        ws.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

        # TODO: Redo part below to apply correct formatting
        num_format = wb.add_format({'num_format': '#,##0'})

        ws.set_column("A:A", 10)  # KEY
        ws.set_column("B:B", 33)  # CEDANT_NAME
        ws.set_column("C:C", 28)  # SHORT_NAME
        ws.set_column("D:D", 11)  # BUYER_ID
        ws.set_column("E:E", 32)  # BUYER_NAME
        ws.set_column("F:F", 21)  # BUYER_VAT_NUMBER
        ws.set_column("G:G", 20)  # BUYER_STATUS_new
        ws.set_column("H:H", 17)  # BUYER_SRM_FLAG
        ws.set_column("I:L", 21)
        ws.set_column("M:N", 25)
        ws.set_column("O:O", 23)  # BUYER_RATING_NOTE
        ws.set_column("P:P", 23)  # BUYER_RATING_DELTA
        ws.set_column("Q:R", 26)  # BUYER_RATING_BAND (new and old)
        ws.set_column("S:S", 30)  # BUYER_RATING_DELTA_SCORE
        ws.set_column("T:V", 16, num_format)  # TPE columns
        ws.set_column("W:W", 13)  # PARENT_ID
        ws.set_column("X:Y", 21)  # PARENT name and rating

        # Add format - Light red fill with dark red text.
        highlight_red_format = wb.add_format({
            'bg_color': '#FFC7CE',
            'font_color': '#9C0006'})

        #Extracting the number of rows in the dataframe
        rows = df_to_excel.shape[0]

        # Find BUYER_STATUS_new column position
        # Here I leave as zero-based index because
        # I start from column A in buyer_stat_col_letter
        buyer_stat_col_pos = df_to_excel.columns.get_loc('BUYER_STATUS_new')
        buyer_stat_col_letter = chr(ord('A') + buyer_stat_col_pos)
        buyer_stat_excel_rng = f'{buyer_stat_col_letter}2:{buyer_stat_col_letter}{rows}'

        # Apply conditional formatting to a range
        # check https://xlsxwriter.readthedocs.io/example_conditional_format.html#ex-cond-format

        # f before ' means that the string accepts variables inside the curly brackets
        # columns C and D store statuses in Excel
        for status in FORBIDDEN_STATUSES:
            ws.conditional_format(
                buyer_stat_excel_rng,
                {
                    'type': 'cell',
                    'criteria': '==',
                    'value': f'"{status}"',
                    'format': highlight_red_format
                }
            )

        writer.save()

        if open_file:
            os.startfile(filename)

    def to_html(self, open_file: bool = True) -> None:
        from jinja2 import Environment, FileSystemLoader
        import webbrowser

        templates_dir = os.path.join('templates')

        env = Environment(loader=FileSystemLoader(templates_dir))

        template = env.get_template('template_buyer_mon_rep.html')

        filename = self.get_report_name(file_extension='html', timestamp=True)

        # # Dict with all elements to complete HTML report
        # elem = self._get_html_elements()

        with open(filename, 'w') as fh:
            fh.write(template.render(
                portfolio_rating='123'
            ))

        print(f"HTML report created at {filename}.")
        if open_file:
            webbrowser.open(filename, new=2)


class NamedExposure:
    def __init__(self) -> None:
        self._named_data = list()

    @property
    def named_data(self) -> List[NamedExposureData]:
        "Return list of all named data, sorted from oldest to newest."
        return self._named_data

    @named_data.setter
    def named_data(self, named_data_obj) -> None:
        # append excel data to _named_data array
        self._named_data.append(named_data_obj)

        # sort array as to have oldest to newest
        self._named_data.sort()

    @property
    def new_date(self):
        return self.named_data[-1].date
    
    @property
    def old_date(self):
        return self.named_data[-2].date

    def import_named_data(self, filename: str):
        named_exp_data = NamedExposureData(filepath=filename)

        # This will append the excel data to named_data, including
        # some meta data
        self.named_data = named_exp_data

        date_str = named_exp_data.date.strftime("%B/%Y")
        print(f"Data added for {date_str}.")
    
    def get_available_dates(self, strftime: bool = False) -> list:
        available_dates = [data.date for data in self.named_data]
        
        if strftime:
            # If user wants more readable dates
            available_dates =  [
                date.strftime('%d/%m/%Y') for date in available_dates
            ]
        
        return available_dates

    def is_date_in_available_data(self, date: str) -> bool:
        available_dates = self.get_available_dates()
        date_ts = pd.to_datetime(date, dayfirst=True)

        if date_ts not in available_dates:
            dates_in_model = self.get_available_dates(strftime=True)
            err_msg = f"Data for date {date} not in model.\n"
            err_msg += f"Dates available: {dates_in_model}"
            raise ValueError(err_msg)
        
        return True

    def get_data_for_date(
        self,
        date: Union[None, str, pd.Timestamp] = None
    ) -> NamedExposureData:
        """Get data for specific date, or most current date.

        Parameters
        ----------
        date : Union[None, str, pd.Timestamp]
            If None, then returns most recent date,
            Else, it returns date corresponding to date argument passed.

        Returns
        -------
        NamedExposureData
            Return class holding data for corresponding date.
        """
        if date is None:
            # If None is passed, then return most recent
            return self.named_data[-1]

        # Check if date is valid
        self.is_date_in_available_data(date)
        date_ts = pd.to_datetime(date, dayfirst=True)
        for data in self.named_data:
            if date_ts == data.date:
                # if date is found, then return data class
                return data
    
    def get_movements(self,
                      groupby: str,
                      old_date: Union[str, None] = None,
                      new_date: Union[str, None] = None) -> pd.DataFrame:
        if old_date is None:
            # -2 corresponds to second most recent dataset
            old_data = self.get_data_for_date(self.named_data[-2].date)
        else:
            old_data = self.get_data_for_date(old_date)
        
        if new_date is None:
            new_data = self.get_data_for_date()
        else:
            new_data = self.get_data_for_date(new_date)
        
        df_old = old_data.get_data(groupby=[groupby])
        df_new = new_data.get_data(groupby=[groupby])

        merge_cols = [groupby, 'TPE_EUR', 'RATING']

        suffixes = (f'-{old_data.date_str}', f'-{new_data.date_str}')

        df_mrg = df_old[merge_cols].merge(df_new[merge_cols],
                                          on=groupby,
                                          suffixes=suffixes)

        df_mrg['TPE-Diff'] = (df_mrg[f'TPE_EUR{suffixes[1]}']
                              - df_mrg[f'TPE_EUR{suffixes[0]}'])
        
        df_mrg['TPE-Diff %'] = (df_mrg['TPE-Diff']
                                / df_mrg[f'TPE_EUR{suffixes[0]}'])

        df_mrg['Rating-Diff'] = (df_mrg[f'RATING{suffixes[1]}']
                                 - df_mrg[f'RATING{suffixes[0]}'])
        df_mrg['Rating-Diff %'] = (df_mrg['Rating-Diff']
                                   / df_mrg[f'RATING{suffixes[0]}'])

        return df_mrg
        
    def compare_dates(
            self,
            old_date: Union[str, None] = None,
            new_date: Union[str, None] = None,
            to_excel: bool = False,
            open_file: bool = False) -> CompareNamedExposure:
        """Return DataFrame with comparison between selected dates.

        Parameters
        ----------
        old_date : Union[str, None], optional
            str representing a date. Has to be older than new_date.
            If None, then uses second most recent date added.
            by default None.
        new_date : Union[str, None], optional
            str representing a date. Has to be newer than old_date.
            If None, then uses second most recent date added.
            by default None.

        Returns
        -------
        CompareNamedExposure
            Class responsible for merging and processing data for comparison.
        """
        if old_date is None:
            print(f"Comparison will use {self.old_date} as old data.")
            old_data = self.get_data_for_date(self.old_date)
        else:
            self.old_date = old_date
            old_data = self.get_data_for_date(old_date)

        if new_date is None:
            print(f"Comparison will use {self.new_date} as new data.")
            new_data = self.get_data_for_date(self.new_date)
        else:
            self.new_date = new_date
            new_data = self.get_data_for_date(new_date)
        
        compared_name = CompareNamedExposure(old_named_data=old_data,
                                             new_named_data=new_data)
        
        if to_excel is True:
            compared_name.to_excel(open_file=open_file)
        else:
            return compared_name

    def get_report_name(self, file_extension: str, timestamp: bool = False):
        old_date = self.old_date.strftime("%y-%b")
        new_date = self.new_date.strftime("%y-%b")

        # Timestamp
        if timestamp:
            ts = pd.to_datetime('now').strftime("%Y%m%d-%Hh%Mm")
            ts_str = f'-produced on {ts}'
        else:
            # If timestamp is false, then we set ts_str to empty string
            # since we're not timestamping the export file
            ts_str = ''

        return f'Buyer_Health_Report-{old_date} X {new_date}{ts_str}.{file_extension}'

    def get_portfolio_rating_movement(self) -> pd.DataFrame:

        df = pd.DataFrame()
        
        for data in self.named_data:
            # iterate through each loaded dataset, and append portfolio
            # rating to DataFrame
            df = pd.concat(
                [df,
                pd.DataFrame(
                    data=[data.get_weighted_average_rating()],
                    index=[data.date],
                    columns=['Rating']
                )]
            )
        
        # Calculate Month over Month percentage change
        df['MoM %'] = df['Rating'].pct_change()

        oldest_index = df.index[0]

        # Calculate difference from oldest data point.
        df[f'% from {oldest_index.year}-{oldest_index.month}'] = (
            df['Rating'] - df.loc[oldest_index, 'Rating']
        ) / df.loc[oldest_index, 'Rating']
        df

        
        return df