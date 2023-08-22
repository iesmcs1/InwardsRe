import pandas as pd
from ..register.register import Register
import logging


class ReinsuranceGrouping:
    def __init__(
        self,
        ceded_reg_path: str,
        master_file_path: str
    ) -> None:

        self.ceded_reg = self._load_ceded_register(path=ceded_reg_path)
        self.master_file = self._load_master_file(path=master_file_path)

        # A flag that is set to False if there are
        # any Treaties not in Master File
        self._ok_to_proceed = False

    def _load_master_file(self, path: str):
        # The first row is always blank
        df = pd.read_excel(path, skiprows=1)

        # The first column is always blank.
        return df.iloc[:, 1:]

    def _load_ceded_register(self, path: str):
        # Only need to load the register and use the Register as is.
        # No further manipulation needed
        return Register(filepath=path, register_type='ceded').register.copy()

    def find_retros_not_in_master_file(self, oldest_uwy: int):
        df_ceded = self.ceded_reg
        df_master_file = self.master_file

        df_raw = df_ceded.merge(df_master_file,
                                how='left',
                                left_on='Ceded Ref',
                                right_on='RT Ref')

        # We only need 4 columns for this task
        df = df_raw[['Ceded Ref', 'RT Ref', 'Short', 'UW Yr']].copy()

        # Makes sure we keep only Ceded References that do not have 
        # a corresponding value in the Master File RT Ref column 
        filter_only_nas = df['RT Ref'].isna()
        
        # These keywords are not relevant for this task
        remove_keywords = 'A.RE|Actuarial Reserve|CEDED BULK RESVERS|ATRADIUS ACTUARIAL R'
        filter_short_col = df['Short'].str.contains(remove_keywords)

        # Older UWY are not important for this task, so we remove them
        filter_uwy = df['UW Yr'] > oldest_uwy

        df_filtered = df.loc[(filter_only_nas)
                             & (~filter_short_col)
                             & (filter_uwy)]
        if df_filtered.shape[0] > 0:
            info_msg = "### The Treaties below are not in the Master File. ##\n"
            info_msg += "\nOutward Re needs to be informed about these, so that\n"
            info_msg += "they can update their records and provide\n"
            info_msg += "new PARTNER_GROUPING_IDs."
            logging.warning(info_msg)
        else:
            print("Master file does not need to be updated.")
            # If no Treaties need to be updated, then ok to proceed
            self._ok_to_proceed = True
        return df_filtered
