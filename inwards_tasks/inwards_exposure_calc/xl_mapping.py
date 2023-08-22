from ..register.register import Register
import pandas as pd

class XLMapping:
    def __init__(
        self,
        register_path: str,
        IRE_GROUPING_path: str
    ) -> None:
        self.register = Register(register_path, 'assumed').register
        self.IRE_GROUPING_path = IRE_GROUPING_path

    def get_unambiguous_xl_data(self) -> pd.DataFrame:
        df = self.register

        # keep only latest Balloon ID
        df_latest = df.drop_duplicates(subset='Balloon ID')

        # keep only treaties that are NOT FAC FACX
        df_no_fac = df_latest.loc[~df['Type/Form'].isin(['FAC', 'FACX'])]

        # create DF with only XL
        df_xl = df_no_fac.loc[df['Type/Form'] == 'XL']

        # Get XL Data
        xl_data = df_xl.groupby(by='Ref2').agg(
            {'Layer Number': 'max', 'Limit': 'sum', 'xs': 'min'}
        ).reset_index()

        return xl_data
