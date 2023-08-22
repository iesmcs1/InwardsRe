import os
from os import path
from sys import displayhook
import time


import pandas as pd
import numpy as np

from inwards_tasks.utils import get_report_period

try:
    from ..treaty_register import Register
    from ..utils import get_abs_path, check_files_in_folder, import_fx_rates
except ImportError:
    import sys
    current_filepath = path.abspath(__file__)
    inwrads_tasks_folder = os.path.split(os.path.split(current_filepath)[0])[0]
    sys.path.append(inwrads_tasks_folder)
    from treaty_register import Register
    from utils import get_abs_path, check_files_in_folder, import_fx_rates


expected_files = ['DEPTOEPAR',
                  'FACULTEXP',
                  'SPECIFEXP',
                  'TARENTTAB',
                  'TREATYEXP',
                  'TREATYREG',
                  'FX_RATES']

class Exposure:
    register_cols = ['Balloon ID',
                     'Signed %',
                     'Run-off',
                     'Curr',
                     'Type/Form',
                     'B_C']

    def __init__(self,
                 folder: str,
                 apply_run_off: bool = True) -> None:

        check_files_in_folder(files=expected_files, folder=folder)
        
        self.folder = folder
        self.report_date = None
        self.apply_run_off = apply_run_off
        self.fx_rates = import_fx_rates(
            filepath=get_abs_path('FX_RATES', folder=self.folder))

        self._register = None
        self._load_register()

        self.treaty_exposure = self._import_and_process_exposure('TREATYEXP')
        self.fac_exposure = self._import_and_process_exposure('FACULTEXP')
        self.spec_exposure = self._import_and_process_exposure('SPECIFEXP')
    
    # def __sub__(self, object):
    #     if self.report_date > object.report_date:
    #         new_exp, old_exp = self, object
    #     elif self.report_date < object.report_date:
    #         old_exp, new_exp = self, object
    #     else:
    #         err = "Exposure classes need to have different dates."
    #         err += f"{self.report_date} and {object.report_date} was passed."
    #         raise ValueError(err)
        
    #     df_named_exp_compari = pd.merge(left=new_exp.named_exposure)

    
    @property
    def register(self) -> pd.DataFrame:
        return self._register
    
    @register.setter
    def register(self, reg_class: Register) -> None:
        # Assign only register DataFrame to property
        df = reg_class.register.copy()

        # Assign report_date from Register class to this class
        self.report_date = reg_class.report_date

        # Exclude duplicates
        # Sort values
        df.sort_values(by=['Comp', 'Seq', 'UW Yr'],
                       ascending=[True, True, False],
                       inplace=True)
        # Keep only Latest UW Year
        df.drop_duplicates(subset=['Balloon ID'], inplace=True)
        
        # Exclude XL treaties, since IM exclude. Same for expired FACs
        filter_no_xl = df['Type/Form'] != 'XL'
        filter_expired_facs = (df['Seq'].str[1] == '3') & (df['Pd End'] < self.report_date)
        filter_invalid_epi = df['EPI is Rev EPI or EPI'] > 1

        df = df.loc[(filter_no_xl)
                     & (~filter_expired_facs)
                     & (filter_invalid_epi)].copy()

        df['B_C'] = df['UPR'].str[0]
        df.loc[df['B_C'] == 'X', 'B_C'] = df['Seq'].str[0]

        df['FX_Rate'] = df['Curr'].map(self.fx_rates)

        df['Lapsed Our TPE-EUR'] = (
            df['Signed %'] * df['TPE'] * df['Run-off'] / df['FX_Rate'])
        
        self._register = df
    
    @property
    def report_period(self) -> str:
        return get_report_period(self.report_date)
    
    @property
    def named_exposure(self) -> pd.DataFrame:
        df = self._join_named_exposure_tables()

        cond = [~df['Exp Fac-100%'].isna(),
                ~df['Exp Spec-100%'].isna(),
                ~df['Exp Treaty-100%'].isna()]

        choice = [df['Exp Fac-100%'],
                  df['Exp Spec-100%'],
                  df['Exp Treaty-100%']]

        # Had to use np.select because loc was given reindex error
        df['Exp-100%'] = np.select(condlist=cond,
                                   choicelist=choice,
                                   default=np.nan)

        # Copy Treaty Exposure Share from Register
        df['Share'] = np.where(~df['Atradius Share'].isna(),
                               df['Atradius Share'],
                               df['Signed %'])

        df['EUR_KN_Exp'] = df['Exp-100%'] * df['Share'] / df['FX_Rate']
        
        df = df[['PK_Balloon_BUYER',
                 'Balloon ID',
                 'Bus ID',
                 'B_C',
                 'Share',
                 'Run-off',
                 'Currency',
                 'FX_Rate',
                 'Exp Treaty-100%',
                 'Exp Spec-100%',
                 'Exp Fac-100%',
                 'Exp-100%',
                 'EUR_KN_Exp']].copy()
        
        if self.apply_run_off:
            df['EUR_KN_Exp'] *= df['Run-off']
        elif not self.apply_run_off:
            df.drop(columns=['Run-off'], inplace=True)
        
        return df

    def _load_register(self) -> None:
        register_filepath = get_abs_path(file='TREATYREG', folder=self.folder)
        self.register = Register(filepath=register_filepath)

    def _import_and_process_exposure(self, file) -> pd.DataFrame:
        df = pd.read_csv(get_abs_path(file=file, folder=self.folder),
                         dtype={'Balloon ID': str, 'Bus ID': str},
                         sep='\t',
                         skipfooter=1,
                         engine='python')

        for col in df.select_dtypes([np.object]):
            df[col] = df[col].str.strip()

        if 'TREATYEXP' in file:
            df.rename(columns={'Current Exposure': 'Exp Treaty-100%'},
                      inplace=True)

        elif 'FACULTEXP' in file:
            df.columns = ['Balloon ID',
                          'Bus ID',
                          'Amount Requested',
                          'Exp Fac-100%',
                          'Currency',
                          'Atradius Share',
                          'LOB',
                          'Answer Date',
                          'Request Date',
                          'Expiry Date',
                          'Comments']

            # Turn AtradiusReshare into %
            df['Atradius Share'] = df['Atradius Share'] / 100

        elif 'SPECIFEXP' in file:
            df.columns = ['Balloon ID',
                          'Bus ID',
                          'Amount Requested',
                          'Exp Spec-100%',
                          'Currency',
                          'Atradius Share',
                          'LOB',
                          'Answer Date',
                          'Request Date',
                          'Expiry Date',
                          'Comments']

            # Turn AtradiusReshare into %
            df['Atradius Share'] = df['Atradius Share'] / 100

        df['PK_Balloon_BUYER'] = df['Balloon ID'] + df['Bus ID']

        df['FX_Rate'] = df['Currency'].map(self.fx_rates)

        df = df.merge(right=self.register[self.register_cols],
                      on='Balloon ID',
                      how='left')

        return df

    def get_duplicated_exposure(self) -> pd.DataFrame:
        df = pd.concat([self.fac_exposure,
                        self.spec_exposure,
                        self.treaty_exposure])

        return df.loc[df['PK_Balloon_BUYER'].duplicated(keep=False)]
    
    def _join_named_exposure_tables(self) -> pd.DataFrame:
        return pd.concat([self.fac_exposure,
                          self.spec_exposure,
                          self.treaty_exposure])


def get_named_exposure_comparison(old_exp: Exposure,
                                  new_exp: Exposure) -> pd.DataFrame:
    df = pd.merge(left=new_exp.named_exposure,
                  right=old_exp.named_exposure,
                  how='outer',
                  on=['PK_Balloon_BUYER'],
                  suffixes=(f'-{new_exp.report_period}',
                            f'-{old_exp.report_period}'))
    
    # column name, function
    delta_cols = [('Δ', '_calculate_total_tpe_diff'),
                  'Δ Org Growth',
                  'Δ New/Expired',
                  'Δ Share',
                  'Δ FX_Rate',
                  'Δ Run-off']

    col_loc = 3
    for col in delta_cols:
        df.insert(loc=col_loc, column=col, value=0)
        col_loc += 1
    
    df.insert()
