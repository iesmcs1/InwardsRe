from .utils import get_path, import_fx_rates
from ..utils import import_run_off_rates
import pandas as pd
import numpy as np

class SolvencyRegister:
    main_cols = [
        'Comp',
        'Seq',
        'UW Yr',
        'Short',
        'Cedant'
    ]

    def __init__(self, report_date, fx_filepath: str, folder: str) -> None:
        self.folder = folder
        self.report_date = pd.to_datetime(report_date, dayfirst=True)
        self.fx_rates = import_fx_rates(filepath=fx_filepath)
        self.run_off = import_run_off_rates()
        self.register = self._import_and_process_register()
    
    def _import_and_process_register(self):
        df = pd.read_csv(get_path(file='TREATYREG', folder=self.folder),
                         header=None, sep='\t')
        
        # Remove excess Lines where 2nd column is NaN
        df = df.loc[~df[1].isna()]
        # Reset index so that I use first row as header
        df.reset_index(drop=True, inplace=True)
        df.columns = df.iloc[0]
        # No need for row with column headers
        df.drop(df.index[0], inplace=True)

        cols_to_strip = ['Comp', 'Seq', 'Cedant', 'Short', 'Curr', 'UPR']
        for col in cols_to_strip:
            df[col] = df[col].str.strip()
        
        date_cols = ['Pd Beg', 'Pd End']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], dayfirst=True)
        
        df['UW Yr'] = df['UW Yr'].astype(int)
        df['UWY_str'] = df['UW Yr']
        df['UWY_str'] = df['UWY_str'].apply(lambda x: '{:0>2}'.format(x))

        # Correct Currencies
        df.loc[df['Curr'] == 'PEI', 'Curr'] = 'PEN'
        df.loc[df['Curr'] == 'MXP', 'Curr'] = 'MXN'
        df.loc[df['Curr'] == 'RUR', 'Curr'] = 'RUB'

        # Rounding necessary to match Solvency Run
        df['Signed %'] = df['Signed %'].astype(np.float64).round(4)
        # Convert Signed lines to percent
        df['Signed %'] = df['Signed %'] / 100
        
        # Manually calculating Our TPE due to errors in the past
        df['Our TPE'] = (df['TPE'].astype(np.float64)
                         * df['Signed %'].astype(np.float64))

        # Make Comp as st in correct format
        df['Comp'] = df['Comp'].apply(lambda x: '{0:0>5}'.format(x))
        # Create Balloon ID for merges
        df['Balloon ID'] = df['Comp'] + df['Seq']
        # Ref1 is Balloon ID + UW Year
        df['Ref1'] = df['Balloon ID'] + df['UWY_str']
        # Sort values
        df.sort_values(by=['Comp', 'Seq', 'UW Yr'],
                       ascending=[True, True, False],
                       inplace=True)
        
        # Keep only Latest UW Year
        df = df.drop_duplicates(subset=['Balloon ID'])

        # Another Constrain is to have only Treaties with  TPE above 2
        excluded_cond = [
            # remove XL Treaties
            (df['Seq'].str[1] == '2'),
            # remove Expired FAC's
            ((df['Seq'].str[1] == '3') | (df['Seq'].str[1] == '4'))
            & ((df['Pd End'] + np.timedelta64(1, 'D')) < self.report_date),
        ]
        excluded_choice = ['Excluded', 'Excluded']

        df['Excluded'] = np.select(condlist=excluded_cond,
                                   choicelist=excluded_choice)
        
        # # Manual Adjustments because of 2020Q3
        # print('solvency_register.py: Manual adjustments for 36024B1 0219')
        # df.loc[df['Ref1'] == '36024B1 0219', 'Excluded'] = 0

        # Keep only non-excluded Treaties
        df = df.loc[df['Excluded'] != 'Excluded']

        # Treaties that have EPI = 0 should have TPE = 0.
        # They should not be excluded, since they can have known TPE.
        df.loc[df['EPI is Rev EPI or EPI'].astype(np.float64) < 1, 'TPE'] = 0

        df['B_C'] = df['UPR'].str[0]
        df.loc[df['B_C'] == 'X', 'B_C'] = df['Seq'].str[0]

        # Import Group Fx Rates
        df['FX_Rate'] = df['Curr'].map(self.fx_rates)

        df['DaysElapsed'] = self.report_date - df['Pd End']
        df.loc[df['DaysElapsed'] <= pd.Timedelta(0), 'MonthsElapsed'] = 0
        df.loc[df['DaysElapsed'] > pd.Timedelta(0), 'MonthsElapsed'] = (
            df['DaysElapsed'] / np.timedelta64(1, 'Y') * 12
        )
        df['MonthsElapsed'] = df['MonthsElapsed'].round(0)
        df.drop(columns='DaysElapsed', inplace=True)

        # Bring in Run-off Rates to Calculate TPE
        df = df.merge(self.run_off,
                      on=['MonthsElapsed', 'UPR'],
                      how='left')
        
        # df['Run-off'] = df.apply(
        #     lambda x: self.run_off.loc[x['MonthsElapsed'], x['UPR']], axis=1)

        # Had to include this line to make it exactly as the IM output.
        # This is an edge case where the UPR ix XCL, Type is FAC and PD is
        # at quarter end.
        df.loc[(df['UPR'] == 'XCL')
               & (df['Type/Form'] == 'FAC')
               & (df['Pd End'] == self.report_date),
               'Run-off'] = 0

        # Convert due to operation error between float64 and int
        df['TPE'] = df['TPE'].astype(np.float64)

        df['Lapsed Our TPE-EUR'] = (
            df['Signed %'] * df['TPE'] * df['Run-off'] / df['FX_Rate']
        )

        # Type/Form == OC is Zero in IM. There was a cleanup in the Data
        # in 2020Q4, so moving forward the next statement should be redundant
        df.loc[df['Type/Form'] == 'OC', 'Lapsed Our TPE-EUR'] = 0

        # Create column to Retrieve Treaty Status

        return df
