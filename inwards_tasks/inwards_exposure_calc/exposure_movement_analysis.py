import pandas as pd
import numpy as np
from .named_and_treaty_exposure import Exposure
from ..utils import get_abs_path
from .utils import get_report_period_str

class Movements:
    def __init__(
        self,
        exposures : dict,
        run_off : bool
    ) -> None:
        """Creates Movements Class object.

        Parameters
        ----------
        exposures : 2-item list of Exposure objects
            The first list item is the latest Exposure Object
        """
        self.new_exposure = Exposure(
            folder=exposures['new_exp']['folderpath'],
            apply_run_off=run_off
        )
        self.old_exposure = Exposure(
            folder=exposures['old_exp']['folderpath'],
            apply_run_off=run_off
        )

        self.dates_str = {
            'new': get_report_period_str(exposures['new_exp']['report_date']),
            'old': get_report_period_str(exposures['old_exp']['report_date'])
        }

        self.named_exposure = self.get_named_exposure_comparison()
        self.treaty_exposure = self.treaty_exposure_comparison()
        self.exposure_summary = self.get_overall_summary()
    
    @staticmethod
    def period_to_string(date):
        if type(date) is str:
            date = pd.to_datetime(date, dayfirst=True)
        date_str = '-{}Q{}'.format(str(date.year)[2:],
                                  date.quarter)
        
        return date_str
    
    @staticmethod
    def _get_buyer_table(data_folder):
        df = pd.read_csv(get_abs_path('DEPTOE', folder=data_folder),
                         sep='\t',
                         dtype={'BusID': str, 'ParentBusID': str},
                         skipfooter=1,
                         engine='python')
        
        df.rename(columns={'BusID': 'Bus ID'}, inplace=True)
        return df
    
    @staticmethod
    def _get_parent_table(data_folder):
        df = pd.read_csv(get_abs_path('ARENTT', folder=data_folder),
                         sep='\t',
                         dtype={'BusID': str},
                         skipfooter=1,
                         engine='python')
        
        df.rename(
            columns={'BusID': 'ParentBusID', 'Country': 'ParentCountry'},
            inplace=True
        )
        return df

    
    def get_buyer_parents(self):
        new_buyer_table = self._get_buyer_table(self.new_exposure.folder)
        old_buyer_table = self._get_buyer_table(self.old_exposure.folder)
        buyer_table = pd.concat([new_buyer_table, old_buyer_table])
        buyer_table.drop_duplicates(subset='Bus ID', inplace=True)
        
        new_parent_table = self._get_parent_table(self.new_exposure.folder)
        old_parent_table = self._get_parent_table(self.old_exposure.folder)
        parent_table = pd.concat([new_parent_table, old_parent_table])
        parent_table.drop_duplicates(subset='ParentBusID', inplace=True)

        df = pd.merge(left=buyer_table,
                      right=parent_table,
                      how='left',
                      on='ParentBusID')
        
        df.loc[df['Parent'].isna(), 'Parent'] = df['Debtor']
        df.loc[df['ParentCountry'].isna(), 'ParentCountry'] = df['Country']

        df.drop_duplicates(subset='Bus ID', inplace=True)

        return df
    
    def get_named_exposure_comparison(self):
        df = pd.merge(
            left=self.new_exposure.named_exposure,
            right=self.old_exposure.named_exposure,
            how='outer',
            on=['PK_Balloon_BUYER', 'Balloon ID', 'Bus ID'],
            suffixes=('-'+self.dates_str['new'], '-'+self.dates_str['old'])
        )
        
        delta_cols = ['Δ',
                      'Δ Org Growth',
                      'Δ New/Expired',
                      'Δ Share',
                      'Δ FX_Rate',
                      'Δ Run-off']

        col_loc = 3
        for col in delta_cols:
            df.insert(loc=col_loc, column=col, value=0)
            col_loc += 1

        df['Δ'] = (df['EUR_KN_Exp-'+self.dates_str['new']].fillna(0)
                  - df['EUR_KN_Exp-'+self.dates_str['old']].fillna(0))

        # This checks that current and previos currency match
        # Also checks that Risk exists in current and previous quarter
        currency_match = (
            df['Currency-'+self.dates_str['new']]
            != df['Currency-'+self.dates_str['old']]
        )
        
        df['Δ Org Growth'] = np.where(
            currency_match,
            0,
            ((df['Exp-100%-'+self.dates_str['new']]
             - df['Exp-100%-'+self.dates_str['old']])
            * df['Share-'+self.dates_str['old']]
            * df['Run-off-'+self.dates_str['old']]
            / df['FX_Rate-'+self.dates_str['old']])
        )

        df['Δ New/Expired'] = np.where(currency_match, df['Δ'], 0)

        df['Δ Share'] = np.where(
            currency_match,
            0,
            (df['Share-'+self.dates_str['new']] - df['Share-'+self.dates_str['old']])
            * df['Exp-100%-'+self.dates_str['new']]
            * df['Run-off-'+self.dates_str['old']]
            / df['FX_Rate-'+self.dates_str['old']]
        )

        df['Δ FX_Rate'] = np.where(
            currency_match,
            0,
            (df['Exp-100%-'+self.dates_str['new']]
            * df['Share-'+self.dates_str['new']]
            * df['Run-off-'+self.dates_str['old']]
            * (1 / df['FX_Rate-'+self.dates_str['new']]
              - 1 / df['FX_Rate-'+self.dates_str['old']]))
        )

        # Zero all Exposures that changed currency between quarters
        df.loc[
            df['Currency-'+self.dates_str['new']] != df['Currency-'+self.dates_str['old']],
            'Δ FX_Rate'
        ] = 0

        df.loc[
            (df['Currency-'+self.dates_str['new']] == 'EUR')
            & (df['Currency-'+self.dates_str['old']] == 'EUR'),
            'Δ FX_Rate'
        ] = 0

        df['Δ Run-off'] = np.where(
            currency_match,
            0,
            (df['Exp-100%-'+self.dates_str['new']]
            * df['Share-'+self.dates_str['new']]
            * (df['Run-off-'+self.dates_str['new']] - df['Run-off-'+self.dates_str['old']])
            / df['FX_Rate-'+self.dates_str['new']])
        )

        buyer_parent = self.get_buyer_parents()

        df = df.merge(
            buyer_parent[['Bus ID', 'ParentBusID', 'Parent', 'ParentCountry']],
            how='left',
            on='Bus ID'
        )

        return df

    def top_by_parent(self, n=20) -> pd.DataFrame:
            df = self.named_exposure
            df_top = df.groupby(
                by=['ParentBusID', 'Parent', 'ParentCountry']
            ).sum()
            df_top = df_top.nlargest(n, 'EUR_KN_Exp-'+self.dates_str['new'])
            df_top.reset_index(inplace=True)

            df_top = df_top[['ParentBusID',
                             'Parent',
                             'ParentCountry',
                             'EUR_KN_Exp-'+self.dates_str['old'],
                             'EUR_KN_Exp-'+self.dates_str['new']]]
            
            df_top['Δ'] = (
                df_top['EUR_KN_Exp-'+self.dates_str['new']]
                - df_top['EUR_KN_Exp-'+self.dates_str['old']]
            )

            df_top['Δ %'] = df_top['Δ'] / df_top['EUR_KN_Exp-'+self.dates_str['old']]
            
            return df_top
    
    def treaty_exposure_comparison(self):
        register_main_cols = ['Balloon ID',
                              'CONTRACT_ID',
                              'B_C',
                              'Run-off',
                              'Lapsed Our TPE-EUR',
                              'Curr',
                              'FX_Rate',
                              'Signed %',
                              'TPE']

        new_reg = self.new_exposure.register[register_main_cols]
        old_reg = self.old_exposure.register[register_main_cols]

        df_reg = pd.merge(left=new_reg,
                          right=old_reg,
                          how='outer',
                          on=['Balloon ID'],
                          suffixes=('-'+self.dates_str['new'],
                                    '-'+self.dates_str['old']))
        
        df_named_by_treaty = self.named_exposure[[
            'Balloon ID',
            'EUR_KN_Exp-'+self.dates_str['old'],
            'EUR_KN_Exp-'+self.dates_str['new']
        ]].groupby(by='Balloon ID').sum()
        
        df_reg = df_reg.merge(right=df_named_by_treaty,
                              how='left',
                              on='Balloon ID')

        new_old_qrt = [self.dates_str['new'], self.dates_str['old']]

        for qrt in new_old_qrt:
            df_reg['IM TPE-'+qrt] = df_reg[[
                'Lapsed Our TPE-EUR-'+qrt,
                'EUR_KN_Exp-'+qrt
            ]].fillna(0).max(axis=1)
        
        for qrt in new_old_qrt:
            df_reg['EUR_UNK_TPE-'+qrt] = (df_reg['IM TPE-'+qrt].fillna(0)
                                         - df_reg['EUR_KN_Exp-'+qrt].fillna(0))
        
        df_reg['Δ EUR_KN_Exp'] = (df_reg['EUR_KN_Exp-'+self.dates_str['new']]
                                 - df_reg['EUR_KN_Exp-'+self.dates_str['old']])
        df_reg['Δ EUR_UNK_TPE'] = (df_reg['EUR_UNK_TPE-'+self.dates_str['new']]
                                 - df_reg['EUR_UNK_TPE-'+self.dates_str['old']])
        
        currency_match = (df_reg['Curr-'+self.dates_str['new']]
                         != df_reg['Curr-'+self.dates_str['old']])
        
        df_reg['Δ TPE-FX_Rate'] = np.where(
            currency_match,
            0,
            (df_reg['TPE-'+self.dates_str['new']]
            * df_reg['Signed %-'+self.dates_str['new']]
            * df_reg['Run-off-'+self.dates_str['old']]
            * (1 / df_reg['FX_Rate-'+self.dates_str['new']]
              - 1 / df_reg['FX_Rate-'+self.dates_str['old']]))
        )
        
        df_reg = df_reg[['Balloon ID',
                         'Δ EUR_KN_Exp',
                         'Δ EUR_UNK_TPE',
                         'Δ TPE-FX_Rate',
                         # main columns - Current Quarter
                         'B_C-'+self.dates_str['new'],
                         'CONTRACT_ID-'+self.dates_str['new'],
                         'Lapsed Our TPE-EUR-'+self.dates_str['new'],
                         'EUR_KN_Exp-'+self.dates_str['new'],
                         'IM TPE-'+self.dates_str['new'],
                         'EUR_UNK_TPE-'+self.dates_str['new'],
                         # main columns - Previous Quarter
                         'B_C-'+self.dates_str['old'],
                         'CONTRACT_ID-'+self.dates_str['old'],
                         'Lapsed Our TPE-EUR-'+self.dates_str['old'],
                         'EUR_KN_Exp-'+self.dates_str['old'],
                         'IM TPE-'+self.dates_str['old'],
                         'EUR_UNK_TPE-'+self.dates_str['old'],
                         # AUX columns - Current Quarter
                         'Run-off-'+self.dates_str['new'],
                         'Curr-'+self.dates_str['new'],
                         'FX_Rate-'+self.dates_str['new'],
                         'Signed %-'+self.dates_str['new'],
                         'TPE-'+self.dates_str['new'],
                         # AUX columns - Previous Quarter
                         'Run-off-'+self.dates_str['old'],
                         'Curr-'+self.dates_str['old'],
                         'FX_Rate-'+self.dates_str['old'],
                         'Signed %-'+self.dates_str['old'],
                         'TPE-'+self.dates_str['old']]]

        return df_reg
    
    def get_overall_summary(self):
        df = self.treaty_exposure

        summary_index = ['Portfolio',
                         'Credit',
                         'Credit - Known',
                         'Credit - Unknown',
                         'Bond',
                         'Bond - Known',
                         'Bond - Unknown']

        summary = pd.DataFrame(data=0,
                               index=summary_index,
                               columns=[self.dates_str['old'],
                                        self.dates_str['new']])

        for i in summary.columns:
            only_bond = df['B_C-'+i].str[0] == 'B'
            only_cred = df['B_C-'+i].str[0] == 'C'
            
            summary.loc['Portfolio', i] = df['IM TPE-'+i].sum()
            summary.loc['Credit', i] = df.loc[only_cred, 'IM TPE-'+i].sum()
            summary.loc['Bond', i] = df.loc[only_bond, 'IM TPE-'+i].sum()
            
            summary.loc['Credit - Known',
                        i] = df.loc[only_cred, 'EUR_KN_Exp-'+i].sum()
            summary.loc['Bond - Known',
                        i] = df.loc[only_bond, 'EUR_KN_Exp-'+i].sum()
            
            summary.loc['Credit - Unknown',
                        i] = df.loc[only_cred, 'EUR_UNK_TPE-'+i].sum()
            summary.loc['Bond - Unknown',
                        i] = df.loc[only_bond, 'EUR_UNK_TPE-'+i].sum()

        # Divide by 1,000,000 to match IM Data BAT
        summary = summary / 1000000

        summary['Δ'] = (summary[self.dates_str['new']]
                        - summary[self.dates_str['old']])

        summary['Δ %'] = summary['Δ'] / summary[self.dates_str['old']]

        summary.insert(loc=0,
                       column='Amount in EUR m',
                       value=summary.index.values)
        
        # Not dropping Index because is being used to generate
        # Summary text in the `movement_analysis_to-txt` function

        return summary
    
    @staticmethod
    def _get_movement_verb(amount, cap=False):
        """
        
        Parameters
        ----------
        amount : int or float
        cap : Boolean
            if word needs to be capitalized or not
        """
        text = None
        if amount > 0:
            text = 'increase'
        else:
            text = 'decrease'

        if cap:
            return text.capitalize()
        return text
        
    def movement_analysis_to_txt(self, class_type: str):
        """

        Parameters
        ----------

        class : 'bond' or 'credit'

        """
        if not class_type in ['bond', 'credit']:
            raise ValueError(
                'class_type should be "bond" or "credit". {} was passed'.format(
                    class_type
                ))
        class_type = class_type.capitalize()

        df = self.exposure_summary[['Δ', 'Δ %']]

        df_kn = self.treaty_exposure
        df_kn = df_kn.loc[df_kn['B_C-'+self.dates_str['new']] == class_type[0]]
        df_kn = pd.concat([df_kn.nlargest(2, columns='Δ EUR_KN_Exp'),
                           df_kn.nsmallest(2, columns='Δ EUR_KN_Exp')])
        df_kn = df_kn[['Balloon ID', 'Δ EUR_KN_Exp']]

        df_unk = self.treaty_exposure
        df_unk = df_unk.loc[df_unk['B_C-' +
                                   self.dates_str['new']] == class_type[0]]
        df_unk = pd.concat([df_unk.nlargest(2, columns='Δ EUR_UNK_TPE'),
                           df_unk.nsmallest(2, columns='Δ EUR_UNK_TPE')])
        df_unk = df_unk[['Balloon ID', 'Δ EUR_UNK_TPE']]

        overall = '{}:\n'.format(class_type)
        overall += '{:.1%} (or €{:.1f}m) from last Quarter\n\n'.format(
            df.loc[class_type, 'Δ %'],
            df.loc[class_type, 'Δ']
        )

        # Known Analysis
        overall += '{} Known:\n'.format(class_type)
        overall += '{}: {:.1%} (or €{:,.1f}m)\n'.format(
            self._get_movement_verb(df.loc[class_type+' - Known', 'Δ'], True),
            df.loc[class_type+' - Known', 'Δ %'],
            df.loc[class_type+' - Known', 'Δ']
        )

        overall += 'Largest Increases:\n'
        overall += '{}'.format(self.get_known_movements(class_type,
                                                        'increase'))

        overall += 'Largest Decreases:\n'
        overall += '{}'.format(self.get_known_movements(class_type,
                                                        'decrease'))

        # Unknown Analysis
        overall += '{} Unkown:\n'.format(class_type)
        overall += '{}: {:.1%} (or €{:,.1f}m)\n'.format(
            self._get_movement_verb(
                df.loc[class_type+' - Unknown', 'Δ'], True),
            df.loc[class_type+' - Unknown', 'Δ %'],
            df.loc[class_type+' - Unknown', 'Δ']
        )
        overall += 'Largest Increases:\n'
        overall += '{}'.format(self.get_unknown_movements(class_type,
                                                          'increase'))
        overall += 'Largest Decreases:\n'
        overall += '{}'.format(self.get_unknown_movements(class_type,
                                                          'decrease'))

        file_name = "{} Exposure Movement.txt".format(class_type)

        with open(file_name, "w") as f:
            f.writelines(overall)
        
        print("File '{}' was created.".format(file_name))

        return overall

    def get_kn_treaty_movement_summary(self, balloon_id):
        """Return movements of a particular Treaty (Balloon ID).

        Parameters
        ----------
        balloon_id : str or list of str
            This will be used to filter the DataFrame.
        """
        text = []
        balloons = []
        if type(balloon_id) is str:
            # if is str, then I add to the balloons list object
            # and iterate it
            balloons.append(balloon_id)
        elif type(balloon_id) is list:
            balloons = balloon_id
        else:
            raise ValueError('balloon_id needs to be str or list of str.')

        df = self.named_exposure.groupby(by='Balloon ID').sum()
        
        # Iterate through each Balloon ID
        for id in balloons:
            tty = df.loc[id]

            # 1e6 is to convert value into millions
            id_text = '{} -> €{:+,.0f}m {}:\n'.format(
                tty.name,
                tty['Δ'] / 1e6,
                self._get_movement_verb(tty['Δ'] / 1e6))

            # The column header pattern for Delta columns is the delta symbol
            # and a space ('Δ '). In case this changes in the future, I just 
            # have to update it
            for i in [col for col in tty.index if 'Δ ' in col]:
                # 1e6 is to convert value into millions
                val_in_mill = tty[i] / 1e6
                # To remove movements that are close to zero
                if val_in_mill >= 1 or val_in_mill <= -1:
                    id_text += '{:4}€{:+,.0f}m {} from {}\n'.format(
                        '',
                        val_in_mill,
                        self._get_movement_verb(val_in_mill),
                        i[2:]
                    )
        
                id_text = id_text.replace('Org Growth',
                                          'Organic Growth/Decline')
                id_text = id_text.replace('FX_Rate', 'FX-Rate')
            text.append(id_text[:-1])

        return '\n'.join(text)
    
    def get_ukn_treaty_movement_summary(self, balloon_id):
        text = []
        balloons = []
        if type(balloon_id) is str:
            # if is str, then I add to the balloons list object
            # and iterate it
            balloons.append(balloon_id)
        elif type(balloon_id) is list:
            balloons = balloon_id
        else:
            raise ValueError('balloon_id needs to be str or list of str.')

        df = self.treaty_exposure

        for id in balloons:
            tty_index = df.loc[df['Balloon ID'] == id].index
            tty = df.loc[tty_index[0]]

            text.append('{} -> €{:+,.0f}m {}:'.format(
                tty['Balloon ID'],
                tty['Δ EUR_UNK_TPE'] / 1e6,
                self._get_movement_verb(amount=tty['Δ EUR_UNK_TPE'])
            ))

        return '\n'.join(text)

    def get_class_movement_summary(self, class_type, sub_type=None) -> str:
        """

        Parameters
        ----------

        class_type : 'bond' or 'credit'
        sub_type : 'kn', 'unk' or None

        """
        if not class_type in ['bond', 'credit']:
            raise ValueError(
                'class_type should be "bond" or "credit". {} was passed'.format(
                    class_type
                ))
        class_type = class_type.capitalize()
        if sub_type == 'kn':
            class_type += ' - Known'
        elif sub_type == 'unk':
            class_type += ' - Unknown'

        df = self.exposure_summary

        overall = '{}:\n'.format(class_type)
        overall += '{:+.1%} (or €{:+,.1f}m) {} from last Quarter\n'.format(
            df.loc[class_type, 'Δ %'],
            df.loc[class_type, 'Δ'],
            self._get_movement_verb(df.loc[class_type, 'Δ'])
        )

        return overall
    
    def _get_most_significant_balloons(self, class_type, sub_type, top_bottom):
        
        if sub_type == 'kn':
            col = 'Δ EUR_KN_Exp'
        elif sub_type == 'unk':
            col = 'Δ EUR_UNK_TPE'
        
        df = self.treaty_exposure
        # DataFrame with only Bond or Credit Treaties
        df = df.loc[df['B_C-'+self.dates_str['new']]
                    == class_type[0].capitalize()]
        
        if top_bottom == 'top':
            kn_exp_balloons = df.nlargest(2, columns=col)
        elif top_bottom == 'bottom':
            kn_exp_balloons = df.nsmallest(2, columns=col)

        return kn_exp_balloons['Balloon ID']

    def get_movement_summary(self, class_type, to_txt=False):
        """Return full text with Known and Unknown explanation.

        Parameters
        ----------
        class_type : 'bond' or 'credit'
        to_txt : bool
            if True, then create txt file with movement summary text
            else, do not create txt file.
        """
        if not type(class_type) is str:
            raise ValueError('class_type arg needs to be str.')
        elif not class_type in ['bond', 'credit']:
            raise ValueError('class_type arg needs to be "bond" or "credit".')

        # Array to hold text. Will use '\n'.join() below
        text = []
        text.append(self.get_class_movement_summary(class_type=class_type))
        text.append(self.get_class_movement_summary(class_type=class_type,
                                                    sub_type='kn'))
        
        text.append('Highlights:')
        # class_type - Treaties with highest KNOWN movements
        # ==================================================
        kn_exp_balloons_top = self._get_most_significant_balloons(
            class_type=class_type,
            sub_type='kn',
            top_bottom='top')

        for balloon in kn_exp_balloons_top:
            text.append(self.get_kn_treaty_movement_summary(balloon))
        
        # Blank line between Increase and Decrease
        text.append('')
        
        kn_exp_balloons_bottom = self._get_most_significant_balloons(
            class_type=class_type,
            sub_type='kn',
            top_bottom='bottom')

        for balloon in kn_exp_balloons_bottom:
            text.append(self.get_kn_treaty_movement_summary(balloon))
        
        
        text.append('')
        text.append(self.get_class_movement_summary(class_type=class_type,
                                                    sub_type='unk'))
        
        text.append('Highlights:')

        unk_exp_balloons_top = self._get_most_significant_balloons(
            class_type=class_type,
            sub_type='unk',
            top_bottom='top')

        for balloon in unk_exp_balloons_top:
            text.append(self.get_ukn_treaty_movement_summary(balloon))

        unk_exp_balloons_bottom = self._get_most_significant_balloons(
            class_type=class_type,
            sub_type='unk',
            top_bottom='bottom')

        for balloon in unk_exp_balloons_bottom:
            text.append(self.get_ukn_treaty_movement_summary(balloon))

        # Output to .txt file
        if to_txt:
            file_name = "{} Exposure Movement.txt".format(
                class_type.capitalize()
            )
        
            with open(file_name, "w") as f:
                f.writelines('\n'.join(text))
        
            print("File '{}' was created.\n".format(file_name))

        return '\n'.join(text)

    def report_to_excel(self):
        excel_file = "Inwards Exposure {}Q{}-produced on {}.xlsx".format(
            self.new_exposure.report_date.year,
            self.new_exposure.report_date.quarter,
            pd.to_datetime('now', utc=True).strftime('%Y%m%d-%Hh%M')
        )

        writer = pd.ExcelWriter(excel_file,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')
        
        dfs_to_excel = {'Summary': self.exposure_summary,
                        'Treaty Summary': self.treaty_exposure,
                        'Named Exposure': self.named_exposure,
                        'Top Exposures': self.top_by_parent()}
        
        for sh_name, v in dfs_to_excel.items():
            v.to_excel(writer, sheet_name=sh_name, index=False)
        
        #----------------------------------------------------------------------
        # Format settings
        wb = writer.book

        num_0d_format = wb.add_format({'num_format': '#,##0'})
        num_1d_format = wb.add_format({'num_format': '#,##0.0'})
        num_2d_format = wb.add_format({'num_format': '#,##0.00'})
        num_4d_format = wb.add_format({'num_format': '#,##0.0000'})

        pct_1d_format = wb.add_format({'num_format': '0.0%'})
        pct_2d_format = wb.add_format({'num_format': '0.00%'})

        # Formatting Summary
        ws_summary = writer.sheets['Summary']
        ws_summary.set_column('A:A', 16)
        ws_summary.set_column('B:D', 9.14, num_1d_format)
        ws_summary.set_column('E:E', 9.14, pct_1d_format)

        text_box_options = {
            'width': 450,
            'height': 680
        }

        ws_summary.insert_textbox(
            'G2',
            self.get_movement_summary(class_type='credit'),
            text_box_options)
        
        ws_summary.insert_textbox(
            'O2',
            self.get_movement_summary(class_type='bond'),
            text_box_options)
        
        # Formatting Treaty Summary
        ws_tty_summary = writer.sheets['Treaty Summary']
        ws_tty_summary.set_column('A:A', 13.86)
        ws_tty_summary.set_column('B:D', 19.43, num_0d_format)
        ws_tty_summary.set_column('E:F', 14)
        ws_tty_summary.set_column('G:G', 28.14, num_0d_format)
        ws_tty_summary.set_column('H:J', 23, num_0d_format)
        ws_tty_summary.set_column('K:L', 13.29)
        ws_tty_summary.set_column('M:M', 28.14, num_0d_format)
        ws_tty_summary.set_column('N:P', 23, num_0d_format)
        # Parameters Current Quarter
        ws_tty_summary.set_column('Q:Q', 16.71, pct_2d_format)
        ws_tty_summary.set_column('R:R', 13.71)
        ws_tty_summary.set_column('S:S', 17, num_4d_format)
        ws_tty_summary.set_column('T:T', 17, pct_2d_format)
        ws_tty_summary.set_column('U:U', 17, num_0d_format)
        # Parameters Previous Quarter
        ws_tty_summary.set_column('V:V', 16.71, pct_2d_format)
        ws_tty_summary.set_column('W:W', 13.71)
        ws_tty_summary.set_column('X:X', 17, num_4d_format)
        ws_tty_summary.set_column('Y:Y', 17, pct_2d_format)
        ws_tty_summary.set_column('Z:Z', 17, num_0d_format)

        # Formatting Named Exposure
        ws_named_exp = writer.sheets['Named Exposure']
        ws_named_exp.set_column('A:A', 22)
        ws_named_exp.set_column('B:C', 11)
        ws_named_exp.set_column('D:I', 13.43, num_0d_format)
        # Parameters Current Quarter
        ws_named_exp.set_column('J:J', 13.29)
        ws_named_exp.set_column('K:L', 15, pct_2d_format)
        ws_named_exp.set_column('M:M', 18)
        ws_named_exp.set_column('N:N', 17.29, num_4d_format)
        ws_named_exp.set_column('O:S', 24, num_0d_format)
        # Parameters Previous Quarter
        ws_named_exp.set_column('T:T', 13.29)
        ws_named_exp.set_column('U:V', 15, pct_2d_format)
        ws_named_exp.set_column('W:W', 18)
        ws_named_exp.set_column('X:X', 17.29, num_4d_format)
        ws_named_exp.set_column('Y:AC', 24, num_0d_format)
        ws_named_exp.set_column('AD:AD', 15.71)
        ws_named_exp.set_column('AE:AE', 44)
        ws_named_exp.set_column('AF:AF', 17.86)

        # Format Top Exposures
        ws_top_exp = writer.sheets['Top Exposures']
        ws_top_exp.set_column('A:A', 10)
        ws_top_exp.set_column('B:B', 38)
        ws_top_exp.set_column('C:C', 14)
        ws_top_exp.set_column('D:F', 20, num_0d_format)
        ws_top_exp.set_column('G:G', 10, pct_1d_format)

        # Apply Autofilter to all sheets
        for sh_name, v in dfs_to_excel.items():
            if sh_name != 'Summary':
                writer.sheets[sh_name].autofilter(0, 0, v.shape[0], v.shape[1]-1)

        writer.save()

        print('File "{}" created.'.format(excel_file))

        return None
