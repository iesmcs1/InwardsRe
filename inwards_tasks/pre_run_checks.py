from .treaty_register import Register
import pandas as pd
import numpy as np
import os

SYMY_COLS_TO_USE = ['CREDIT_LIMIT_ID',
                    'BUYER_NUMBER',
                    'BUYER_NAME',
                    'BUYER_COUNTRY_ISO',
                    'LEGACY_POLICY',
                    'CUSTOMER_NUMBER',
                    'CUSTOMER_NAME',
                    'CLA_DATE',
                    'CLD_DATE',
                    'CLD_TOTAL_AMOUNT',
                    'CLD_TOTAL_AMOUNT_EURO',
                    'POLICY_CURRENCY',
                    'COMMITMENT_TYPE',
                    'EFFECTIVE_TO_DATE']

REGISTER_COLS_TO_USE = ['Comp',
                        'Seq',
                        'UW Yr',
                        'Short',
                        'Cedant',
                        'Uwriter',
                        'Pd Beg',
                        'Pd End',
                        'Inception',
                        'Type/Form',
                        'Term',
                        'Curr',
                        'UPR',
                        'UPR Description',
                        'Signed %',
                        'Bus Id']


def is_amount_na_or_zero(s: pd.Series) -> pd.Series:
    filter_amount_na_or_zero = (s == 0) | (s.isna())

    return np.where(filter_amount_na_or_zero,
                    f'{s.name} is 0 or empty.',
                    'OK')


def is_na(s: pd.Series) -> pd.Series:
    filter_is_na = s.isna()
    return np.where(filter_is_na,
                    f'{s.name} is empty. Will be removed from Solvency II data.',
                    'OK')

def is_duplicated(s: pd.Series,
                    tag_latest: bool = False) -> pd.Series:

    # Check for all duplicated entries.
    filter_isna = s.isna()
    filter_all_duplicated = s.duplicated(keep=False)

    condlist = [filter_isna,
                filter_all_duplicated]

    choicelist = ['Invalid Key',
                    'Duplicated entry']

    if tag_latest:
        # Filter all duplicated, marking the first entry
        # as not a duplicated.

        # This will return all the other duplicated entries
        # except the 1st one as True (duplicated).

        # The "~" means the inverse. In this case, it's returning the
        # 1st entry marked as not duplicated.
        filter_latest_duplicated = ~s.duplicated(keep='first')

        # Condition needs to be inserted first because we check if
        # it's duplicated and first appearance in data
        condlist.insert(1, (filter_all_duplicated) &
                        (filter_latest_duplicated))
        choicelist.insert(1, 'Duplicated entry (Latest)')

    return np.select(condlist=condlist,
                        choicelist=choicelist,
                        default='OK')

def has_blank_space_in_beg_or_end(s: pd.Series) -> pd.Series:
    filter_has_blank_space = (
        s.str.startswith(' ')) | (s.str.endswith(' '))

    return np.where(filter_has_blank_space,
                    f'Blank space in Beg or End in {s.name} column.',
                    'OK')

def is_balloon_id_pattern_correct(s: pd.Series) -> pd.Series:
    # Pattern is 5 digits, followed by B or C, followed by 1 digit
    # followed by a space and 2 more digits.
    # "~" negate so we keep only patterns that do not match.
    filter_balloon_id_pattern = s.str.contains(r"\d{5}[BC]\d{1}\s{1}\d{2}",
                                                regex=True)

    return np.where(filter_balloon_id_pattern,
                    'OK',
                    "Balloon ID pattern is incorrect.")


def is_customer_same_as_legacy_policy(dataframe: pd.DataFrame) -> pd.Series:
    filter_customer_legacy_pol_match = (
        dataframe['CUSTOMER_NAME'].str[:5] == dataframe['LEGACY_POLICY'].str[:5]
    )

    return np.where(filter_customer_legacy_pol_match,
                    'OK',
                    (
                        'CUSTOMER_NAME Comp ID '
                        + dataframe['CUSTOMER_NAME'].str[:5]
                        + ' does not match LEGACY_POLICY '
                        + dataframe['LEGACY_POLICY']
                    ))


def is_commit_type_and_leg_pol_valid(dataframe: pd.DataFrame) -> pd.Series:

    filter_commit_type_s = dataframe['COMMITMENT_TYPE'].str[0] == 'S'
    filter_legaly_pol_sa = dataframe['LEGACY_POLICY'].str[-2:] != 'SA'

    filter_commit_type_f = dataframe['COMMITMENT_TYPE'].str[0] == 'F'
    filter_legaly_pol_not_fac = dataframe['LEGACY_POLICY'].str[6] != '3'

    filter_commit_type_not_f = dataframe['COMMITMENT_TYPE'].str[0] != 'F'
    filter_legaly_pol_fac = dataframe['LEGACY_POLICY'].str[6] == '3'

    condlist = [
        # If Special and legacy pol doesn't end in SA
        (filter_commit_type_s) & (filter_legaly_pol_sa),
        # Checking if Commit Type is F and Leg Pol i[6] not 3
        (filter_commit_type_f) & (filter_legaly_pol_not_fac),
        # Checking if Commit Type is not F and Leg Pol i[6] is 3
        (filter_commit_type_not_f) & (filter_legaly_pol_fac)
    ]

    choicelist = [
        (
            'COMMITMENT_TYPE ' + dataframe['COMMITMENT_TYPE']
            + ' does not match LEGACY_POLICY ' + dataframe['LEGACY_POLICY']
            + ' (missing SA)'
        ),
        (
            'COMMITMENT_TYPE ' + dataframe['COMMITMENT_TYPE']
            + ' does not match LEGACY_POLICY '
            + dataframe['LEGACY_POLICY'] + ' (Seq Code)'
        ),
        (
            'COMMITMENT_TYPE ' + dataframe['COMMITMENT_TYPE']
            + ' does not match LEGACY_POLICY '
            + dataframe['LEGACY_POLICY'] + ' (Seq Code)'
        )
    ]

    return np.select(condlist, choicelist, 'OK')

class DataChecks():
    def __init__(
        self,
        report_date: str,
        symy_filepath: str = None,
        register_filepath: str = None,
        sharepoint_filepath: str = None,
        expiry_dates_filepath: str = None,
    ) -> None:
        """Class to run checks on datasets.

        Parameters
        ----------
        report_date : str
            Date in the format `dd/mm/yyyy`.
        symy_filepath : str, optional
            Path to SYMY extract file, in format `.txt`, by default None
        register_filepath : str, optional
            Path to Treaty Register in excel file, by default None
        sharepoint_filepath : str, optional
            Path to SharePoint extract in excel file, by default None
        expiry_dates_filepath : str, optional
            Path to SYMY Expiry date extract file, in format `.txt`,
            by default None
        """

        self.report_date = pd.to_datetime(report_date, dayfirst=True)
        self.symy_data = None
        self._sharepoint_data = None
        self._register = None
        self.expiry_dates = None

        if symy_filepath is not None:
            self.import_symy_data(symy_filepath)

        if sharepoint_filepath is not None:
            self.import_sharepoint_data(sharepoint_filepath)

        if register_filepath is not None:
            self.register = Register(filepath=register_filepath).register

        if expiry_dates_filepath is not None:
            self.import_expiry_dates_data(expiry_dates_filepath)

        # To be used to export everything to Excel
        self.data_checks_dfs = dict()

    @property
    def register(self) -> pd.DataFrame:
        return self._register

    @register.setter
    def register(self, dataframe: pd.DataFrame) -> None:
        df = dataframe.copy()
        # Drop duplicated Balloon ID's as to keep latest UWY
        # When analysing data, we only need latest entry
        df.drop_duplicates(subset='Balloon ID', keep='first', inplace=True)

        self._register = df
        return None
    
    @property
    def sharepoint_data(self) -> pd.DataFrame:
        return self._sharepoint_data
    
    @sharepoint_data.setter
    def sharepoint_data(self, df: pd.DataFrame) -> None:
        # Convert columns to pd.datetime, then return only date
        date_cols = ['Answer date',
                     'Request date',
                     'Expiry date']

        for col in date_cols:
            df[col] = pd.to_datetime(df[col], dayfirst=True).dt.date

        # Sort by Answer date to keep only latest when removing duplicates.
        df.sort_values(
            by=['BUS ID', 'Balloon ID', 'Answer date'],
            ascending=[True, True, False],
            inplace=True
        )

        try:
            # remove columns that are not used
            cols_to_drop = ['Item Type', 'Path']
            for col_to_drop in cols_to_drop:
                df.drop(columns=col_to_drop, inplace=True)
                print(f"Columns {col_to_drop} removed from analysis.")
        except KeyError:
            # If columns above are not in excel file, then do nothing
            pass

        # Add KEY column
        df['KEY'] = df['Balloon ID'].str.strip() + df['BUS ID']

        # # Add Our Share column
        # df['OUR_EXP_ORG_CURR'] = df['Amount decided'] * df['Atradius Re Share']

        self._sharepoint_data = df

        return None

    def import_sharepoint_data(self, filepath) -> pd.DataFrame:
        df = pd.read_excel(filepath, dtype={'BUS ID': str})

        self.sharepoint_data = df

        return None
    
    def import_symy_data(self, filepath) -> pd.DataFrame:
        df = pd.read_csv(filepath,
                         sep='\t',
                         dtype={'BUYER_NUMBER': str},
                         usecols=SYMY_COLS_TO_USE)

        df['Balloon ID'] = df['LEGACY_POLICY'].str[:10]

        df['KEY'] = df['Balloon ID'] + df['BUYER_NUMBER']

        self.symy_data = df

        return None

    def import_expiry_dates_data(self, filepath) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep='\t', dtype={'BUYER_NUMBER': str})

        df['CLD_EXPIRY_DATE'] = pd.to_datetime(df['CLD_EXPIRY_DATE'],
                                               dayfirst=True)
        df['Balloon ID'] = df['LEGACY_POLICY'].str[:10]

        self.expiry_dates = df

        return None

    def is_currency_same_as_register(self, dataframe) -> pd.Series:
        if self.register is None:
            return "Can't run check. No data found for Register."

        register_cols = ['Balloon ID', 'Curr']

        df = dataframe.merge(self.register[register_cols],
                             on='Balloon ID',
                             how='left')

        return np.where(
            df['POLICY_CURRENCY'] == df['Curr'],
            'OK',
            "SYMY Curr " + df['POLICY_CURRENCY'] + " different than Register " + df['Curr'])

    def is_commit_type_and_upr_valid(self, dataframe: pd.DataFrame) -> pd.Series:
        if self.register is None:
            return "Can't run check. No data found for Register."

        register_cols = ['Balloon ID', 'UPR']

        df = dataframe.merge(self.register[register_cols],
                             on='Balloon ID',
                             how='left')

        filter_upr_for_tty = df['UPR'].str[0] != df['COMMITMENT_TYPE'].str[0]
        filter_upr_for_sa_or_fac = df['UPR'].str[0] != df['COMMITMENT_TYPE'].str[1]

        condlist = [df['UPR'].isna(),
                    (filter_upr_for_tty) & (filter_upr_for_sa_or_fac)]

        choicelist = [
            'UPR Code not available for ' + df['Balloon ID'],
            (
                'COMMITMENT_TYPE ' + df['COMMITMENT_TYPE']
                + ' does not match UPR Code ' + df['UPR']
            )
        ]

        return np.select(condlist, choicelist, default='OK')

    def is_symy_exp_in_sharepoint(self, symy_data: pd.DataFrame) -> pd.Series:
        if self.sharepoint_data is None:
            return "Can't run check. No data found for SharePoint."

        df_sp_key = self.sharepoint_data['KEY']

        filter_not_specials = symy_data['COMMITMENT_TYPE'].str[0] != 'S'

        condlist = [filter_not_specials,
                    symy_data['KEY'].isin(df_sp_key)]

        choicelist = ['OK',
                      'OK']

        msg_if_invalid = 'SYMY SA not in SharePoint. Requires manual check'

        return np.select(condlist, choicelist, msg_if_invalid)

    def is_balloon_id_in_register(self,
                                  balloon_ids: pd.Series) -> pd.Series:

        if self.register is None:
            return "Can't run check. No data found for SharePoint."

        register_balloons = self.register['Balloon ID']

        return np.where(balloon_ids.isin(register_balloons),
                        'OK',
                        'Balloon ID not in Register.')

    @staticmethod
    def _keep_only_error_in_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to keep only rows with error."""
        check_cols = [col for col in dataframe.columns if 'check_' in col]

        # We're going to use eval to build filter. So we start with
        # an empty string
        filter_only_error = ''

        for col in check_cols:
            if filter_only_error == '':
                # If it's first argument, then we add to string
                filter_only_error += f"(dataframe['{col}'] != 'OK')"
            else:
                # Else, we include the "|" (meaning "or")
                filter_only_error += f" | (dataframe['{col}'] != 'OK')"

        return dataframe[eval(filter_only_error)]

    def check_sharepoint_data(self,
                              keep_only_error: bool = False) -> pd.DataFrame:
        """Perform checks on data extracted from SharePoint.

        Notes
        -----
        - Check 1: Amount decided is 0 or NaN.
        - Check 2: Empty BUS ID in SharePoint data.
        - Check 3: Duplicated entries. It also tags the latest entry.
        - Check 4a: Empty spaces in beginning or end of Balloon ID.
        - Check 4b: Empty spaces in beginning or end of BUS ID.
        - Check 5: Checks for Balloon ID pattern:
          - (5 digits, BC, 1 digit, 1 blank space, 2 digits)
        - Check 6: Balloon ID exists in Register.
        """
        if self.sharepoint_data is None:
            return "Can't run checks. No data found for SharePoint."

        df = self.sharepoint_data.copy()

        # Check 1 - Invalid amounts
        df['check_1'] = is_amount_na_or_zero(df['Amount decided'])

        # Check 2 - Invalid BUS ID in SharePoint data.
        df['check_2'] = is_na(df['BUS ID'])

        # Check 3 - Duplicated entries, tagging the latest
        df['check_3'] = is_duplicated(df['KEY'], tag_latest=True)

        # Check 4 - Blank spaces in BUS ID or Balloon ID
        df['check_4a'] = has_blank_space_in_beg_or_end(df['Balloon ID'])
        df['check_4b'] = has_blank_space_in_beg_or_end(df['BUS ID'])

        # Check 5 - Correct Balloon ID pattern
        df['check_5'] = is_balloon_id_pattern_correct(df['Balloon ID'])

        # Check 6 - Balloon ID exists in Register
        df['check_6'] = self.is_balloon_id_in_register(
            balloon_ids=df['Balloon ID'])

        if keep_only_error:
            df = self._keep_only_error_in_dataset(df)

        self.data_checks_dfs['SharePoint_checks'] = df

        return df

    def check_symy_data(self, keep_only_error: bool = False) -> pd.DataFrame:
        """Return DataFrame with check columns.

        Notes
        -----
        - Check 1: Customer first 5 digits are the same as Legacy Policy.
        - Check 2: CLD_TOTAL_AMOUNT is zero or empty.
        - Check 3:
          - COMMITMENT_TYPE beginning in "S" belongs to
            LEGACY_POLICY ending with "SA".
          - COMMITMENT_TYPE beginning in "F" belongs to
            LEGACY_POLICY SEQ(2nd char) = "3".
        - Check 4: Balloon ID (LEGACY_POLICY 1st 10 digits) is in Register.
        - Check 5: COMMITMENT_TYPE is valid according to UPR Code in Register.
        - Check 6: SYMY Currency the same as Register.
        - Check 7: SYMY Exposure is in SharePoint.

        Returns
        -------
        pd.DataFrame
            DataFrame with check columns.
        """
        if self.symy_data is None:
            return "Can't run checks. No data found for SYMY."
        df = self.symy_data.copy()

        df['check_1'] = is_customer_same_as_legacy_policy(
            df[['CUSTOMER_NAME', 'LEGACY_POLICY']])

        df['check_2'] = is_amount_na_or_zero(df['CLD_TOTAL_AMOUNT'])

        df['check_3'] = is_commit_type_and_leg_pol_valid(
            df[['COMMITMENT_TYPE', 'LEGACY_POLICY']])

        df['check_4'] = self.is_balloon_id_in_register(df['Balloon ID'])

        df['check_5'] = self.is_commit_type_and_upr_valid(dataframe=df)

        df['check_6'] = self.is_currency_same_as_register(dataframe=df)

        df['check_7'] = self.is_symy_exp_in_sharepoint(symy_data=df)

        if keep_only_error:
            df = self._keep_only_error_in_dataset(df)

        self.data_checks_dfs['SYMY_checks'] = df

        return df

    def check_symy_duplicated_data(self) -> pd.DataFrame:
        if self.symy_data is None:
            return None

        df = self.symy_data
        # Sometimes exposures are duplicated, but CLD is 0.
        # This removes false positives.
        # It has to be done before checking for duplicates
        filter_positive_amounts = df['CLD_TOTAL_AMOUNT'] > 0
        df_positive_amt = df.loc[filter_positive_amounts]

        # Filter to keep only duplicated amounts
        filter_duplicated = df_positive_amt.duplicated(
            subset=['BUYER_NUMBER', 'Balloon ID'],
            keep=False)

        df_filtered = df_positive_amt.loc[filter_duplicated].copy()

        df_filtered.sort_values(by=['BUYER_NUMBER', 'LEGACY_POLICY'],
                                inplace=True)

        self.data_checks_dfs['SYMY_DUPLICATED'] = df_filtered

        return df_filtered

    def check_symy_expiry_date_check(
        self,
        keep_only_error: bool = False
    ) -> pd.DataFrame:
        "Checks if Expiry date in SYMY is the same as Treaty Register and SP."
        if self.expiry_dates is None:
            print("No Expiry dates loaded. Checks skipped.")
            return None
        
        df_exp_dates = self.expiry_dates.copy()

        # Need to check Fac's ans SA's in the Expiry Dates extract
        filter_ct_fac = df_exp_dates['COMMITMENT_TYPE'].str[0] == 'F'
        filter_leg_pol_fac = df_exp_dates['LEGACY_POLICY'].str[6] == '3'
        filter_ct_sa = df_exp_dates['COMMITMENT_TYPE'].str[0] == 'S'

        df_facs = df_exp_dates.loc[
            (filter_ct_fac) | (filter_leg_pol_fac) | (filter_ct_sa)
        ]
        
        df = df_facs.merge(self.register[['Balloon ID', 'Pd Beg', 'Pd End']],
                           on='Balloon ID',
                           how='left')

        # Check Pd End and CLD_EXPIRY_DATE
        df['check_1'] = np.where(df['Pd End'] != df['CLD_EXPIRY_DATE'],
                                 'CLD_EXPIRY_DATE and Pd End not matching',
                                 'OK')

        df['check_2'] = np.where(df['Pd Beg'] > self.report_date,
                                 'Exposure starting after Reporting date',
                                 'OK')

        if keep_only_error:
            df = self._keep_only_error_in_dataset(df)

        self.data_checks_dfs['SYMY_FAC_EXPIRY_DATES'] = df

        return df

    def check_treaty_register(self,
                              min_uw_yr: int = None,
                              keep_only_error: bool = False) -> pd.DataFrame:
        if self.register is None:
            print("No Register loaded. Checks skipped.")
            return None

        df = self.register[REGISTER_COLS_TO_USE].copy()

        if min_uw_yr is None:
            # If None, then Filter only to keep previous year
            previous_year = pd.to_datetime('now', utc=True).year - 1

            # Since we just need the last 2 digits, we slice it
            # and convert to int so we are able to filter dataframe
            min_uwy = int(str(previous_year)[-2:])
            df = df.loc[df['UW Yr'] > min_uwy].copy()
        elif not isinstance(min_uw_yr, int):
            # If uw_year is not int
            err_msg = "uw_year takes only an int argument."
            err_msg += f"Type {type(min_uw_yr)} was passed."
            raise TypeError(err_msg)
        else:
            current_year = int(pd.to_datetime('now', utc=True).strftime("%y"))
            if min_uw_yr >= current_year:
                err_msg = "Please use only uw_year=20 or lower. "
                err_msg += f"{min_uw_yr} was passed."
                raise ValueError(err_msg)
            else:
                df = df.loc[df['UW Yr'] > min_uw_yr].copy()

        filter_only_fac = df['Type/Form'].isin(['FAC', 'FACX'])
        filter_empty_bus_id = df['Bus Id'].isna()

        df['check_1'] = np.where((filter_only_fac) & (filter_empty_bus_id),
                                 'FAC and BUS ID column is empty.',
                                 'OK')

        if keep_only_error:
            df = self._keep_only_error_in_dataset(df)

        self.data_checks_dfs['REGISTER_checks'] = df

        return df

    def run_checks(self,
                   keep_only_error: bool = False,
                   to_excel: bool = False,
                   open_file: bool = False):
        """Generate check report.

        Parameters
        ----------
        keep_only_error : bool, optional
            If True, then keep only rows with errors,
            else include all rows, by default False.
        to_excel : bool, optional
            If True, then export report to Excel, by default False
        open_file : bool, optional
            If True, then open Excel report generated, by default False
        """
        self.check_sharepoint_data(keep_only_error=keep_only_error)
        self.check_symy_data(keep_only_error=keep_only_error)
        self.check_symy_duplicated_data()
        self.check_symy_expiry_date_check(keep_only_error=keep_only_error)
        self.check_treaty_register(keep_only_error=keep_only_error)

        if to_excel:
            self.checks_to_excel(open_file=open_file)

    def checks_to_excel(self, open_file: bool = False) -> None:
        """Generate Excel report for checks.

        Parameters
        ----------
        open_file : bool, optional
            If True, then open Excel file., by default False

        Returns
        -------
        None
        """
        file_name = 'Pre-run Data checks-produced on {}.xlsx'.format(
            pd.to_datetime('now', utc=True).strftime('%Y%m%d-%Hh%M')
        )

        writer = pd.ExcelWriter(file_name,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')
        wb = writer.book

        self._add_notes_sheet(excel_writer=writer)

        for sheetname, dataframe in self.data_checks_dfs.items():
            dataframe.to_excel(writer, index=False, sheet_name=sheetname)

            if dataframe.shape[0] > 0:
                worksheet = writer.sheets[sheetname]

                # Get the dimensions of the dataframe.
                (max_row, max_col) = dataframe.shape

                # Create a list of column headers, to use in add_table().
                column_settings = [
                    {'header': column} for column in dataframe.columns
                ]

                # Add the Excel table structure. Pandas will add the data.
                worksheet.add_table(
                    0, 0, max_row, max_col - 1, {'columns': column_settings,
                                                 'name': sheetname}
                )

        # ---------------------------------------------------------------------
        # Formatting

        # Get the xlsxwriter workbook and worksheet objects.
        num_format = wb.add_format({'num_format': '#,##0'})
        pct_format = wb.add_format({'num_format': '0.00%'})

        try:
            # SharePoint Formatting
            ws_sp = writer.sheets['SharePoint_checks']
            ws_sp.set_column('A:B', 12)
            ws_sp.set_column('C:C', 18, num_format)
            ws_sp.set_column('D:D', 19)
            ws_sp.set_column('E:F', 14)
            ws_sp.set_column('G:G', 18)
            ws_sp.set_column('H:H', 20, num_format)
            ws_sp.set_column('I:I', 11)
            ws_sp.set_column('J:J', 6)
            ws_sp.set_column('K:L', 15)
            ws_sp.set_column('M:M', 19)
            ws_sp.set_column('N:T', 13)
            ws_sp.set_column('P:P', 23)  # Duplicated entries column

        except KeyError:
            pass

        # Set the column width and format.
        try:
            ws_symy_checks = writer.sheets['SYMY_checks']
            ws_symy_checks.set_column('A:B', 18)
            ws_symy_checks.set_column('C:C', 35)
            ws_symy_checks.set_column('D:D', 22.46)
            ws_symy_checks.set_column('E:E', 16.86)
            ws_symy_checks.set_column('F:G', 21.86)
            ws_symy_checks.set_column('H:I', 11.43)
            ws_symy_checks.set_column('J:J', 22.14, num_format)
            ws_symy_checks.set_column('K:K', 28.43, num_format)
            ws_symy_checks.set_column('L:M', 19.71)
            ws_symy_checks.set_column('N:N', 14.14)
            ws_symy_checks.set_column('O:O', 11.57)
            ws_symy_checks.set_column('P:X', 15.71)
        except KeyError:
            pass

        try:
            ws_reg_checks = writer.sheets['REGISTER_checks']
            ws_reg_checks.set_column('A:C', 8.29)
            ws_reg_checks.set_column('D:D', 32)
            ws_reg_checks.set_column('F:J', 12.29)
            ws_reg_checks.set_column('K:M', 7.14)
            ws_reg_checks.set_column('N:N', 23.43)
            ws_reg_checks.set_column('O:O', 10.57, pct_format)
            ws_reg_checks.set_column('P:P', 9.43)
            ws_reg_checks.set_column('Q:Q', 15.71)
        except KeyError:
            pass

        writer.save()

        print('File "{}" created.'.format(file_name))

        if open_file:
            os.startfile(file_name)

        return None

    def _add_notes_sheet(self, excel_writer):
        wb = excel_writer.book
        ws_notes = wb.add_worksheet('Notes')

        #ws_notes = excel_writer.sheets['Notes']
        ws_notes.set_column('A:A', 6)
        ws_notes.set_column('B:B', 84)
        ws_notes.set_column('C:C', 11)

        title_format = wb.add_format({'bold': True,
                                      'font_color': '#1f497d',
                                      'font_size': 16})

        # (Text , Format) if Title, or just Text
        info = [
            ('SharePoint Checks', title_format),
            'check_1: Amount decided is 0 or NaN.',
            'check_2: Empty BUS ID in SharePoint data.',
            'check_3: Duplicated entries. It also tags the latest entry.',
            'check_4a: Empty spaces in beginning or end of Balloon ID.',
            'check_4b: Empty spaces in beginning or end of BUS ID.',
            'check_5: Checks for Balloon ID pattern - 5 digits, BC, 1 digit, 1 blank space, 2 digits',
            'check_6: Balloon ID exists in Register.',
            
            ('SYMY Checks', title_format),
            'check_1: Checks if the first 5 numbers of Customer name match LEGACY_POLICY.',
            'check_2: Checks if CLD_TOTAL_AMOUNT is zero or empty.',
            'check_3:',
            '    Checks if COMMITMENT_TYPE = "S" belongs to LEGACY_POLICY ending with "SA".',
            '    Checks if COMMITMENT_TYPE = "F" belongs to LEGACY_POLICY SEQ(2nd char) = "3".',
            'check_4: Checks if Balloon ID in SYMY exists in Register.',
            'check_5: Checks if COMMITMENT_TYPE in SYMY matches UPR Code in Register.',
            'check_6: Checks if SYMY Currency the same as Register.',
            'check_7: Check if SYMY entry has a match in SharePoint.',
            
            ('SYMY DUPLICATED', title_format),
            'List of all duplicated exposures using BUYER_NUMBER and Balloon ID.',

            ('SYMY Expiry Dates', title_format),
            'check_1: Checks if Expiry date in SYMY and Register are matching.',
            f'check_2: Checks if Pd Beg is before reporting date ({self.report_date}).',
            
            ('REGISTER Checks', title_format),
            'check_1: Checks if Type/Form = "FAC" has BUS ID.',
        ]

        section = ''
        row_range = 1
        for i in info:
            if isinstance(i, tuple):
                # If tuple, it means that it's a tile
                row_range += 2
                ws_notes.write(f"A{row_range}", i[0], i[1])
                section = i[0]
                if i[0].startswith('SharePoint'):
                    ws_notes.write(f'C{row_range}', 'Error Count')
            else:
                # Else, it's text
                row_range += 1
                ws_notes.write(f"B{row_range}", i)
                if i.startswith('check_'):
                    if section.startswith('SharePoint'):
                        # Write count errors formula
                        ws_notes.write(f"C{row_range}", f'=COUNTIF(INDIRECT("SharePoint_checks["&LEFT(B{row_range},FIND(":",B{row_range})-1)&"]"),"<>OK")')
                    elif section.startswith('SYMY Checks'):
                        ws_notes.write(f"C{row_range}", f'=COUNTIF(INDIRECT("SYMY_checks["&LEFT(B{row_range},FIND(":",B{row_range})-1)&"]"),"<>OK")')
                        if i.startswith('check_3'):
                            ws_notes.write(f"D{row_range}", 'does not affect SII run')
                        elif i.startswith('check_5'):
                            ws_notes.write(f"D{row_range}", 'does not affect SII run')
                    elif section.startswith('REGISTER'):
                        ws_notes.write(f"C{row_range}", f'=COUNTIF(INDIRECT("REGISTER_checks["&LEFT(B{row_range},FIND(":",B{row_range})-1)&"]"),"<>OK")')
                elif i.startswith('List of all duplicated exposures'):
                    ws_notes.write(f"C{row_range}", '=COUNTIF(SYMY_DUPLICATED!A:A,"<>")-1')
                

