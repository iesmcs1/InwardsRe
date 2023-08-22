from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from typing import Tuple
import pandas as pd


class ReinsuranceGroupings:
    def __init__(self) -> None:
        self.ceded_register = None
        self.master_file = None
        self.txt_file = None
        self.ok_to_proceed = False

    def import_and_process_register(self, filepath: str):
        df = pd.read_excel(filepath, header=None)
        self.ceded_register = self._process_register(df)

    @staticmethod
    def _process_register(df: pd.DataFrame) -> pd.DataFrame:
        # Remove excess Lines where 2nd column is NaN
        df = df.loc[~df[1].isna()].copy()

        # Reset index so that I use first row as header
        df.reset_index(drop=True, inplace=True)
        df.columns = df.iloc[0]

        # No need for row with column headers
        df.drop(df.index[0], inplace=True)

        df['UW Yr'] = df['UW Yr'].astype(int)
        df['UWY_str'] = df['UW Yr']
        df['UWY_str'] = df['UWY_str'].apply(lambda x: '{:0>2}'.format(x))

        # There is one case with a space. RT16 08 -> RT1608
        df['Seq'] = df['Seq'].str.replace(" ", "")

        # Add leading zero to for Reference column
        df['Ceded Ref'] = df['Seq'] + df['UWY_str']

        # Remove Actuarial Reserves

        # These are false positives. Remove from data
        remove_keywords = 'A.RE|Actuarial Reserve|CEDED BULK RESVERS|ATRADIUS ACTUARIAL R'
        filter_short_col = df['Short'].str.contains(remove_keywords)

        return df[~filter_short_col]

    def import_and_process_master_file(self, filepath: str):
        df = pd.read_excel(filepath, skiprows=1)
        self.master_file = self._process_master_file(df)

    @staticmethod
    def _process_master_file(df: pd.DataFrame) -> pd.DataFrame:
        # Exclude rows that are headers
        filter_column_headers_out = df['Agreement Name'] != 'Agreement Name'

        # apply filters
        df = df.loc[(filter_column_headers_out)].copy()

        df['Agreement Start Date'] = pd.to_datetime(
            df['Agreement Start Date'], dayfirst=True)
        
        # Ultimately, we're checking if Retro has Grouping ID, because that's
        # what's in the txt file. ffill is forward fill and makes sure
        # that all agreements has the same grouping ID
        df['Grouping ID'].ffill(inplace=True)
        df['Grouping Name'].ffill(inplace=True)

        # Return all rows and remove columns where all rows are NaN
        # In this case, we check if all rows are NaN using isna().all().
        # If all rows are NaN, it will return True, so we negate it using ~
        return df.loc[:, ~df.isna().all()]
    
    def import_txt_file(self):
        pass

    def check_register_in_master_file(self, oldest_uwy: int):
        df_reg = self.ceded_register
        df_mas = self.master_file

        # Filter UW Year in Ceded Register
        df_reg = df_reg.loc[df_reg['UW Yr'] >= oldest_uwy].copy()

        # Filter to keep only cases in Master File not in Ceded Register
        reg_in_master = ~df_reg['Ceded Ref'].isin(df_mas['RT Ref'])

        return df_reg.loc[reg_in_master]

    def is_ok_to_proceed(self, oldest_uwy: int):
        df = self.check_register_in_master_file(oldest_uwy=oldest_uwy)
        if df.shape[0] > 0:
            print('## From Register, missing in Master File:')
            for _, row in df.iterrows():
                print("Cedant: ", row['Cedant'])
                print("Short: ", row['Short'])
                print("Ceded Ref: ", row['Ceded Ref'])
            print("Can't proceed.")
        else:
            self.ok_to_proceed = True
    
    def to_excel():
        pass
    
def is_valid_year(year: str) -> bool:
    try:
        # try to convert to int
        int(year)
    except ValueError:
        # if can't convert to int, then invalid
        print("Invalid number.")
        return None

    short_year_format = 2
    
    # Convert full year to only last 2 digits
    current_year = int(str(pd.to_datetime('now').year)[-2:])
    if len(year) != 2:
        # if uwy has more or less than 2 digits, then not valid
        print("Year has to have 2 digits (e.g.: 20, 18, etc.).")
        return False
    elif int(year) > current_year:
        # if uwy is greated than current year, not valid
        print(f"Year greater than current year ({current_year})")
        return False
    
    return True

def ask_uw_year() -> int:
    valid_year = False
    while not valid_year:
        # ask user for Oldest UW Year
        oldest_uwy = input(
            "What is the oldest UW Year that should be investigated? ")
        valid_year = is_valid_year(oldest_uwy)
    
    return int(oldest_uwy)



def select_file(filename: str,
                filetype: Tuple[str],
                check_str: str = None) -> str:
    """[summary]

    Parameters
    ----------
    filename : str
        file name for user
    filetype : Tuple[str]
        file types to filter dialog box
    check_str : str
        string that has to be inside selected file.

    Returns
    -------
    str
        full path to selected file.
    """
    # we don't want a full GUI, so keep the root window from appearing
    Tk().withdraw()

    # show an "Open" dialog box and return the path to the selected file
    window_title = f'Please select latest {filename} file'
    return askopenfilename(title=window_title, filetypes=filetype)


def main():
    rg = ReinsuranceGroupings()

    excel_file_type = [("Excel files", ".xlsx .xls")]

    # Ask user to select latest Ceded Register
    ceded_reg_filepath = select_file(filename="Ceded Register",
                                     filetype=excel_file_type)

    # Load Register into Class
    rg.import_and_process_register(ceded_reg_filepath)

    # Ask user to select latest Master File
    master_file_filepath = select_file(filename="Master File",
                                       filetype=excel_file_type)

    # Load Master File into Class
    rg.import_and_process_master_file(master_file_filepath)

    oldest_uwy = ask_uw_year()

    # This makes sure we check that all Retro's are in the Master File.
    rg.is_ok_to_proceed(oldest_uwy)

    if not rg.ok_to_proceed:
        print("Can't proceed.")
        return None 
    
    while True:
        print("Main menu:")
        print("1 - Generate excel output.")
        print("9 - exit\n")
        asn = input("Please select one option from the menu above: ")
        if asn == "1":
            print("Generating excel output.")
        elif asn == "9":
            print("Exiting...")
            return None
        else:
            print("Invalid menu option. Please try again.")

if __name__ == "__main__":
    main()
