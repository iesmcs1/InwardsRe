import pandas as pd

def import_fc_computed_amounts(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, sep='\t', index_col=0)
        df.index = pd.to_datetime(df.index, dayfirst=True)

        # we only need the 2 columns below
        return df[['COMPUTED_AMT_EUR']]
    except FileNotFoundError:
        print(f"FC_COMPUTED_AMOUNT not found in\n{filepath}.\n")
        return None

# class FCComputedAmounts:
#     def __init__(self, folder_path) -> None:
#         self.FC_COMPUTED = None
#         self.folder_path = folder_path

#         self.import_fc_computed_amounts()

#     def import_fc_computed_amounts(self):
#         "Import FC_COMPUTED_AMOUNTS in folder."

#         filepath = os.path.join(self.folder_path, 'FC_COMPUTED_AMOUNTS.txt')
#         try:
#             # Use first column as index, as it is REPORTING_DAT
#             # then convert index to datetime, so that it can be merged into
#             # SCR summary table.
#             df = pd.read_csv(filepath, sep='\t', index_col=0)
#             df.index = pd.to_datetime(df.index, dayfirst=True)

#             self.FC_COMPUTED = df['COMPUTED_AMT_EUR'][0]
#             print("FC_COMPUTED_AMOUNTS.txt loaded to model.")
#         except FileNotFoundError:
#             # if file does not exist
#             print(f"{filepath} file not found.\n")

#     def is_verified_amount(self) -> bool:
#         if self.FC_COMPUTED is None:
#             return False
#         else:
#             return True
