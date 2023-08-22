import pandas as pd
import numpy as np
from treaty_register import Register
from utils import get_report_period

REGISTER_COLS = ['REPORTING_PERIOD',
                 'Comp',
                 'CONTRACT_ID',
                 'Run-off',
                 'EPI is Rev EPI or EPI',
                 'YearFrac',
                 'EPI']


class EcapDashboards:
    def __init__(self) -> None:
        self.so_report = None
        self.register = None
        self.cedant_name = None
        
        # _report_date is used only when importing run data
        self._report_date = None
    
    def import_cedant_name(self, filepath: str) -> None:
        """[summary]

        Notes
        -----
        File containing Cedant names needs to be in the following format:  
        
        ```
        <Comp><"\\t"><Cedant Name>
        e.g.:
        11111	Some Cedant
        ```

        This function will add the list of cedant names to self.cedant_name

        Parameters
        ----------
        filepath : str
            Absolute path to txt file (Tab-separated Values).

        """
        df = pd.read_csv("cedant_names.txt",
                         sep='\t',
                         dtype={'Comp':str})
        # Set Comp as index to use map when merging cedant names
        df.index = df['Comp']
        self.cedant_name = df['CedantName']
        
        return None
    
    def import_run_data(self,
                        so_report_filepath: str,
                        register_filepath: str = None,
                        epi_filepath: str = None) -> None:
        
        if register_filepath is not None and epi_filepath is not None:
            raise ValueError("Function can only import Register or EPI file. Both were passed.")
        elif register_filepath is not None:
            self._import_process_register(register_filepath)
        elif epi_filepath is not None:
            pass


        # Import SO_REPORT data into class
        self._import_process_so_report(so_report_filepath)
        
        

        # Call function to make sure data is sorted chronological order.
        self._sort_reporting_period()

        print("Data for {} added to class.".format(
            get_report_period(self._report_date, full_year_str=True)
        ))

        # Set report_date to None, to avoid errors
        self._reset_report_date()

        return None

    def _import_process_so_report(self, filepath: str) -> None:
        # Import file into variable
        df = pd.read_csv(filepath, sep='\t', dtype={'CALCRUN': str})
        # Timestamp column
        df['REPORTING_PERIOD'] = pd.to_datetime(df['REPORTING_PERIOD'])
        df['PD*TPE'] = df['ULTIMATE_POD'] * df['EXP_GROSS_OF_RETRO']

        self._set_report_date_of_imported_data(df)
        
        if self.so_report is None:
            # If 1st time importing, then set so_report to df
            self.so_report = df
        else:
            # else concat to existing dataframe in so_report
            # drop_duplicates in case I import same data twice.
            self.so_report = pd.concat([self.so_report, df]).drop_duplicates()
        
        return None
    
    def _set_report_date_of_imported_data(self, dataframe) -> None:
        # Only used when importing run data, to Timestamp Register
        self._report_date = dataframe['REPORTING_PERIOD'][0]
    
    def _reset_report_date(self) -> None:
        self._report_date = None

    def _import_process_register(self, filepath: str) -> None:
        # Check if filepath is correct
        self._check_register_filepath(filepath=filepath)
        # Calling .register in Register class because we only need 
        # the final dataframe.
        df = Register(filepath,
                      register_type='assumed',
                      report_date=self._report_date,
                      # need to import with run-off rates to calculate
                      # correct EPI
                      get_treaty_run_off=True,
                      timestamp=True).register
        
        # Include Year Fraction calculation
        df['YearFrac'] = self.calculate_treaty_year_fraction(df)

        # Calculate EPI. This takes into consideration treaties that are over
        # 1 year. If Treaty is 3 years long, then EPI will be divided by 3
        df['EPI'] = df['EPI is Rev EPI or EPI'] * df['Run-off'] / df['YearFrac']

        df = df[REGISTER_COLS]

        if self.register is None:
            # If 1st time importing, then set so_report to df
            self.register = df
        else:
            # else concat to existing dataframe in so_report
            # drop_duplicates in case I import same data twice.
            self.register = pd.concat([self.register, df]).drop_duplicates(
                subset=['REPORTING_PERIOD', 'CONTRACT_ID']
            )
        
        return None
    
    def import_process_epi(self, filepath: str) -> None:
        pass
    
    @staticmethod
    def calculate_treaty_year_fraction(df: pd.DataFrame) -> pd.Series:
        s = (df['Pd End'] - df['Pd Beg']) / np.timedelta64(1, 'Y')
        return s.round(decimals=2)

    def _check_register_filepath(self, filepath):
        "Check if the pattern YYYYQXX is in filepath"
        period = get_report_period(self._report_date, full_year_str=True)
        if period not in filepath:
            raise ValueError("File name {} does not contain {}.".format(
                filepath, period
            ))

    def _sort_reporting_period(self):
        "Order data in chronological order."
        self.so_report = self.so_report.sort_values(by='REPORTING_PERIOD')
        self.register = self.register.sort_values(by='REPORTING_PERIOD')

    def get_overall_movements(self, sub_type: str = None):
        """Return sum of Exposure, EPI EL and ECap.

        Parameters
        ----------
        sub_type : str, optional
            'credit' or 'bond', by default None
        """
        if sub_type is not None and sub_type not in ['credit', 'bond']:
            raise ValueError("sub_type needs to be 'bond' or 'credit'.")
        df = self.so_report.copy()
        df['MODEL_SUB_TYPE'] = df['MODEL_SUB_TYPE'].str[0]
        if type(sub_type) is str:
            # Apply filter
            sub_type_filter = df['MODEL_SUB_TYPE'] == sub_type[0].upper()
            df = df.loc[sub_type_filter]
        
        df = df.drop('ULTIMATE_POD', axis=1)
        df = df.groupby(
            by=['REPORTING_PERIOD', 'CONTRACT_ID']
        ).sum().reset_index()
        
        df = df.merge(self.register,
                      on=['REPORTING_PERIOD','CONTRACT_ID'],
                      how='left')
        
        # Group everything using sum
        df_grouped = df.groupby(by=['REPORTING_PERIOD']).sum()
        
        # Calculates Weighted PD. I realized that I just need PD*TPE, then the
        # Weighted PD can be calculated at any stage above buyer level
        # granularity
        df_grouped['PD'] = (df_grouped['PD*TPE'] /
                            df_grouped['EXP_GROSS_OF_RETRO'])
        
        df_grouped['ECap / TPE'] = (df_grouped['EC_CONSUMPTION_ND']
                                    / df_grouped['EXP_GROSS_OF_RETRO'])
        
        df_grouped['ECap / EPI'] = (df_grouped['EC_CONSUMPTION_ND']
                                    / df_grouped['EPI'])
        
        df_grouped['EL / TPE'] = (df_grouped['EXPECTED_LOSS']
                                  / df_grouped['EXP_GROSS_OF_RETRO'])
        
        df_grouped['EL / EPI'] = df_grouped['EXPECTED_LOSS'] / df_grouped['EPI']
        
        reoder_cols = ['EXP_GROSS_OF_RETRO',
                       'EPI',
                       'EC_CONSUMPTION_ND',
                       'ECap / TPE',
                       'ECap / EPI',
                       'PD',
                       'EXPECTED_LOSS',
                       'EL / TPE',
                       'EL / EPI']
        
        rename_cols = {'EXP_GROSS_OF_RETRO': 'Exposure (€m)',
                       'EPI': 'EPI (€m)',
                       'EC_CONSUMPTION_ND': 'ECap (€m)',
                       'EXPECTED_LOSS': 'EL (€m)'}

        df_grouped = df_grouped[reoder_cols].rename_axis(None)

        df_grouped.columns.name = 'Overall' if sub_type is None else sub_type.capitalize()

        return df_grouped.rename(columns=rename_cols)
    
    def generate_dashboard_excel_files(self) -> None:
        pass

        
    def generate_movement_to_file(self):
        overall_mov_file = "Movement to {}.xlsx".format(
            get_report_period(self.so_report['REPORTING_PERIOD'].max(),
                              full_year_str=True)
        )

        writer = pd.ExcelWriter(overall_mov_file,
                                engine='xlsxwriter',
                                date_format='dd/mm/yyyy',
                                datetime_format='dd/mm/yyyy')

        sh_name = 'Overall_Movements'
        
        startrow = 0
        overall_mov_options = [None, 'bond', 'credit']
        
        for op in overall_mov_options:
            df = self.get_overall_movements(op)
            # rename index to format YYYYQXX for readability
            df.index = df.index.map({
                x: get_report_period(x, full_year_str=True) for x in df.index
            })
            df.to_excel(
                writer,
                sheet_name=sh_name,
                startrow=startrow
            )

            writer.sheets[sh_name].write(f'A{startrow+1}', df.columns.name)

            # shape attribute is always lenght of (row, col)
            startrow += 3 + df.shape[0]
            # make sure df is None
            df = None
        
        # Get the xlsxwriter objects from the dataframe writer object.
        workbook  = writer.book 
        worksheet = writer.sheets[sh_name]

        worksheet.hide_gridlines(2)

        #-----------------------------------------------------------------------
        # Format numbers

        # Add some cell formats.
        format_num_in_m_0f = workbook.add_format({'num_format': '#,##0,,'})
        format_num_in_m_1f = workbook.add_format({'num_format': '#,##0.0,,'})
        format_pct_0f = workbook.add_format({'num_format': '0%'})
        format_pct_2f = workbook.add_format({'num_format': '0.00%'})

        # Format ranges that are numbers
        worksheet.set_column('B:B', 14, format_num_in_m_0f)
        
        for col in ['C:D', 'H:H']:
            worksheet.set_column(col, 14, format_num_in_m_1f)
        
        # Format ranges that are percentages with 2 decimals
        for col in ['E:E', 'G:G', 'I:I']:
            worksheet.set_column(col, 14, format_pct_2f)
        
        # Format ranges that are percentages with 0 decimals
        for col in ['F:F', 'J:J']:
            worksheet.set_column(col, 14, format_pct_0f)
        

        #-----------------------------------------------------------------------
        # Insert Ecap, EL and TPE chart

        # Create a chart object.
        line_chart = workbook.add_chart({'type': 'line'})

        # Configure the series of the chart from the dataframe data.
        line_chart.add_series({
            'name': f'={sh_name}!D1',
            'categories': f'={sh_name}!A3:A5',
            'values': f'={sh_name}!D3:D5',
            'marker': {'type': 'automatic'},
            'data_labels': {'value': True,
                            'position': 'above',
                            'num_format': '#,##0.0,,'}  # val in millions
        })

        line_chart.add_series({
            'name': f'={sh_name}!H1',
            'categories': f'={sh_name}!A3:A5',
            'values': f'={sh_name}!H3:H5',
            'marker': {'type': 'automatic'},
            'data_labels': {'value': True,
                            'position': 'below',
                            'num_format': '#,##0.0,,'}  # val in millions
        })

        column_chart = workbook.add_chart({'type': 'column'})

        # Configure the data series for the primary chart.
        column_chart.add_series({
            'name':       f'={sh_name}!B1',
            'categories': f'={sh_name}!A3:A5',
            'values':     f'={sh_name}!B3:B5',
            'y2_axis':    True,
            'data_labels': {'value': True,
                            'position': 'inside_base',
                            'num_format': '#,##0.0,,'}  # val in millions
        })

        combo_chart_title = 'ECap (€m), Expected Loss (€m) and '
        combo_chart_title += 'related Exposure (€m) development'

        line_chart.set_title({'name': combo_chart_title})
        line_chart.set_y_axis({'major_gridlines': {'visible': False}})
        line_chart.set_size({'width': 600, 'height': 400})
        line_chart.set_legend({'position': 'bottom'})

        # Combine the charts.
        line_chart.combine(column_chart)

        # Insert the chart into the worksheet.
        worksheet.insert_chart('I22', line_chart)

        #-----------------------------------------------------------------------
        # Insert Ratio Chart

        # Create a chart object.
        ratio_chart = workbook.add_chart({'type': 'line'})

        ratio_chart.add_series({
            'name': f'={sh_name}!E1',
            'categories': f'={sh_name}!A3:A5',
            'values': f'={sh_name}!E3:E5',
            'marker': {'type': 'automatic'},
            'data_labels': {'value': True,
                            'position': 'above'}
        })

        ratio_chart.add_series({
            'name': f'={sh_name}!I1',
            'categories': f'={sh_name}!A3:A5',
            'values': f'={sh_name}!I3:I5',
            'marker': {'type': 'automatic'},
            'data_labels': {'value': True,
                            'position': 'below'}
        })

        ratio_chart.set_title({'name': 'ECap/TPE and EL/TPE ratio development'})
        ratio_chart.set_size({'width': 600, 'height': 400})
        ratio_chart.set_y_axis({'major_gridlines': {'visible': False}})
        ratio_chart.set_legend({'position': 'bottom'})

        # Insert the chart into the worksheet.
        worksheet.insert_chart('B22', ratio_chart)

        writer.save()

        print(f"{overall_mov_file} file created.")
            




