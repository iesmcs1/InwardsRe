# Buyer Monitoring Report

## 5 W's

### What
This report is gives an overview of the portfolio movements regarding known exposure (named buyers) only. This reports is produced in HTML. This means that it opens in any modern browser.

To **view the report, an internet connection is required**. In order to produce the report with the smallest file size possible, the libraries that make it interactive are loaded through the internet. This is why the user needs internet connection.

This report **does not** contain any information regarding treaty's unknown and total TPE, ECap and PD.

### When
This report is produced at the beginning of every month, after XXXXXXXXXXXXX.

### Who
The code contained in this repository is design in a way to make it easy for anyone with basic coding skills to generate the report.

The only thing that needs to be updated in the Jupyter Notebook file `Buyer_Rating_Report.ipynb` are the file names to be read and loaded by the code, and used to produce the html report.

# Code Structure

The code is divided into 2 main files:
- `buyer_rating_report.py`
- `buyer_report_output.py`

## Buyer Rating Report File
This file contains the code to:
- process the data for each run
  - Class `NamedExposureData`: Responsible for dealing with queries for a specific run (e.g. 2021-April)
  - Class `NamedExposure`: class used to aggregate all named exposure data
  - Class `CompareNamedExposure`: produces an Excel output comparing 2 named exposure data.

## Buyer Report Output File
The main purpose of this file is to hold the class `BuyerReportHTML` that will produce all the necessary graphs and tables, and generate the HTML output. This makes it easier to make changes to the HTML output.