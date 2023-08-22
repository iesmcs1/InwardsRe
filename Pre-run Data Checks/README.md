# Data Checks for Solvency II

Every 3 months, before quarter end (31/03, 30/06, 30/09 or 31/12) we need to check if the data in our systems are correct. To do that, we need to follow the steps below:

## Step 1 - Download datasets

### Step 1a - Download SYMY Main and SYMY Expiry dates

These are 2 datasets that you can download from the same source. They are just different queries. **You will need** to have access to **Toad 12.12** and to the **database SYMYTCP.WORLD**.

The **SYMY main** SQL query is located [here](Queries/SYMY%20Main%20Query.sql). This query is an extract of our entire portfolio as at the date of the extract.

The **SYMY Expiry dates** SQL query is located [here](Queries/SYMY%20Expiry%20Date%20Query.sql). This query is an extract of all Special and Facultative exposures in the portfolio, that have an expiry date.

These queries must be extracted to a TSV format (Tab Separated Values), which is basically a text file (`.txt`) file.

### Step 1b - Download SharePoint extract

This is the [link](https://harmony.atradiusnet.com/workspaces/reinreins/sar/_layouts/15/start.aspx#/SitePages/Home.aspx) to the SharePoint website where we store our Special Acceptance and Facultative analysis.

- Under the page title "Reinsurance Underwriting", click **Special Acceptance request**
- In this page, right below the page title (Special Acceptance Request), there is a small menu. Click the **ellipsis** and select **Solvency 2_all UWY** view.
- In the *Solvency 2_all UWY* view, in the menu at the top of the page, click on **Library** > **Export to Excel**, then save Excel file to a folder.

### Step 1c - Treaty Register

The Treaty register extract is already in a folder in the S Drive (probably in `S:\Non Group Re\Treaty Register Report`), so you can just reference this file when running the code. But, it's preferable to have the file in your desktop, as to speed up the process of running the data checks.

Also, since the checks will be performed before quarter end, you can just use the latest Treaty Register.

After finishing step 1, you should have 4 files.

## Step 2 - Update Filepaths and report_date

In this step, you only have to update the file paths in the cells referencing the files you just downloaded.

The report date is the date of the next quarter end. For example, if running the checks on 20 of March of 2022, then `report_date = 31/03/2022`, which is the next quarter end date.

## Step 3 - Run report

After updating the filepaths and report_date, it's time to run the code to generate the Excel report.

This is achived by running `DataChecks().run_checks()`. If you want to export checks to an Excel file, set parameter `to_excel = True`.