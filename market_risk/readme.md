# 1. Market Risk

Solvency Capital Requirements (SCR) calculations using Python.

The table with each reporting run is [here](https://github.com/viccorrera/InwardsRe/blob/main/ECAP_Dashboards/readme.md#reporting-runs).

## 1.1. Table of Content

- [1. Market Risk](#1-market-risk)
  - [1.1. Table of Content](#11-table-of-content)
  - [1.2. Background](#12-background)
  - [1.3. How to use it](#13-how-to-use-it)
  - [1.4. Acronym list](#14-acronym-list)
    - [1.4.1. *Equity Risk columns*](#141-equity-risk-columns)
- [2. Equity Risk](#2-equity-risk)
  - [2.1. Objectives](#21-objectives)
  - [2.2. Required Data](#22-required-data)
    - [2.2.1. *TBSL_EQUITIES*](#221-tbsl_equities)
    - [2.2.2. *Equity Risk Shocks*](#222-equity-risk-shocks)
    - [2.2.3. *FC_COMPUTED_AMOUNTS*](#223-fc_computed_amounts)
  - [Steps to get an html report](#steps-to-get-an-html-report)
- [3. Currency Risk](#3-currency-risk)
    - [3.0.1. FC_COMPUTED_AMOUNTS](#301-fc_computed_amounts)
    - [3.0.2. SCR Calculation](#302-scr-calculation)
    - [3.0.3. IR_FX_RISK](#303-ir_fx_risk)
- [4. Spread Risk](#4-spread-risk)
  - [4.1. Objectives](#41-objectives)
  - [4.2. Required Data](#42-required-data)
    - [4.2.1. *Spread Risk Data*](#421-spread-risk-data)

<br>

## 1.2. Background  

Project initiated in April 2021 to replace Excel based analyses within the Risk
Management Function with Python equivalent.

The key objectives of the project for each specific area of work are:
- Replace current Excel process with equivalent process using Python. 
- Ensure all features of the previous analysis remain available to the end user.
- Ensure sufficient testing has been carried out to ensure confidence in moving
to the new process.
- Ensure all coding and methodologies are well documented, allowing a 
knowledgeable third party to easily replicate if necessary.
- Ensure data storage, coding and documnetation is flexible in terms of ad-hoc
analysis requirements.
- The new process should produce PDF style "regular" documentation that in a
normal flow will be sufficient to cover the analysis.
- The new process should also produce an Excel data bank for use if necessary.

## 1.3. How to use it

Each risk module will be have a dedicated Jupyter Notebook for running the code.

For example, `2_equity.ipynb` is dedicated to calculate the Equity SCR
and produce an HTML/Excel report.

Inside each Jupyter Notebook there will be further instructions on how to set 
up the folders necessary to run the code.

[Download Excel Reference file ](https://github.com/viccorrera/InwardsRe/raw/main/market_risk/reference_file.xlsx)

## 1.4. Acronym list

- RPT -> Reporting (Original) Currency

### 1.4.1. *Equity Risk columns*

- BOF -> Basic Own Funds
- MV -> Market Value
- NPV -> Net Present Value
- FK_RW_MODEL_TYP (column):
  - SP -> Special Products
  - ICP -> Instalment Credit Protection
  - IR -> Inward Reinsurance
  - BO -> Bonding
  - CI -> Credit Insurance  
- FK_RW_PRODUCT (column):
  - FI -> Fidelity
  - ICPP -> Instalment Credit Protection
  - CI -> Credit Insurance
  - SE -> Services
  - BO -> Bonding
- FK_RW_LINE_OF_BUSINESS (column):
  - 21 -> Credit and suretyship proportional reinsurance
  - 28 -> Non-proportional property reinsurance
- FK_RW_FINANCIAL_CF_TYP (column):
  - PCO -> Provision Claims Outstanding
  - PR -> Premium Reserve



---

<br>

# 2. Equity Risk

## 2.1. Objectives

- Replicate the calculation of the Atradius Re quarterly Equity Risk SCR using the semantic layer data and the EIOPA prescribed shocks. 
- Produce an analysis of the key drivers of the current Equity Risk SCR. 
- Produce an analysis of the key drivers of the movement of the Equity Risk SCR.
- Produce analysis overview in HTML format. 
- Produce an Excel data bank for reference.

<br>  

## 2.2. Required Data

- Current quarter Atradius Re Equity Portfolio data (taken from Semantic Layer).
- Previous quarter Atradius Re Equity Portfolio data (taken from previous quarter excel data bank)
- EIOPA prescribed Equity Risk SCR shocks. 
- IM data extract for calculation verification.

List of Sheets below:

### 2.2.1. *TBSL_EQUITIES*

**Source**: EDWQ3 environment, otherwise known as the semantic layer.

This environment is populated following the Data Source Load and Data
Reconciliation phases of the quarterly SII Run and is therefore available prior
to the run. This data is from **current quarter** only.

As the Equity Risk module is deterministic, the calculation of the SCR can be
produced prior to the run, with a verification made using the 
FC_COMPUTED_AMOUNTS table post-run.

This table consists of the Atradius Re equity portfolio at the time of
extraction, please see tab `TBSL_EQUITIES 2021Q1` from `reference_file.xlsx`. 

**SQL Query**  
The column aliases **cannot be modified**, otherwise the code won't work.

<details>
	<summary>Click to view query</summary>
	<p>
		
```SQL
SELECT
    eq.REPORTING_DAT,
    eq.ISSUER_IDENTIFIER AS ISSUER_ID,
    als.ALIAS_NAME,
    eq.COUNTRY_CODE,
    CONCAT(eq.EQUITY_CATEGORY, eq.EQUITY_TYP) AS EQUITYCATEGORY,
    eq.EQUITY_TYP,
    eq.EQUITY_IDENTIFIER AS EQUITY_ID,
    eq.ECAI,
    eq.U_ECAI,
    eq.SECTOR,
    eq.SUBPORTFOLIO,
    eq.PURCHASE_DAT,
    eq.SOLVENCY_II_VALUATION AS SOL_II_VAL,
    eq.FUNCTIONAL_CURRENCY_CODE,
    fx.CONVERSION_RATE AS FX_RATE,
    (eq.SOLVENCY_II_VALUATION / fx.CONVERSION_RATE) AS SOL_II_VAL_EUR
FROM
    EDWQ3.TBSL_EQUITIES eq,
    EDWQ3.TBSL_ALIASES als,
    EDWQ3.TBSL_CURR_EXCHANGE_RATES fx
WHERE
    /*AtRe branch code*/
    eq.BRANCH_CODE IN ('277','279')
    /*Month end FX-rate (VAR)*/
    AND fx.CONVERSION_TYPE_CODE = 'VAR'
    /*Connecting tables*/
    AND eq.FUNCTIONAL_CURRENCY_CODE = fx.TARGET_CURRENCY (+)
    AND eq.ISSUER_IDENTIFIER = als.ALIAS_IDENTIFIER (+)
    AND eq.ISSUER_IDENTIFIER = als.ALIAS_IDENTIFIER (+)
    AND eq.ALIAS_TYPE_CODE = als.ALIAS_TYPE_CODE (+)
```

</p>

</details>

<br>

### 2.2.2. *Equity Risk Shocks*
	
Reference table for the shocks to be applied to equities.  

**Source**: 
https://www.eiopa.europa.eu/tools-and-data/symmetric-adjustment-equity-capital-charge_en  

Based on this shock prescription, the Python code ([link here](https://github.com/viccorrera/InwardsRe/blob/main/modules/market_risk/equity_risk.py#L731)) will calculate all
shocks for the Equity Risk module.

These shocks will be verified in the code against the data produced for the
Internal Model Run.  

This data is taken from SMF (https://sol-prod-master.atradiusnet.com:8443/smf/parametervalues.jsf)


<br>


### 2.2.3. *FC_COMPUTED_AMOUNTS*

**Source**: ECPP environment

This table gives the final reportable SCR amounts for each SCR module.

We must verify the amount calculated in the code using the semantic layer data
and the EIOPA prescribed shock matches the amout in this table.

```SQL
SELECT 
    REPORTING_PERIOD,
    FK_CD_COMPUTED_AMOUNT_TYPE,
    S2_LEGAL_ENTITY,
    FK_RW_ISO_CURRENCY_EUR,
    COMPUTED_AMT_EUR
FROM CALCXXXX.FC_COMPUTED_AMOUNTS --update CALCRUN--
WHERE 
    S2_LEGAL_ENTITY = 'D45'
    AND FK_CD_COMPUTED_AMOUNT_TYPE = 'MKTeq'
```

<br>

## Steps to get an html report

In order to produce an Equity Risk report, the file `2_equity.ipynb` must be used.

In this file there are further instructions on how to run the code, and what to update.

---

<br>


# 3. Currency Risk

Currency Risk refers to the effect of a change in the exchange rates of particular currencies on basic own funds.
The change should be adverse, in other words it could relate to a depreciation or an appreciation of a currency, depending on the magnitude of receivables versus liabilities in that currency.

**Use Automation Designer**

### 3.0.1. FC_COMPUTED_AMOUNTS
- `REPORTING_PERIOD`: Quarter End Date of the Reporting Quarter
- `FC_CD_COMPUTED_AMOUNT_TYPE`: Risk Type
- `S2_LEGAL_ENTITY`: Legal Entity. Atradius is always D45
- `FK_RW_ISO_CURRENCY_RPT`: Reporting/Original Currency
- `FK_RW_ISO_CURRENCY_EUR`: EUR Currency
- `COMPUTED_AMT_RPT`: Computed SCR in Reporting/Original Currency
- `COMPUTED_AMT_EUR`: Computed SCR of Reporting/Original Currency converted to EUR

### 3.0.2. SCR Calculation

```
CASE
.
.
.
END AS SCR_SHOCK
```
- Represents Shock table from Spreadsheet in previous Quarters

### 3.0.3. IR_FX_RISK

- 'grp' stands for Group Currency (EUR)
- The sum of column `AMOUNT_LOSS` is the SCR for Currency Risk

# 4. Spread Risk

## 4.1. Objectives

- Replicate the calculation of the Atradius Re quarterly Spread Risk SCR using the semantic layer data.
- Produce an analysis of the key drivers of the current Spread Risk SCR. 
- Produce an analysis of the key drivers of the movement of the Spread Risk SCR.
- Produce analysis overview in HTML format. 
- Produce an Excel data bank for reference.

<br>

## 4.2. Required Data

- Current quarter Atradius Re Spread Risk data (taken from Semantic Layer).
- Previous quarter Atradius Re Spread Risk data (taken from previous quarter excel data bank)
- IM data extract for calculation verification.

List of Sheets below:

### 4.2.1. *Spread Risk Data*

**Source**: EDWQ3 environment, otherwise known as the semantic layer.

This environment is populated following the Data Source Load and Data
Reconciliation phases of the quarterly SII Run and is therefore available prior
to the run. This data is from **current quarter** only.

As the Spread Risk module is deterministic, the calculation of the SCR can be
produced prior to the run, with a verification made using the 
FC_COMPUTED_AMOUNTS table post-run.

This single file combines information from 3 different tables in Toad.

**SQL Query**  

<details>
  <summary>Click to view query</summary>

```SQL
SELECT 
    bval.REPORTING_DAT,
    b.BOND_IDENTIFIER,
    bval.ASSET_MANAGER,
    b.COUNTERPARTY_RISK_TYPE_CODE,
    b.ISSUER_IDENTIFIER AS ISSUER_ID,
    b.ISSUER_DES,
    b.ISSUER_GROUP_DES,
    b.SECTOR,
    b.BOND_CURRENCY_CODE,
    fx.CONVERSION_RATE AS FX_RATE,
    b.COUNTRY_CODE,
    b.TYPE_OF_BOND,
    b.MATURITY_DAT,
    bval.MODIFIED_DURATION,
    bval.ECAI,
    bval.U_ECAI,
    bval.MARKET_VALUE,
    (bval.MARKET_VALUE / fx.CONVERSION_RATE) AS MARKET_VALUE_EUR,
    CASE
        WHEN bval.MODIFIED_DURATION <= 5 THEN 5
        WHEN bval.MODIFIED_DURATION <= 10 THEN 10
        WHEN bval.MODIFIED_DURATION <= 15 THEN 15
        WHEN bval.MODIFIED_DURATION <= 20 THEN 20
        ELSE 9999
    END AS DURATION_END_YEAR,
    (b.MATURITY_DAT - bval.REPORTING_DAT) / 365.25 AS DURATION_TO_MATURITY,
    --FLOOR((b.MATURITY_DAT - bval.REPORTING_DAT) * 4 / 365.25) AS DURATION_QTR,
    --FLOOR((b.MATURITY_DAT - bval.REPORTING_DAT) / 365.25) AS DURATION_YR
FROM
    EDWQ3.TBSL_BONDS b,
    EDWQ3.TBSL_BOND_HOLDING_VALS bval,
    EDWQ3.TBSL_CURR_EXCHANGE_RATES fx
WHERE
    b.BOND_IDENTIFIER = bval.BOND_IDENTIFIER
    AND b.DATA_SOURCE_CODE = bval.DATA_SOURCE_CODE
    AND bval.BRANCH_CODE in ('277','279')
    /*FX Parameters*/
    AND fx.CONVERSION_TYPE_CODE = 'VAR'
    AND b.BOND_CURRENCY_CODE = fx.TARGET_CURRENCY
```

</details>
