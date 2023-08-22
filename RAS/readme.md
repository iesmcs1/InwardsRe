# Risk Appetite Statement (RAS)

The following variables need to be updated before running the Notebook

1. `reg_path` : full path to the Treaty Register of the reporting period.
   
2. `rep_date` : reporting date of the reporting period (last day in the quarter).

3. `ecap_filepath` : full path to the ECap file (`.txt` format).
    - This has to be grouped by Comp (5 digits)


## ECap File

The following Query can be used to get the ECap file.

```SQL
SELECT
    SUBSTR(CUSTOMER_ID,0 ,5) AS COMP,
    SUM(EC_CONSUMPTION_ND) AS ECAP
FROM
    /*Update Reporting Run below*/
    CALCxxxx.SO_REPORTING
    /*Update Reporting Run above*/
WHERE
    MODEL_TYPE = 'IR'
GROUP BY
    SUBSTR(CUSTOMER_ID,0 ,5)
```