/*
This query groups Commercial and Political Exposure together.
It sums the ECap and EL, and removes TPE and PD from Political exposures.
*/

SELECT
    CONTRACT_ID,
    CUSTOMER_ID,
    -- Return Model Sub Type as Bond / Credit and KN / UKN
    MAX(
        CASE
            WHEN ULTIMATE_NAME IS NULL THEN CONCAT(SUBSTR(MODEL_SUB_TYPE, 0, 2), '_UNK')
            ELSE CONCAT(SUBSTR(MODEL_SUB_TYPE, 0, 2), '_KN')
        END
    ) AS MODEL_TYPE_SUB,
    ALIAS_ID,
    ULTIMATE_ID,
    TRIM('"' FROM (SUBSTR(REGEXP_REPLACE(ULTIMATE_NAME, '^[+-]'), 0, 70))) AS ULTIMATE_NAME,
    ULTIMATE_ISO_COUNTRY,
    ULTIMATE_COUNTRY,
    ULTIMATE_INDUSTRY_SECTOR,
    MAX(ULTIMATE_RATING_TYPE),
    MAX(ULTIMATE_RATING),
    
    SUM(
        -- removes CREDIT_LIMIT_NET_EXPOSURE from Political Exposure
        CASE
            WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN CREDIT_LIMIT_NET_EXPOSURE
            ELSE 0
        END) AS EXP_GROSS_OF_RETRO,
    SUM(
        -- removes ULTIMATE_POD from Political Exposure
        CASE
            WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN ULTIMATE_POD
            ELSE 0
        END) AS ULTIMATE_POD,
    SUM(EXPECTED_LOSS),
    SUM(EC_CONSUMPTION_ND),
    (SELECT MAX(REPORTING_PERIOD) FROM CALC&CALCVAR..FC_COMPUTED_AMOUNTS) AS REPORTING_PERIOD,
    ('&CALCVAR.') AS CALCRUN
FROM
    CALC&CALCVAR..SO_REPORTING
WHERE
    MODEL_TYPE = 'IR'
GROUP BY
    CONTRACT_ID,
    CUSTOMER_ID,
    ALIAS_ID,
    ULTIMATE_ID,
    ULTIMATE_NAME,
    ULTIMATE_ISO_COUNTRY,
    ULTIMATE_COUNTRY,
    ULTIMATE_INDUSTRY_SECTOR