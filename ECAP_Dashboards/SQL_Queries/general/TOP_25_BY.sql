SELECT
    *
FROM
    (
        SELECT
            NVL(OLD_ULTIMATE_ID, NEW_ULTIMATE_ID) AS ALIAS_ID,
            NVL(OLD_PARENT_NAME, NEW_PARENT_NAME) AS PARENT_NAME,
            OLD_NO_CUSTOMERS,
            NEW_NO_CUSTOMERS,
            OLD_PD,
            NEW_PD,
            NVL(NEW_PD, 0) - NVL(OLD_PD, 0) AS DELTA_PD,
            NVL(OLD_EXP, 0) AS OLD_EXP,
            NVL(NEW_EXP, 0) AS NEW_EXP,
            NVL(NEW_EXP, 0) - NVL(OLD_EXP, 0) AS DELTA_EXP,
            NVL(OLD_ECAP, 0) AS OLD_ECAP,
            NVL(NEW_ECAP, 0) AS NEW_ECAP,
            NVL(NEW_ECAP, 0) - NVL(OLD_ECAP, 0) AS DELTA_ECAP,
            OLD_RATING_TYPE,
            OLD_RATING,
            NEW_RATING_TYPE,
            NEW_RATING,
            NVL(OLD_ECAP, 0) / NVL(OLD_EXP, 1) AS OLD_ECAP_TPE_RATIO,
            NVL(NEW_ECAP, 0) / NVL(NEW_EXP, 1) AS NEW_ECAP_TPE_RATIO
        FROM
            (
                SELECT
                    *
                FROM
                    (
                        SELECT
                            ULTIMATE_ID AS OLD_ULTIMATE_ID,
                            SUBSTR(ULTIMATE_NAME, 0, 70) as OLD_PARENT_NAME,
                            COUNT(DISTINCT(CUSTOMER_ID)) AS OLD_NO_CUSTOMERS,
                            AVG(
                                CASE
                                    WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN ULTIMATE_POD
                                END
                            ) AS OLD_PD,
                            SUM(
                                CASE
                                    WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN CREDIT_LIMIT_NET_EXPOSURE
                                END
                            ) AS OLD_EXP,
                            SUM(EC_CONSUMPTION_ND) AS OLD_ECAP,
                            MAX(ULTIMATE_RATING_TYPE) AS OLD_RATING_TYPE,
                            MAX(ULTIMATE_RATING) AS OLD_RATING
                        FROM
                            CALCXXXX.SO_REPORTING -- UPDATE OLD QUARTER --
                        WHERE
                            MODEL_TYPE LIKE 'IR'
                            AND MODEL_SUB_TYPE LIKE '%_KN'
                        GROUP BY
                            ULTIMATE_ID,
                            SUBSTR(ULTIMATE_NAME, 0, 70)
                    ) x FULL OUTER JOIN (
                        SELECT
                            ULTIMATE_ID AS NEW_ULTIMATE_ID,
                            SUBSTR(ULTIMATE_NAME, 0, 70) as NEW_PARENT_NAME,
                            COUNT(DISTINCT(CUSTOMER_ID)) AS NEW_NO_CUSTOMERS,
                            AVG(
                                CASE
                                    WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN ULTIMATE_POD
                                END
                            ) AS NEW_PD,
                            SUM(
                                CASE
                                    WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN CREDIT_LIMIT_NET_EXPOSURE
                                END
                            ) AS NEW_EXP,
                            SUM(EC_CONSUMPTION_ND) AS NEW_ECAP,
                            MAX(ULTIMATE_RATING_TYPE) AS NEW_RATING_TYPE,
                            MAX(ULTIMATE_RATING) AS NEW_RATING
                        FROM
                            CALCXXXX.SO_REPORTING -- UPDATE NEW QUARTER --
                        WHERE
                            MODEL_TYPE LIKE 'IR'
                            AND MODEL_SUB_TYPE LIKE '%_KN'
                        GROUP BY
                            ULTIMATE_ID,
                            SUBSTR(ULTIMATE_NAME, 0, 70)
                    ) y on x.OLD_ULTIMATE_ID = y.NEW_ULTIMATE_ID
            )
        ORDER BY
            --NEW_ECAP DESC -- by ECap --
            --NEW_EXP DESC  -- by Exposure --
    )
WHERE
    ROWNUM <= 25
