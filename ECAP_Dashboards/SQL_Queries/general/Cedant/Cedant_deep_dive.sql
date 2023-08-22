SELECT
    NVL(OLD_ULTIMATE_ID, NEW_ULTIMATE_ID) AS ALIAS_ID,
    NVL(OLD_PARENT_NAME, NEW_PARENT_NAME) AS PARENT_NAME,
    OLD_PD,
    NEW_PD,
    NVL(NEW_PD, 0) - NVL(OLD_PD, 0) AS DELTA_PD,
    OLD_EXP,
    NEW_EXP,
    NVL(NEW_EXP, 0) - NVL(OLD_EXP, 0) AS DELTA_EXP,
    OLD_ECAP,
    NEW_ECAP,
    NVL(NEW_ECAP, 0) - NVL(OLD_ECAP, 0) AS DELTA_ECAP,
    OLD_RATING_TYPE,
    OLD_RATING,
    NEW_RATING_TYPE,
    NEW_RATING
    --OLD_RSQUARED,
    --NEW_RSQUARED,
    --NVL(NEW_RSQUARED, 0) - NVL(OLD_RSQUARED, 0) AS DELTA_RSQUARED
FROM
    (
        SELECT
            *
        FROM
            (
                SELECT
                    ULTIMATE_ID AS OLD_ULTIMATE_ID,
                    TRIM('+' FROM (TRIM('"' FROM (SUBSTR(ULTIMATE_NAME, 0, 70))))) as OLD_PARENT_NAME,
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
                    MAX(ULTIMATE_RATING) AS OLD_RATING,
                    SUM(
                        CASE
                            WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN ULTIMATE_RSQUARED
                        END
                    ) AS OLD_RSQUARED
                FROM
                    /*Update below with OLD RUN*/
                    CALCXXXX.SO_REPORTING
                    /*Update above with OLD RUN*/
                WHERE
                    MODEL_TYPE LIKE 'IR'
                    /*Update below with Balloon ID*/
                    AND CUSTOMER_ID = ''
                    /*Update above with Balloon ID*/
                GROUP BY
                    ULTIMATE_ID,
                    ULTIMATE_NAME
            ) x FULL OUTER JOIN (
                SELECT
                    ULTIMATE_ID AS NEW_ULTIMATE_ID,
                    TRIM('+' FROM (TRIM('"' FROM (SUBSTR(ULTIMATE_NAME, 0, 70))))) as NEW_PARENT_NAME,
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
                    MAX(ULTIMATE_RATING) AS NEW_RATING,
                    SUM(
                        CASE
                            WHEN MODEL_SUB_TYPE LIKE '%_COM_%' THEN ULTIMATE_RSQUARED
                        END
                    ) AS NEW_RSQUARED
                FROM
                    /*Update below with NEW RUN*/
                    CALCXXXX.SO_REPORTING
                    /*Update above with NEW RUN*/
                WHERE
                    MODEL_TYPE LIKE 'IR'
                    /*Update below with Balloon ID*/
                    AND CUSTOMER_ID = ''
                    /*Update above with Balloon ID*/
                GROUP BY
                    ULTIMATE_ID,
                    ULTIMATE_NAME
            ) y on x.OLD_ULTIMATE_ID = y.NEW_ULTIMATE_ID
    )
