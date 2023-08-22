SELECT
    *
FROM
    (
        SELECT
            ULTIMATE_ID,
            ULTIMATE_NAME,
            ULTIMATE_COUNTRY,
            COUNT(
                DISTINCT CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                    AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                    AND MODEL_TYPE LIKE 'IR' THEN CONTRACT_ID
                    ELSE ''
                END
            ) AS NUM_OF_BUYERS,
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                    AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                    AND MODEL_TYPE LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                    ELSE 0
                END
            ) AS ADV_RATE_EXP,
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_KN'
                    AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                    AND MODEL_TYPE LIKE 'IR' THEN EC_CONSUMPTION_ND
                END
            ) AS ECAP_ADV_RATE,
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE NOT LIKE '%POL%'
                    AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                    AND MODEL_TYPE NOT LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                    ELSE 0
                END
            ) AS GROUP_ADV_RATE_EXP,
            (
                SUM(
                    CASE
                        WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                        AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                        AND MODEL_TYPE LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE * ULTIMATE_POD
                        ELSE 0
                    END
                ) / SUM(
                    CASE
                        WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                        AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                        AND MODEL_TYPE LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                        ELSE 0
                    END
                )
            ) AS PD
        FROM
            CALCXXXX.SO_REPORTING
        GROUP BY
            ULTIMATE_ID,
            ULTIMATE_NAME,
            ULTIMATE_COUNTRY
            
        ORDER BY
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                    AND ULTIMATE_RATING_TYPE LIKE 'ADV'
                    AND MODEL_TYPE LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                    ELSE 0
                END
            ) DESC
    )
WHERE
    ROWNUM <= 30
