SELECT
    *
FROM
    (
        SELECT
            ULTIMATE_NAME,
            ULTIMATE_ID,
            COUNT(
                DISTINCT CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                    AND ULTIMATE_RATING_TYPE IS NULL
                    AND MODEL_TYPE LIKE 'IR' THEN CONTRACT_ID
                    ELSE ''
                END
            ) AS NUM_OF_BUYERS,
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                    AND ULTIMATE_RATING_TYPE IS NULL
                    AND MODEL_TYPE LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                    ELSE 0
                END
            ) AS UNRATED_EXP,
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE NOT LIKE '%POL%'
                    AND ULTIMATE_RATING_TYPE IS NULL
                    AND MODEL_TYPE NOT LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                    ELSE 0
                END
            ) AS GROUP_UNRATED_EXP
        FROM
            CALCXXXX.SO_REPORTING
        GROUP BY
            ULTIMATE_NAME,
            ULTIMATE_ID
        ORDER BY
            SUM(
                CASE
                    WHEN MODEL_SUB_TYPE LIKE '%_COM_KN'
                    AND ULTIMATE_RATING_TYPE IS NULL
                    AND MODEL_TYPE LIKE 'IR' THEN CREDIT_LIMIT_NET_EXPOSURE
                    ELSE 0
                END
            ) DESC
    )
WHERE
    ROWNUM <= 40
