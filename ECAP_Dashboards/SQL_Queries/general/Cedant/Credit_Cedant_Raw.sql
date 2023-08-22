SELECT
    CONTRACT_ID as TREATY_ID,
    SUM(
        CASE
            WHEN MODEL_SUB_TYPE LIKE 'CI_COM_%' THEN CREDIT_LIMIT_NET_EXPOSURE
            ELSE 0
        END
    ) AS TPE,
    SUM(EXPECTED_LOSS) AS EXPECTED_LOSS,
    SUM(EC_CONSUMPTION_ND) AS ECAP,
    (
        SUM(
            CASE
                WHEN MODEL_SUB_TYPE LIKE 'CI_COM_%' THEN CREDIT_LIMIT_NET_EXPOSURE * ULTIMATE_POD
                ELSE 0
            END
        ) / SUM(
            CASE
                WHEN MODEL_SUB_TYPE LIKE 'CI_COM_%' THEN CREDIT_LIMIT_NET_EXPOSURE
                ELSE 0
            END
        )
    ) AS PD
FROM
    CALCXXXX.SO_REPORTING
WHERE
    MODEL_TYPE = 'IR'
    AND MODEL_SUB_TYPE LIKE 'CI_%'
GROUP BY
    CONTRACT_ID
ORDER BY
    CONTRACT_ID
