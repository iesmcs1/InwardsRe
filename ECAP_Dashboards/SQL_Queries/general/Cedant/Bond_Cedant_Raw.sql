SELECT
    CONTRACT_ID as TREATY_ID,
    SUM(CREDIT_LIMIT_NET_EXPOSURE) AS TPE,
    SUM(EXPECTED_LOSS) AS EXPECTED_LOSS,
    SUM(EC_CONSUMPTION_ND) AS ECAP,
    SUM(CREDIT_LIMIT_NET_EXPOSURE * ULTIMATE_POD) / SUM(CREDIT_LIMIT_NET_EXPOSURE) AS PD
FROM
    CALCXXXX.SO_REPORTING
WHERE
    MODEL_TYPE = 'IR'
    AND MODEL_SUB_TYPE LIKE 'BO_COM_%'
GROUP BY
    CONTRACT_ID
ORDER BY
    CONTRACT_ID
