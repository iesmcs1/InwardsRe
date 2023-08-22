SELECT
    NVL(OLD_COMP_ID, NEW_COMP_ID) AS COMP_ID,
    OLD_ECAP,
    NEW_ECAP,
    NVL(NEW_ECAP, 0) - NVL(OLD_ECAP, 0) AS DELTA_ECAP
FROM
    (
        SELECT
            *
        FROM
            (
                SELECT
                    SUBSTR(CONTRACT_ID, 0, 5) AS OLD_COMP_ID,
                    SUM(EC_CONSUMPTION_ND) AS OLD_ECAP
                FROM
                    CALCXXXX.SO_REPORTING -- UPDATE OLD QUARTER --
                WHERE
                    MODEL_TYPE = 'IR'
                    AND MODEL_SUB_TYPE LIKE 'BO_%'
                GROUP BY
                    SUBSTR(CONTRACT_ID, 0, 5)
            ) old FULL OUTER JOIN (
                SELECT
                    SUBSTR(CONTRACT_ID, 0, 5) AS NEW_COMP_ID,
                    SUM(EC_CONSUMPTION_ND) AS NEW_ECAP
                FROM
                    CALCXXXX.SO_REPORTING -- UPDATE NEW QUARTER --
                WHERE
                    MODEL_TYPE = 'IR'
                    AND MODEL_SUB_TYPE LIKE 'BO_%'
                GROUP BY
                    SUBSTR(CONTRACT_ID, 0, 5)
            ) new ON old.OLD_COMP_ID = new.NEW_COMP_ID
    )
