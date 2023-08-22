SELECT
    ULTIMATE_ID,
    ULTIMATE_NAME,
    ULTIMATE_ISO_COUNTRY,
    MODEL_SUB_TYPE,
    CUSTOMER_ID,
    POD,
    EXPOSURE,
    ECAP
FROM
    (
        SELECT
            *
        FROM
            (
                SELECT
                    *
                FROM
                    (
                        SELECT
                            ULTIMATE_ID
                        FROM
                            /*Update below with Latest RUN*/
                            CALCXXXX.SO_REPORTING
                            /*Update above with Latest RUN*/
                        GROUP BY
                            ULTIMATE_ID
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
                    ROWNUM <= 40
            ) x
            LEFT JOIN (
                select
                    ULTIMATE_ID AS ULTIMATE_ID_HELPER,
                    ULTIMATE_NAME,
                    ULTIMATE_ISO_COUNTRY,
                    MODEL_SUB_TYPE,
                    CUSTOMER_ID,
                    ULTIMATE_POD as POD,
                    CREDIT_LIMIT_NET_EXPOSURE as EXPOSURE,
                    EC_CONSUMPTION_ND as ECAP
                from
                    /*Update below with Latest RUN*/
                    CALCXXXX.SO_REPORTING
                    /*Update above with Latest RUN*/
                where
                    MODEL_TYPE LIKE 'IR'
                    and ULTIMATE_RATING_TYPE LIKE 'ADV'
            ) y ON x.ULTIMATE_ID = y.ULTIMATE_ID_HELPER
    )
