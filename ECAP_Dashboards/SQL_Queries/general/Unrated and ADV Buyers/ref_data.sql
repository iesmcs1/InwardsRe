SELECT
    SUM(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_POL_%' THEN 0
            ELSE a.CREDIT_LIMIT_NET_EXPOSURE
        END
    ) AS TPE,
    SUM(a.EC_CONSUMPTION_ND) as ECAP,
    /*----------------*/
    /*UNRATED SECTION*/
    SUM(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE IS NULL THEN a.CREDIT_LIMIT_NET_EXPOSURE
            ELSE 0
        END
    ) AS UNRATED_EXP,
    COUNT(
        DISTINCT CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE IS NULL THEN ULTIMATE_ID
            ELSE ''
        END
    ) AS UNRATED_BUYERS,
    (
        SUM(
            CASE
                WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
                AND a.ULTIMATE_RATING_TYPE IS NULL THEN a.CREDIT_LIMIT_NET_EXPOSURE * a.ULTIMATE_POD
                ELSE 0
            END
        ) / SUM(
            CASE
                WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
                AND a.ULTIMATE_RATING_TYPE IS NULL THEN a.CREDIT_LIMIT_NET_EXPOSURE
                ELSE 0
            END
        )
    ) AS UNRATED_BUYERS_WEIGHTED_AV_PD,
    MIN(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE IS NULL THEN a.ULTIMATE_POD
            ELSE 100
        END
    ) AS UNRATED_MIN_PD,
    MAX(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE IS NULL THEN a.ULTIMATE_POD
            ELSE 0
        END
    ) AS UNRATED_MAX_PD,
    SUM(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_KN'
            AND a.ULTIMATE_RATING_TYPE IS NULL THEN a.EC_CONSUMPTION_ND
            ELSE 0
        END
    ) AS UNRATED_ECAP,
    /*----------------*/
    /*ADV RATE SECTION*/
    SUM(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN a.CREDIT_LIMIT_NET_EXPOSURE
            ELSE 0
        END
    ) AS ADV_RATED_EXP,
    COUNT(
        DISTINCT CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN ULTIMATE_ID
            ELSE ''
        END
    ) AS ADV_RATED_BUYERS,
    (
        SUM(
            CASE
                WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
                AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN a.CREDIT_LIMIT_NET_EXPOSURE * a.ULTIMATE_POD
                ELSE 0
            END
        ) / SUM(
            CASE
                WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
                AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN a.CREDIT_LIMIT_NET_EXPOSURE
                ELSE 0
            END
        )
    ) AS ADV_RATED_BUYERS_WEIGHTED_PD,
    MIN(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN a.ULTIMATE_POD
            ELSE 100
        END
    ) AS ADV_RATED_MIN_PD,
    MAX(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_COM_KN'
            AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN a.ULTIMATE_POD
            ELSE 0
        END
    ) AS ADV_RATED_MAX_PD,
    SUM(
        CASE
            WHEN a.MODEL_SUB_TYPE LIKE '%_KN'
            AND a.ULTIMATE_RATING_TYPE LIKE 'ADV' THEN a.EC_CONSUMPTION_ND
            ELSE 0
        END
    ) AS ADV_RATED_ECAP
FROM
    CALCXXXX.SO_REPORTING a  --update CALCRUN --
WHERE
    a.MODEL_TYPE = 'IR'
