SELECT
    SUM(NPV_INTEREST_RATE_VAL) AS SUM_NPV,
    SUM(SHOCK_UP_VAL) AS SUM_SHOCK_UP,
    SUM(SHOCK_DOWN_VAL) AS SUM_SHOCK_DOWN,
    MAX((SELECT MAX(REPORTING_PERIOD) FROM CALC&CALCVAR..FC_COMPUTED_AMOUNTS)) AS REPORTING_PERIOD,
    ('&CALCVAR.') AS CALCRUN
FROM
    CALC&CALCVAR..IR_TP_POSITION_GRANULAR
WHERE
    FK_RW_ATRADIUS_COMPANY = 'D45'
    AND POSITION_TYP = 'NET'
