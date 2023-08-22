WITH OLD_QRT AS (
    SELECT
        PK_FK_RW_RATING_VAL_RATTYP AS OLD_RATING_TYPE,
        PK_FK_RW_RATING_VAL_VAL AS OLD_RATING
    FROM
        /*Update below with CURRENT calcrun*/
        CALCxxxx.RW_ALIAS_RATING
        /*Update above with CURRENT calcrun*/
    WHERE
        /*Update below with ALIAS_ID*/
        PK_FK_RW_ALIAS_ID = ''
        /*Update above with ALIAS_ID*/
        AND PK_FK_RW_ALIAS_DATASOURCE = 'ACI'
),
NEW_QRT AS (
    SELECT
        PK_FK_RW_RATING_VAL_RATTYP AS NEW_RATING_TYPE,
        PK_FK_RW_RATING_VAL_VAL AS NEW_RATING
    FROM
        /*Update below with CURRENT calcrun*/
        CALCxxxx.RW_ALIAS_RATING
        /*Update above with CURRENT calcrun*/
    WHERE
        /*Update below with ALIAS_ID*/
        PK_FK_RW_ALIAS_ID = ''
        /*Update above with ALIAS_ID*/
        AND PK_FK_RW_ALIAS_DATASOURCE = 'ACI'
)
SELECT
    NVL(OLD_RATING_TYPE, NEW_RATING_TYPE) AS RATING_TYPE,
    OLD_RATING,
    NEW_RATING
FROM
    OLD_QRT
    FULL OUTER JOIN (NEW_QRT)
    ON OLD_QRT.OLD_RATING_TYPE = NEW_QRT.NEW_RATING_TYPE