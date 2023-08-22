WITH OLD_QRT AS (
    SELECT
        sprd.FK_BOND_ID AS OLD_BOND_ID,
        sprd.MARKET_VAL AS OLD_MARKET_VAL
    FROM
        /*Update below with PREVIOUS calcrun*/
        CALCxxxx.IR_SPREAD sprd,
        CALCxxxx.RW_ALIAS rwalias
        /*Update above with PREVIOUS calcrun*/
    WHERE
        sprd.FK_RW_ATRADIUS_COMP_S2_LE = 'D45'
        AND sprd.FK_RW_ALIAS_ALIASID_ISSUER = rwalias.PK_ALIAS_ID (+)
        AND sprd.FK_RW_DATA_SOURCE = rwalias.PK_FK_RW_DATA_SOURCE (+)
        /*Update below with ALIAS_ID*/
        AND rwalias.GROUP_ID = ''
        /*Update above with ALIAS_ID*/
),
NEW_QRT AS (
    SELECT
        sprd.FK_BOND_ID AS NEW_BOND_ID,
        sprd.MARKET_VAL AS NEW_MARKET_VAL
    FROM
        /*Update below with CURRENT calcrun*/
        CALCxxxx.IR_SPREAD sprd,
        CALCxxxx.RW_ALIAS rwalias
        /*Update above with CURRENT calcrun*/
    WHERE
        sprd.FK_RW_ATRADIUS_COMP_S2_LE = 'D45'
        AND sprd.FK_RW_ALIAS_ALIASID_ISSUER = rwalias.PK_ALIAS_ID (+)
        AND sprd.FK_RW_DATA_SOURCE = rwalias.PK_FK_RW_DATA_SOURCE (+)
        /*Update below with ALIAS_ID*/
        AND rwalias.GROUP_ID = ''
        /*Update above with ALIAS_ID*/
)
SELECT
    NVL(OLD_BOND_ID, NEW_BOND_ID) AS FK_BOND_ID,
    OLD_MARKET_VAL,
    NEW_MARKET_VAL
FROM
    OLD_QRT
    FULL OUTER JOIN (NEW_QRT)
    ON OLD_QRT.OLD_BOND_ID = NEW_QRT.NEW_BOND_ID