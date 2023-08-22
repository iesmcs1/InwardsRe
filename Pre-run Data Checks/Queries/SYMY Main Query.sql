/* Update WHERE clause cld.EFFECT_TO_DAT if extracting data at quarter end */

select
cld.ID Credit_Limit_ID
,cld.ORBUD_ORNNN_ID Buyer_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(org.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Buyer_Name -- If we don't do this, excel will read the data incorrectly
,org.D_ORNOL_STATUS_CODE Buyer_Status 
,ibr.FINAL_RATING IBR_Value
,ibr.FINAL_RATING_TYP IBR_Type
,org.ORCOY_ID Buyer_Country_ID
,org.D_ORCOY_MAIN_NAME Buyer_Country
,cob.CODE Buyer_Country_ISO
,rel.ORBUD_PARENT_ORNNN_ID Parent_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(par.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Parent_Name -- If we don't do this, excel will read the data incorrectly
,par.ORCOY_ID Parent_Country_ID
,par.D_ORCOY_MAIN_NAME Parent_Country
,cop.CODE Parent_Country_ISO
,cld.BUPIY_ID Policy_Number
,pol.LEG_SYS_REF_NR Legacy_Policy
,pol.ORCUR_ORNNN_ID Customer_Number 
,cus.D_ORONE_FIRST_LINE_NAME Customer_Name 
,cus.D_ORCOY_MAIN_NAME Customer_Country
,TRUNC(cld.CLA_RECORDED_DAT) CLA_Date
,TRUNC(cld.CLD_STATUS_DAT) CLD_Date
,cld.CLA_AMT CLA_Amount
,ROUND(cld.CLA_AMT / ex.AMT) CLA_Amount_Euro
,cld.CLD_FIRST_AMT CLD_First_Amount
,ROUND(cld.CLD_FIRST_AMT / ex.AMT) CLD_First_Amount_Euro
,cld.CLD_SECOND_AMT CLD_Second_Amount
,ROUND(cld.CLD_SECOND_AMT / ex.AMT) CLD_Second_Amount_Euro
,cld.CLD_TOTAL_AMT CLD_Total_Amount
,ROUND(cld.CLD_TOTAL_AMT / ex.AMT) CLD_Total_Amount_Euro
,cld.D_BUPIY_ORCUY_CODE Policy_Currency
,cld.BUCCY_ID Commitment_Category_Number
,com.NAME Commitment_Category
,cld.BUCTE_CODE Commitment_Type
,cld.BUDTE_CODE Decision_Type_Code
,dec.DES Decision_Type 
,su.INDIV_UID Underwriter_User_ID
,cld.ORSUS_CLD_ID Underwriter_System_User_ID
,su.D_ORIDL_INDIV_NAME Underwriter_Name
,TRUNC(cld.EFFECT_TO_DAT) Effective_to_Date
,org.D_MAIN_TRADE_NACE_CODE
,trdsec.NAME as BUYER_TRADE_SECTOR
,trdgrp.NAME as BUYER_TRADE_GROUP_SECTOR
,TO_CHAR(MAX(SYSDATE), 'DD/MM/YYYY') AS REFRESH_DATE

from
ORABUP0.TBBU_CREDIT_LIMITS cld
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS org
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS cus
,ORABUP0.TBBU_POLICIES pol
,ORABUP0.TBOR_CURRENCY_EXCHANGE_RATES ex
,ORABUP0.TBBU_COMMITMENT_CATEGORIES com
,ORABUP0.TBOR_SYSTEM_USERS su
,ORABUP0.TBBU_DECISION_TYPES dec
,ORABUP0.TBOR_RISK_GROUP_RELATIONS rel
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS par
,ORABUP0.TBOR_COUNTRIES cop
,ORABUP0.TBOR_COUNTRIES cob
,ORABUP0.TBBU_OVERALL_OUTPUT_IBR ibr
,ORABUP0.TBBU_TRADE_SECTORS trdsec
,ORABUP0.TBBU_TRADE_GROUPS trdgrp

where
cld.BUCCY_ID = 14 --COMMITMENT_CATEGORY_NUMBER for Inwards Re--

AND cld.CLD_STATUS_CODE = 'FINL' --query was returning DRAFT decisions also, and creating duplicates--

/* Comment first row and Uncomment second row
   if downloading data for Quarter end */
AND cld.EFFECT_TO_DAT > SYSDATE --ALWAYS ON--
--AND cld.EFFECT_TO_DAT >= '31-DEC-2021' --ALWAYS ON--
/* update above accordingly */

/* com table has category name */
AND com.ID = cld.BUCCY_ID --ALWAYS ON--
/* su table has the name of who made the decision */
AND cld.ORSUS_CLD_ID = su.ID (+) --ALWAYS ON--
/* dec table has a description of the decisions made on the CL */
AND cld.BUDTE_CODE = dec.CODE (+) --ALWAYS ON--
/* org table has information on the companies, such as name, country, etc */
AND org.ID = cld.ORBUD_ORNNN_ID --ALWAYS ON--
/* pol table has information on the policy, such as Customer and Legacy Policy ref */
AND cld.BUPIY_ID = pol.ID --ALWAYS ON--
/* cus will bring information for the customer, name and country */
AND cus.ID = pol.ORCUR_ORNNN_ID --ALWAYS ON--
/* ex is the FX-Rate table */
AND ex.typ = 'FIX' --ALWAYS ON--   
/* filter to keep only latest FX-Rate */
AND ex.effect_to_dat > SYSDATE --ALWAYS ON-- 
/* CLD currency to Fx-rate table */
AND cld.D_BUPIY_ORCUY_CODE = ex.ORCUY_CODE --ALWAYS ON--  
/* rel table contains information on the relation between child and parent companies */
AND cld.ORBUD_ORNNN_ID = rel.ORBUD_CHILD_ORNNN_ID (+)
/* rel table filter to return only valid entries */
AND rel.EFFECT_TO_DAT (+) > SYSDATE
/* linking rel table to parent table */
and rel.ORBUD_PARENT_ORNNN_ID = par.ID (+) 
/* linking parent table and country table */
and par.ORCOY_ID = cop.ID (+) 
/* linking buyer table and country table */
and org.ORCOY_ID = cob.ID (+)  
/* linking buyer number to IBR rating table */
and cld.ORBUD_ORNNN_ID = ibr.ORBUD_ORNNN_ID (+) --ALWAYS ON--
/* filtering IBR rating table to return only valid entries */
and ibr.EFFECT_TO_DAT (+) > SYSDATE --ALWAYS ON--
/* BUPIY_ID is policy number, and it's linking to the policy table */
and cld.BUPIY_ID = pol.ID
/* linking buyer table to trade sector table, to map sector name */
and org.D_MAIN_TRADE_NACE_CODE = trdsec.TRADE_NACE_CODE (+)
/* linking trade sector to group sector */
and trdsec.BUTGP_ID = trdgrp.ID (+)
/* return only limits that are not 0. */
and cld.CLD_TOTAL_AMT > 0

group by 
cld.ID 
,cld.ORBUD_ORNNN_ID
,org.D_ORONE_FIRST_LINE_NAME
,org.ORCOY_ID  
,org.D_ORCOY_MAIN_NAME 
,cld.BUPIY_ID 
,pol.ORCUR_ORNNN_ID  
,cus.D_ORONE_FIRST_LINE_NAME 
,cus.D_ORCOY_MAIN_NAME 
,cld.CLA_RECORDED_DAT
,cld.CLD_STATUS_DAT
,cld.CLA_AMT 
,cld.CLD_FIRST_AMT 
,cld.CLD_SECOND_AMT 
,cld.CLD_TOTAL_AMT 
,cld.D_BUPIY_ORCUY_CODE 
,cld.BUCCY_ID
,com.NAME 
,cld.BUCTE_CODE
,cld.BUDTE_CODE
,dec.DES  
,cld.ORSUS_CLD_ID
,su.INDIV_UID
,su.D_ORIDL_INDIV_NAME 
,cld.EFFECT_TO_DAT
,ex.AMT
--,cld.D_BUPIY_LEG_SYS_REF_NR
,rel.ORBUD_PARENT_ORNNN_ID 
,par.D_ORONE_FIRST_LINE_NAME 
,par.ORCOY_ID 
,par.D_ORCOY_MAIN_NAME 
,cop.CODE
,cob.CODE
,org.D_ORNOL_STATUS_CODE
,ibr.FINAL_RATING 
,ibr.FINAL_RATING_TYP 
,pol.LEG_SYS_REF_NR
,org.D_MAIN_TRADE_NACE_CODE
,trdsec.NAME
,trdgrp.NAME

order by
cld.CLD_STATUS_DAT asc
--cld.EFFECT_TO_DAT desc
