/*
Query to generate extract to be used for Solvency II purposes.

Update WHERE clause cld.EFFECT_TO_DAT if extracting data at quarter end
*/

select
cld.ID Credit_Limit_ID
,cld.ORBUD_ORNNN_ID Buyer_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(org.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Buyer_Name -- If we don't do this, excel will read the data incorrectly
,cob.CODE Buyer_Country_ISO
,rel.ORBUD_PARENT_ORNNN_ID Parent_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(par.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Parent_Name -- If we don't do this, excel will read the data incorrectly
,cop.CODE Parent_Country_ISO
,pol.LEG_SYS_REF_NR Legacy_Policy
,TRUNC(cld.CLA_RECORDED_DAT) CLA_Date
,TRUNC(cld.CLD_STATUS_DAT) CLD_Date
,cld.CLA_AMT CLA_Amount
,cld.CLD_TOTAL_AMT CLD_Total_Amount
,cld.D_BUPIY_ORCUY_CODE Policy_Currency
,cld.BUCTE_CODE Commitment_Type
,MAX('') AS CLD_EXPIRY_DATE 
,TO_CHAR(MAX(SYSDATE), 'DD/MM/YYYY') AS REFRESH_DATE

from
ORABUP0.TBBU_CREDIT_LIMITS cld
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS org
,ORABUP0.TBBU_POLICIES pol
,ORABUP0.TBOR_RISK_GROUP_RELATIONS rel
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS par
,ORABUP0.TBOR_COUNTRIES cop
,ORABUP0.TBOR_COUNTRIES cob

where
cld.BUCCY_ID = 14 --COMMITMENT_CATEGORY_NUMBER--
AND cld.CLD_STATUS_CODE = 'FINL' --query was returning DRAFT decisions also, and creating duplicates--

/* Comment first row and Uncomment second row
   if downloading data for Quarter end */
AND cld.EFFECT_TO_DAT > SYSDATE --ALWAYS ON--
--AND cld.EFFECT_TO_DAT >= '31-DEC-2021' --ALWAYS ON--
/* update above accordingly */

and org.ORCOY_ID = cob.ID (+)

AND org.ID = cld.ORBUD_ORNNN_ID --ALWAYS ON--
AND cld.ORBUD_ORNNN_ID = rel.ORBUD_CHILD_ORNNN_ID (+)
AND rel.EFFECT_TO_DAT (+) > SYSDATE
and rel.ORBUD_PARENT_ORNNN_ID = par.ID (+)
and par.ORCOY_ID = cop.ID (+)  
AND cld.BUPIY_ID = pol.ID --ALWAYS ON--  
and cld.CLD_TOTAL_AMT > 0

group by 
cld.ID 
,cld.ORBUD_ORNNN_ID
,org.D_ORONE_FIRST_LINE_NAME
,cob.CODE 
,cld.BUPIY_ID  
,cld.CLA_RECORDED_DAT
,cld.CLD_STATUS_DAT
,cld.CLA_AMT 
,cld.CLD_FIRST_AMT 
,cld.CLD_SECOND_AMT 
,cld.CLD_TOTAL_AMT 
,cld.D_BUPIY_ORCUY_CODE 
,cld.BUCCY_ID
,cld.BUCTE_CODE
,cld.BUDTE_CODE
,cld.ORSUS_CLD_ID 
,cld.EFFECT_TO_DAT
,rel.ORBUD_PARENT_ORNNN_ID 
,par.D_ORONE_FIRST_LINE_NAME  
,cop.CODE 
,pol.LEG_SYS_REF_NR