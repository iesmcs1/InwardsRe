/* Update WHERE clause cld.EFFECT_TO_DAT if extracting data at quarter end */

select
cld.ID Credit_Limit_ID
,cld.ORBUD_ORNNN_ID Buyer_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(org.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Buyer_Name -- If we don't do this, excel will read the data incorrectly
,cld.BUPIY_ID Policy_Number
,pol.LEG_SYS_REF_NR Legacy_Policy
,cld.BUCCY_ID Commitment_Category_Number
,com.NAME Commitment_Category
,cld.BUCTE_CODE Commitment_Type
,con.BUCDE_CODE Condition_Code
,con.DAT CLD_Expiry_Date
,TO_CHAR(MAX(SYSDATE), 'DD/MM/YYYY') AS REFRESH_DATE

from
ORABUP0.TBBU_CREDIT_LIMITS cld
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS org
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS cus
,ORABUP0.TBBU_POLICIES pol
,ORABUP0.TBOR_CURRENCY_EXCHANGE_RATES ex
,ORABUP0.TBBU_CONDITION_VARIABLES con
,ORABUP0.TBBU_COMMITMENT_CATEGORIES com
,ORABUP0.TBOR_SYSTEM_USERS su
,ORABUP0.TBBU_DECISION_TYPES dec
,ORABUP0.TBOR_RISK_GROUP_RELATIONS rel
,ORABUP0.TBOR_NON_NCM_ORGANISATIONS par
,ORABUP0.TBOR_COUNTRIES cop
,ORABUP0.TBOR_COUNTRIES cob

where
cld.BUCCY_ID = 14 --COMMITMENT_CATEGORY_NUMBER--

/* Comment first row and Uncomment second row
   if downloading data for Quarter end */
AND cld.EFFECT_TO_DAT > SYSDATE --ALWAYS ON--
--AND cld.EFFECT_TO_DAT >= '31-DEC-2021' --ALWAYS ON--
/* update above accordingly */

--and (cld.EFFECT_TO_DAT > SYSDATE -3 and cld.EFFECT_TO_DAT < SYSDATE) --ALWAYS ON--
and com.ID = cld.BUCCY_ID --ALWAYS ON--
and cld.ORSUS_CLD_ID = su.ID (+) --ALWAYS ON--
and cld.BUDTE_CODE = dec.CODE (+) --ALWAYS ON--
and cld.ID = con.BUCLT_ID (+) --ALWAYS ON--  
and org.ID = cld.ORBUD_ORNNN_ID --ALWAYS ON--
and cld.BUPIY_ID = pol.ID --ALWAYS ON--
and cus.ID = pol.ORCUR_ORNNN_ID --ALWAYS ON--
and ex.typ = 'FIX' --ALWAYS ON--   
and ex.effect_to_dat > SYSDATE --ALWAYS ON-- 
and cld.D_BUPIY_ORCUY_CODE = ex.ORCUY_CODE --ALWAYS ON--  
and cld.ORBUD_ORNNN_ID = rel.ORBUD_CHILD_ORNNN_ID (+)
and rel.EFFECT_TO_DAT (+) > SYSDATE
and rel.ORBUD_PARENT_ORNNN_ID = par.ID (+) 
and par.ORCOY_ID = cop.ID (+) 
and org.ORCOY_ID = cob.ID (+)  
and con.BUCDE_CODE in 'T502'
--and cld.ORBUD_ORNNN_ID = ibr.ORBUD_ORNNN_ID (+) --ALWAYS ON--
--and ibr.EFFECT_TO_DAT (+) > SYSDATE --ALWAYS ON--
and cld.BUPIY_ID = pol.ID 

group by 
cld.ID
,cld.ORBUD_ORNNN_ID
,org.D_ORONE_FIRST_LINE_NAME
,org.ORCOY_ID  
,org.D_ORCOY_MAIN_NAME 
,cld.BUPIY_ID 
,pol.ORCUR_ORNNN_ID  
,cus.D_ORONE_FIRST_LINE_NAME 
,cus.ORCOY_ID
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
,cld.D_COVER_COND_FLAG 
,cld.EFFECT_TO_DAT
,ex.AMT
,cld.CUST_REF_NR
,cld.D_BUPIY_LEG_SYS_REF_NR
,cld.HISTORIC_CODE
,rel.RELATED_BUYER_TYP 
,rel.ORBUD_PARENT_ORNNN_ID 
,par.D_ORONE_FIRST_LINE_NAME 
,par.ORCOY_ID 
,par.D_ORCOY_MAIN_NAME 
,cop.CODE
,cob.CODE
,org.D_ORNOL_STATUS_CODE
,con.DAT
,con.BUCDE_CODE
--,ibr.FINAL_RATING 
--,ibr.FINAL_RATING_TYP 
,pol.LEG_SYS_REF_NR

order by
cld.CLD_STATUS_DAT asc
--cld.EFFECT_TO_DAT desc
