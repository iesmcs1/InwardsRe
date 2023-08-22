select
--*
cld.ID Credit_Limit_ID
,cld.ORBUD_ORNNN_ID Buyer_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(org.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Buyer_Name -- If we don't do this, excel will read the data incorrectly
,org.D_ORNOL_STATUS_CODE Buyer_Status 
--,ibr.FINAL_RATING IBR_Value
--,ibr.FINAL_RATING_TYP IBR_Type
,org.ORCOY_ID Buyer_Country_ID
,org.D_ORCOY_MAIN_NAME Buyer_Country
,cob.CODE Buyer_Country_ISO
,rel.RELATED_BUYER_TYP Relation_Type
,rel.ORBUD_PARENT_ORNNN_ID Parent_Number
,TRIM('+' FROM (TRIM('"' FROM (SUBSTR(par.D_ORONE_FIRST_LINE_NAME, 0, 70))))) AS Parent_Name -- If we don't do this, excel will read the data incorrectly
,par.ORCOY_ID Parent_Country_ID
--,par.D_ORCOY_MAIN_NAME Parent_Country
--,cop.CODE Parent_Country_ISO
,cld.BUPIY_ID Policy_Number
,pol.LEG_SYS_REF_NR Legacy_Policy
,pol.ORCUR_ORNNN_ID Customer_Number 
,cus.D_ORONE_FIRST_LINE_NAME Customer_Name 
,cus.ORCOY_ID Customer_Country_ID
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
,cld.CUST_REF_NR Customer_Reference
,cld.D_BUPIY_ORCUY_CODE Policy_Currency
,cld.BUCCY_ID Commitment_Category_Number
,com.NAME Commitment_Category
,cld.BUCTE_CODE Commitment_Type
,cld.BUDTE_CODE Decision_Type_Code
,dec.DES Decision_Type 
,su.INDIV_UID Underwriter_User_ID
,cld.ORSUS_CLD_ID Underwriter_System_User_ID
,su.D_ORIDL_INDIV_NAME Underwriter_Name
,cld.D_COVER_COND_FLAG Cover_Condition
,LISTAGG(con.BUCDE_CODE,' ### ') within group (order by con.BUCDE_CODE) as Condition_Codes --NOT WORKING CORRECT--
,LISTAGG(con.TEXT,' ### ') within group (order by con.BUCDE_CODE) as Condition_Text --NOT WORKING CORRECT--
,cld.HISTORIC_CODE Historic_Code
,TRUNC(cld.EFFECT_TO_DAT) Effective_to_Date

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
,ORABUP0.TBBU_OVERALL_OUTPUT_IBR ibr

where
--cld.ID = 100635969 --100598983 --97720000 --82904000 --95739718 --CREDIT_LIMIT_ID--
--and cld.ORBUD_ORNNN_ID = 895082 --(3429686) --BUYER_NUMBER--
--org.ORCOY_ID = 5 --BUYER_COUNTRY_ID-- 
--cld.BUPIY_ID in (1046868,1046871)--1600--1017863 --1600 --1016736 --145112 --POLICY_NUMBER--
--and cld.CLD_STATUS_DAT > '01-jul-2019' --CLD_DATE--
--and CLD_TOTAL_AMT < CLA_AMT
cld.BUCCY_ID = 14 --COMMITMENT_CATEGORY_NUMBER--
--and BUDTE_CODE in 'DC01' --DECISION_TYPE_CODE--
--and cld.ORSUS_CLD_ID = 8 --73128 --UNDERWRITER_SYSTEM_USER_ID-- 
--and cld.D_COVER_COND_FLAG in 'Y' --COVER_CONDITION--
--and ROWNUM < 100

and cld.EFFECT_TO_DAT > SYSDATE --ALWAYS ON--
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
and cld.ORBUD_ORNNN_ID = ibr.ORBUD_ORNNN_ID (+) --ALWAYS ON--
and ibr.EFFECT_TO_DAT (+) > SYSDATE --ALWAYS ON--
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
--,cld.D_BUPIY_LEG_SYS_REF_NR
,cld.HISTORIC_CODE
,rel.RELATED_BUYER_TYP 
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

order by
cld.CLD_STATUS_DAT asc
--cld.EFFECT_TO_DAT desc
