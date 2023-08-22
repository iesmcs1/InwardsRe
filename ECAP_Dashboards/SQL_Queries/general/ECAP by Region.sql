/*When running this query, the "Variables" window will appear.
Make sure "Literal" is selected and click "Ok".*/

SELECT
    a.ULTIMATE_COUNTRY,
    b.BUYER_REGION,
    CASE
        /*Multiple lines for same ATRE_REGION due to formatting*/
        WHEN b.BUYER_REGION = 'Africa' THEN 'Africa'
        WHEN b.BUYER_REGION IN ('China', 'Indian Sub Continent', 'North Asia', 'SE Asia') THEN 'Asia'
        WHEN b.BUYER_REGION = 'Russia, Ukraine, CIS' THEN 'CIS'
        WHEN b.BUYER_REGION IN ('Balkans', 'Baltic') THEN 'Emerging Europe'
        WHEN b.BUYER_REGION IN ('Bulgaria, Hungary, Romania') THEN 'Emerging Europe'
        WHEN b.BUYER_REGION IN ('Czech Republic, Slovakia', 'Poland') THEN 'Emerging Europe' 
        WHEN b.BUYER_REGION IN ('Brazil', 'Caribbean RS 5', 'Central-America', 'South America') THEN 'Latin America'
        WHEN b.BUYER_REGION IN ('Middle East', 'Turkey') THEN 'Middle East & North Africa'
        WHEN b.BUYER_REGION IN ('Canada', 'Mexico', 'United States') THEN 'North America'
        WHEN b.BUYER_REGION IN ('Australia, New Zealand', 'Oceania') THEN 'Pacific Region'
        WHEN b.BUYER_REGION IN ('Austria, Switzerland', 'Belgium, Luxembourg') THEN 'Western Europe'
        WHEN b.BUYER_REGION IN ('Cyprus, Greece, Malta', 'Denmark') THEN 'Western Europe'
        WHEN b.BUYER_REGION IN ('Finland', 'France', 'Germany', 'Ireland') THEN 'Western Europe'
        WHEN b.BUYER_REGION IN ('Italy', 'Netherlands', 'Norway', 'Portugal') THEN 'Western Europe'
        WHEN b.BUYER_REGION IN ('Spain', 'Sweden', 'UK', 'UK Territories') THEN 'Western Europe'
        ELSE 'NOT_CLASSIFIED'
    END AS ATRE_REGION,
    SUM(EC_CONSUMPTION_ND) AS ECAP
FROM
    /*Update Reporting run below*/
    CALCXXXX.SO_REPORTING a,
    CALCXXXX.RW_ATRADIUS_COUNTRY b
    /*Update Reporting run above*/
WHERE
    MODEL_TYPE = 'IR'
    AND a.ULTIMATE_COUNTRY = b.REGION_NAME
GROUP BY
    a.ULTIMATE_COUNTRY,
    b.BUYER_REGION
