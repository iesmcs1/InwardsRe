# ECap-Dashboard

## 2021Q2 Run

- 36042B1 0119:
  - Presenting a small difference of 86k between this code (1.725m) and the IM (1.638m).
  - This is due to this code using 50% of run-off, while the IM used 47.5%
  - This difference is due to the way the Months Elapsed are calculated.


## Update 2021Q1
From 2021Q1 onwards, we will be using the Reporting Runs with Reserve Risks (RR)  
due to changes in how we deal with diversification benefits.

## Reporting Runs

- RR are Reserve Risk runs.
- Allocations (SO_REPORTING) should be taken from non-RR schema. Everything else from RR.
- For the ECap Dashboards, the runs are non-RR.

| Period | Schema (non-RR) | Schema (RR) |
|:------:|:---------------:|:-----------:|
2021 Q4  | CALC6854        | CALC6855    | 
2021 Q3  | CALC6820        | CALC6821    | 
2021 Q2  | CALC6788        | CALC6789    | 
2021 Q1  | CALC6754        | CALC6756    | 
2020 Q4  | CALC6724        | CALC6725    | 
2020 Q3  | CALC6673        | CALC6678    | 
2020 Q2  | CALC6619        | CALC6621    | 
2020 Q1  | CALC6573        | CALC6574    | 
2019 Q4  | CALC6492        | CALC6496    | 
2019 Q3  | CALC6425        | CALC6427    | 
2019 Q2  | CALC6326        | CALC6328    | 
2019 Q1  | CALC6264        | CALC6265    | 
2018 Q4  | CALC6182        | CALC6179    | 

## SO_REPORTING
### Columns
- `EC_CONSUMPTION_GR`: ECap allocation for QS only
- `EC_CONSUMPTION_ND`: ECap allocation for the QS + any associated layers of XL. ***We use this one***.
