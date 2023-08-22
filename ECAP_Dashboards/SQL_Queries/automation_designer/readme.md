# Automation Designer

The queries in this folder are designed to be used in Automation Designer, in Toad.

## What to do when NOT in Automation Designer

When running these queries in "Editor" window, you will need to change `&CALCVAR.` to
the reported run number, as to end up with `CALCXXXX`, where XXXX is the reported run.
Here's a full example:

```SQL
SELECT *
FROM CALC6724.SO_REPORTING
```

## Steps to set up Automation Desginer

- Open **Automation Designer** and create a new app.

![open automation designer and create app](/ECAP_Dashboards/SQL_Queries/automation_designer/images/01-open_auto_designer_and_create_app.gif)

- Select the newly created app and go to **Control** and then **Create Variable**.

![create variable](/ECAP_Dashboards/SQL_Queries/automation_designer/images/02-create_variable.gif)

- Double click the created variable and edit to the desired reported run number (e.g.: 6754, 6724, etc).

![edit variable](/ECAP_Dashboards/SQL_Queries/automation_designer/images/03-edit_variable.gif)

- Now add the step to export the dataset.

![add export step](/ECAP_Dashboards/SQL_Queries/automation_designer/images/04-add_export_datasaet.gif)

- Double click the export step and modify the below:
1. Choose "Delimited Text".
2. Choose folder and file name to be exported to.
3. Select **Encoding: UTF-8**.
4. Select **Delimiter: Tab**.
5. Apply changes to save options.

![edit export step](/ECAP_Dashboards/SQL_Queries/automation_designer/images/05-export_options.png)
