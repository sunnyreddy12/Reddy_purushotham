# Proposal for Landslide detection Project

## 1. Purushotham reddy

### Project Title
**Proposal for Landslide detection Project**

### Author
Purushotham reddy

### Author's Links
- [GitHub Profile](https://www.linkedin.com/in/purushotham-reddy-654774159)
- [LinkedIn Profile](https://www.linkedin.com/in/purushotham-reddy-654774159)


## 2. Background

### What is it about?
This research project aims to comprehensively understand and mitigate the risks associated with landslides in Uttarakhand, particularly during the monsoon period.

### Why does it matter?
Landslides during the monsoon season disrupt daily life and communication infrastructure in Uttarakhand. It's crucial to address this issue to safeguard lives and properties.

### Research Questions
- What are the primary causes of landslides in Uttarakhand?
- How can we effectively monitor and predict landslide events?

## 3. Data

### Data Sources
- Global Landslide Catalog (GLC)
- Global Fatal Landslide Database (GFLD)
- Rainfall data from the Tropical Rainfall Measuring Mission (TRMM)

### Data Details
- The final dataset is made by joining all the above sets and filtering the rows
- Data Size: 6MB
- Data Shape: Varies (e.g., GLC contains various types of landslide reports)
- Time Period: Historical data of 15 years
- Each row represents: Details of landslide events, The Rainfall data and the soil rigidity module.
- Data Dictionary:
  - Columns Names: 	depth	landslide	antecedent_1days	antecedent_2days	antecedent_3days	antecedent_4days	antecedent_5days	antecedent_6days	antecedent_7days	antecedent_8days	...	antecedent_21days	antecedent_22days	antecedent_23days	antecedent_24days	antecedent_25days	antecedent_26days	antecedent_27days	antecedent_28days	antecedent_29days	antecedent_30days
  - Potential Values (for categorical variables):[0,1]
- Target/Label Variable: Landslide
- Feature/Predictor Variables: Possibility of a landfall

This proposal provides a structured approach to understanding and Predicting landslides in Uttarakhand, backed by comprehensive data analysis and research.
