# Proposal for Landslide detection Project

## 1. Purushotham reddy -Final Semester

### Project Title
**Proposal for Landslide detection Project**

### Author
Purushotham reddy

### Author's Links
- [GitHub Profile](https://www.linkedin.com/in/purushotham-reddy-654774159)
- [LinkedIn Profile](https://www.linkedin.com/in/purushotham-reddy-654774159)
- [Presentaion Ppt](https://docs.google.com/presentation/d/1M1A2yExo7eLTOEpvrmhAZ1h9nkXAbbNF/edit?usp=sharing&ouid=104889154945175504308&rtpof=true&sd=true)
- [Streamlit App](https://reddypurushotham606finalpy-zvfkjf594w999gfgcoy7rs.streamlit.app/)

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
  - Columns Names: 	depth,	landslide,	antecedent_data for 30 days	
  - Potential Values (for categorical variables):[0,1]
- Target/Label Variable: Landslide
- Feature/Predictor Variables: Possibility of a landfall

This proposal provides a structured approach to understanding and Predicting landslides in Uttarakhand, backed by comprehensive data analysis and research.

## 4. Exploratory Data Analysis (EDA)

### Data Exploration using Jupyter Notebook
In this project, I performed an extensive exploratory data analysis (EDA) using a Jupyter Notebook. The main focus was on the target variable and selected features, with all other columns excluded. The following steps outline the actions taken during the EDA:

The provided code includes functions for preprocessing rainfall and landslide data, merging both datasets, and performing additional data transformations. The steps I took include:

### 1. Preprocessing Rainfall Data
- Replaced missing values and set column names.
- Filled NaN values and calculated the depth based on intensity and frequency.
- Set the time column as the index.

### 2. Preprocessing Landslide Data
- Set column names and the time column as the index.
- Extracted year, month, and day information.

### 3. Merging Both Data
I successfully merged the preprocessed rainfall and landslide datasets.

### 4. Antecedent Rainfall Calculation
Defined a function (`antecedent_rainfall`) to calculate antecedent rainfall based on a specified number of days.

### 5. Data Scaling
Standardized the data using RobustScaler.

### 6. Data Splitting
Implemented StratifiedShuffleSplit to split the data into training and test sets of 80% and 20%, ensuring the preservation of class distribution.

### 7. Data Exploration
Displayed descriptive statistics, class distribution, and correlation matrices.

### 8. Dimensionality Reduction
Used T-SNE, PCA, and Truncated SVD for dimensionality reduction and visualization.

### 9. Undersampling
Undertook undersampling to balance the class distribution for subsequent model training.

### 10. Visualization of Clusters
Visualized clusters using reduced-dimensional representations.

### 11. Model Training
I split the data into test and train data with a 20 and 80 split and proceeded with model training.
## 5. Model Training

### Models Used
The predictive analytics will involve the following classification algorithms:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- CatBoost Classifier
- XGBoost

### Training Process
1. **Data Splitting:** The dataset will be divided into training and testing sets using an 80/20 split. This is achieved through the `train_test_split` function from scikit-learn.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

2. **Model Initialization:** Instances of each classifier will be created.

    ```python
    log_reg = LogisticRegression()
    knn = KNeighborsClassifier()
    svc = SVC()
    dt_classifier = DecisionTreeClassifier()
    cat_boost = CatBoostClassifier()
    xg_boost = xgb.XGBClassifier()
    ```

3. **Model Training:** The models will be trained using the training data.

    ```python
    log_reg.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    dt_classifier.fit(X_train, y_train)
    cat_boost.fit(X_train, y_train)
    xg_boost.fit(X_train, y_train)
    ```

### Python Packages Used
- scikit-learn
- xgboost
- imbalanced-learn
- catboost
- matplotlib
- seaborn
- numpy
- pandas
- joblib

### Development Environment
jupyter, github codespaces, visual studio google colab, streamlit, snowflake were used in developing the code.

### Performance Measurement
Performance will be assessed using various metrics, including but not limited to:
- ROC-AUC score
- Precision-Recall curve
- Confusion matrix
- Classification report

Cross-validation and grid search will be employed to fine-tune hyperparameters and ensure robust model evaluation.

## 6. Application of the Trained Models

## Web App Development

I have developed a web application to allow users to interact with the trained predictive analytics models. The web app is built using Streamlit.

### Implementation

1. **Streamlit Integration:** I integrated the Streamlit framework to create a user-friendly and responsive web application.

2. **Model Loading:** The trained machine learning models are loaded into the web app for real-time predictions.

    ```python
    # Example code to load models in Streamlit
    import streamlit as st
    from joblib import load

    # Load trained models
    log_reg_model = load('logistic_regression_model.joblib')
    knn_model = load('knn_model.joblib')
    svc_model = load('svc_model.joblib')
    dt_model = load('decision_tree_model.joblib')
    cat_boost_model = load('catboost_model.joblib')
    xg_boost_model = load('xgboost_model.joblib')
    ```

3. **User Interface:** I designed the web app with an intuitive user interface where users can input relevant data for predictions.

4. **Prediction Display:** Upon user input, the web app leverages the trained models to generate predictions and displays the results.

### Usage

To run the web app locally, ensure Streamlit is installed:

## 7. Conclusion

### Summary of My Work and Potential Applications

In this project, I focused on detecting landslides based on antecedent rainfall patterns, specifically identifying occurrences when the rainfall exceeds 180mm over a 15-day period. The developed threshold models provide crucial insights for establishing a Landslide Early Warning System in our study region. These thresholds serve as key indicators for anticipating and potentially mitigating landslide events.

### Potential Applications

1. **Early Warning System Development:** The rainfall thresholds identified in my work can play a pivotal role in developing an Early Warning System for landslides in our study region. This enhances our region's preparedness and response measures.

2. **Sensor Implementation:** I propose further implementation of sensors at the original landslide site to provide real-time data for analyzing rainfall-induced landslides. The incorporation of various sensors will contribute to refining and improving the accuracy of the threshold models in our specific region.

### CatBoost as My Optimal Choice

Among the machine learning models I evaluated, CatBoost emerges as my optimal choice for landslide prediction. It demonstrates outstanding performance with high accuracy, precision, and recall. The obtained accuracy of 98.47% underscores CatBoost's efficacy in handling the complexity of landslide prediction, establishing it as a valuable tool for mitigating risks associated with this natural phenomenon.

### Limitations

While my study provides valuable insights, it's important to acknowledge its limitations:

1. **Regional Specificity:** The developed models and thresholds are tailored to our study region, and their generalizability to other regions may vary.

2. **Data Dependency:** The accuracy and reliability of my models are contingent on the quality and quantity of available data.

### Lessons Learned

Throughout my project, I learned several valuable lessons:

1. **Model Selection Impact:** The choice of machine learning models significantly influences predictive performance.

2. **Data Quality Matters:** The accuracy of predictions heavily relies on the quality and relevance of input data.

### Future Research Directions

Looking forward, I suggest the following directions for future research:

1. **Generalization Studies:** Investigate the applicability of my developed models to different geographical and climatic conditions.

2. **Enhanced Sensor Networks:** Expand and optimize sensor networks for more comprehensive data collection.

3. **Real-Time Monitoring:** Implement real-time monitoring systems to provide timely alerts and improve landslide risk management strategies.

These insights and directions form the basis for continued advancements in landslide prediction and mitigation efforts in our region.

## 8. References

1. Caine N (1980) The rainfall intensity: duration control of shallow landslides and debris flows.

2. Hong, M., Kim, J. & Jeong, S. Landslides (2018) 15: 523

3. Kanungo, D. P., & Sharma, S. (2014). Rainfall thresholds for prediction of shallow landslides around Chamoli-Joshimath region, Garhwal Himalayas, India. Landslides, 11(4), 629-638.




