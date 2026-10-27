\# Stroke Prediction — Summary \& Conclusion



\## Objective

Predict stroke risk from patient health and demographic data, with a focus 

on identifying at-risk individuals (high recall) given the clinical cost 

of missed cases.



\## Data

\- Source: Kaggle Stroke Prediction Dataset (5,110 patients, 12 features)

\- Missing values: \~4% of BMI values, imputed using KNN imputation; a 

&#x20; `bmi\_missing` flag was retained

\- `smoking\_status` contained \~30% "Unknown" values, largely explained by 

&#x20; young patients (median age \~23) who were never asked — kept as its own 

&#x20; category rather than imputed



\## Key EDA Findings

\- \*\*Age\*\* is the strongest predictor: stroke rate rises from \~0.2% (under 20) 

&#x20; to \~20% (80+)

\- \*\*Hypertension\*\* and \*\*heart disease\*\* roughly triple/quadruple stroke risk

\- \*\*Glucose level\*\* shows a moderate positive relationship with stroke

\- \*\*BMI\*\* shows little independent relationship with stroke

\- \*\*Work type\*\* appeared predictive but was found to be confounded by age 

&#x20; (e.g., self-employed patients average 60 years old vs. 6.8 for children)



\## Modeling Approach

\- Class imbalance (95% no-stroke / 5% stroke) addressed using SMOTE on the 

&#x20; training set only

\- Compared Logistic Regression vs. Random Forest — both showed similar AUC 

&#x20; (\~0.78-0.79), but Logistic Regression achieved much higher recall at 

&#x20; matched thresholds

\- Discovered and corrected a feature-scaling issue: unscaled logistic 

&#x20; regression produced coefficients that contradicted EDA (e.g., hypertension 

&#x20; appearing "protective"); applying StandardScaler fixed this and improved 

&#x20; AUC from 0.789 to 0.844

\- Performed a threshold sweep (0.3-0.7) to balance recall against precision 

&#x20; and accuracy, since the default 0.5 threshold is not well-suited to 

&#x20; imbalanced classification



\## Final Model

\*\*Logistic Regression, scaled features, SMOTE-balanced training, threshold = 0.6\*\*

\- ROC-AUC: 0.844

\- Recall (stroke): 0.76

\- Precision (stroke): 0.17

\- Accuracy: 0.81



\## Interpretation

Top predictors (by scaled coefficient): age, average glucose level, 

hypertension, smoking status. These align with established clinical risk 

factors for stroke, supporting the model's validity beyond pure statistical fit.



\## Limitations

\- Precision remains low — in a real deployment, this model would need to 

&#x20; function as a \*screening\* tool (flagging patients for further review), 

&#x20; not a diagnostic one

\- SMOTE-generated synthetic data may not perfectly reflect real patient 

&#x20; distributions, especially for categorical features

\- Dataset is relatively small (5,110 rows); results may not generalize to 

&#x20; broader populations without further validation



\## Possible Next Steps

\- Try XGBoost or gradient boosting with proper hyperparameter tuning

\- Explore cost-sensitive learning instead of SMOTE

\- Validate on an external stroke dataset

