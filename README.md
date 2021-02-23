# Titanic
Predicting survival of the Titanic sinking.

The Titanic repository contains: 
- 'data': has the raw train ('train.csv') and test ('test.csv') data 
- 'Titanic.ipynb': the notebook for this repository
- 'images': all the visualisations from 'Titanic.ipynb'
- 'submission.csv': csv file of predictions.

## The Problem
Many passengers did not survive the unfortunate sinking of the Titanic. The data and information of these passengers on at https://www.kaggle.com/c/titanic/overview. The following table indicates the variables in the provided data. 

| Variable | Definition | Key |
| ----------- | ----------- | ----------- |
| PassengerId | Unique identifiers for the passengers |
| Survival | Survival | 0 = No, 1 = Yes|
| Pclass |	Ticket class |	1 = 1st, 2 = 2nd, 3 = 3rd|
| Sex |	Sex	|
| Age |	Age in years|
| Sibsp |	# of siblings / spouses aboard the Titanic|	
|Parch|	# of parents / children aboard the Titanic|
|Ticket|	Ticket number|	
|Fare|	Passenger fare|	
|Cabin|	Cabin number|	
|Embarked	| Port of Embarkation |	C = Cherbourg, Q = Queenstown, S = Southampton|

**The objective is to predict which passengers were more likely to survive based on the information given in the train and test datasets**, named 'train.csv' and 'test.csv' respectively. 

Predictions were in submissions and an accuracy score is also received. 

## Data
Descriptive summaries below are for 'object', 'integer' and 'float' data types in 'train.csv'. 

'Object' variables
|        | Name                                 | Sex   | Ticket   | Cabin   | Embarked   |
|:-------|:-------------------------------------|:------|:---------|:--------|:-----------|
| count  | 891                                  | 891   | 891      | 204     | 889        |
| unique | 891                                  | 2     | 681      | 147     | 3          |
| top    | Hogeboom, Mrs. John C (Anna Andrews) | male  | CA. 2343 | G6      | S          |
| freq   | 1                                    | 577   | 7        | 4       | 644        |

'Integer' and 'float' variables
|       |   PassengerId |   Survived |     Pclass |      SibSp |      Parch |      Age |     Fare |
|:------|--------------:|-----------:|-----------:|-----------:|-----------:|---------:|---------:|
| count |       891     | 891        | 891        | 891        | 891        | 714      | 891      |
| mean  |       446     |   0.383838 |   2.30864  |   0.523008 |   0.381594 |  29.6991 |  32.2042 |
| std   |       257.354 |   0.486592 |   0.836071 |   1.10274  |   0.806057 |  14.5265 |  49.6934 |
| min   |         1     |   0        |   1        |   0        |   0        |   0.42   |   0      |
| 25%   |       223.5   |   0        |   2        |   0        |   0        |  20.125  |   7.9104 |
| 50%   |       446     |   0        |   3        |   0        |   0        |  28      |  14.4542 |
| 75%   |       668.5   |   1        |   3        |   1        |   0        |  38      |  31      |
| max   |       891     |   1        |   3        |   8        |   6        |  80      | 512.329  |


## Cleaning Data
Cleaning data involves finding and resolving missing data, duplicates, invalid data and irrelevant data. 

### Missing data
Proxies replaced missing data in the **'Age'**, **'Embarked'** and **'Fare'** variables across both loaded files. 
- Missing **'Embarked'** data was proxied with the mode and missing **'Fare'** data was proxied with the median value. 
- For **'Age'**, the missing data was proxied as the median age of individuals with matching *'Pclass'*,*'SibSp'* and *'Parch'*. It is clear that from observing the age distributions for each category in each relevant variable (see below), age distributions under each category for *'Pclass'*,*'SibSp'*. and *'Parch'* are distinct. Note that this is not the case for *'Embarked'*, *'Sex'* and *'Fare'*. The visualisations below show age distributions by: mean and standard deviation (left), a box plot (middle), and a violin plot (right) for each variable. 

*Embarked*
![categorical_feature_plotsEmbarked](https://github.com/Bennett-Heung/Titanic/blob/main/images/categorical_feature_plotsEmbarked.png)

*Fare*
![categorical_feature_plotsFare](https://github.com/Bennett-Heung/Titanic/blob/main/images/categorical_feature_plotsFare.png)

*Parch*
![categorical_feature_plotsParch](https://github.com/Bennett-Heung/Titanic/blob/main/images/categorical_feature_plotsParch.png)

*Pclass*
![categorical_feature_plotsPclass](https://github.com/Bennett-Heung/Titanic/blob/main/images/categorical_feature_plotsPclass.png)

*Sex*
![categorical_feature_plotsSex](https://github.com/Bennett-Heung/Titanic/blob/main/images/categorical_feature_plotsSex.png)

*SibSp*
![categorical_feature_plotsSibSp](https://github.com/Bennett-Heung/Titanic/blob/main/images/categorical_feature_plotsSibSp.png)

### Duplicates
No duplicates were found. 

### Invalid data
Only invalid 'Fare' data was suspected. Outliers of 'Fare' values of more than 300 were removed, while 'Fare' values of 0 were left unchanged due to uncertainty behind reasoning of them in the dataset for such numerous individuals. 

*Distribution of passenger fares: distribution plot (left), box plot (right)*
![numeric_feature_plotsFare](https://github.com/Bennett-Heung/Titanic/blob/main/images/numeric_feature_plotsFare.png)

'SibSp' and 'Parch' variables were also set up in the process to provide proxies for missing 'Age' data and for One Hot Encoding for the next sections. 

### Dropping data
'PassengerId', 'Name', 'Ticket and 'Cabin' (due to significant amounts of missing data) were not considered for the rest of the process. 

## Exploratory Data Analysis
The following visualisations show the distributions of passengers who did survive ('Survived'=1) and not survive ('Survived'=0).

*Age* 
![Line_chart_survivedAge](https://github.com/Bennett-Heung/Titanic/blob/main/images/Line_chart_survivedAge.png)

*Fare*
![Line_chart_survivedFare](https://github.com/Bennett-Heung/Titanic/blob/main/images/Line_chart_survivedFare.png)

*Embarked*
![bar_chart_survivedEmbarked](https://github.com/Bennett-Heung/Titanic/blob/main/images/bar_chart_survivedEmbarked.png)

*Parch*
![bar_chart_survivedParch](https://github.com/Bennett-Heung/Titanic/blob/main/images/bar_chart_survivedParch.png)

*Pclass*
![bar_chart_survivedPclass](https://github.com/Bennett-Heung/Titanic/blob/main/images/bar_chart_survivedPclass.png)

*Sex*
![bar_chart_survivedSex](https://github.com/Bennett-Heung/Titanic/blob/main/images/bar_chart_survivedSex.png)

*SibSp*
![bar_chart_survivedSibSp](https://github.com/Bennett-Heung/Titanic/blob/main/images/bar_chart_survivedSibSp.png)


**Findings**: 
- Less number of individuals survived than non-survivors
- First Ticket Class holders were more likely to survive 
- Females were more likely to surivive
- Passengers with less than two siblings/spouses aboard were more likely to survive
- Despite most survivors had no parents/children aboard, but in terms of proportion, of other survivors were more likely to survive
- Most survivors embarked from Southampton, but in terms of proportion, passengers who embarked from Cherbourg were more likely to survive 
- Not much to report on survival across age, however higher passenger fares showed more survival rates than lower passenger fares. 

*Correlations Matrix*
![corr_heatmap](https://github.com/Bennett-Heung/Titanic/blob/main/images/corr_heatmap.png)

**Findings**:

Correlations with 'Survived':
- Male passengers (-0.55) and the Third Ticket Class passengers (-0.32) were less likely to have survived.
- Passenger Fares (Fare) were most positively correlated to survival (0.26). 

Correlations of variables unrelated to 'Survived': 
- Age and Parch=2 were negatively correlated.
- Third Ticket Class (Pclass=3) was negatively correlated to Age and Fare also.
- Individuals with 2 parents/children aboard (Parch=2) were positively correlated to Fare and more-than-four siblings/spouses (SibSp = 4+).

Note that not all One Hot Encoding variables were included, as the concept of 'k-1' dummies for 'k' categories was applied as a 'k'-th dummy variable carries no new information and create multicollinearlity. 

## Modelling 

Standardising 'Age' and 'Fare' was the final step in feature engineering to improve the modelling process. 

A preview of the data for the modelling process is below.

|    |       Age |       Fare |   Pclass_2 |   Pclass_3 |   Sex_male |   SibSp_1 |   SibSp_2 |   SibSp_3 |   SibSp_4 |   SibSp_4+ |   Parch_1 |   Parch_2 |   Parch_2+ |   Embarked_Q |   Embarked_S |
|---:|----------:|-----------:|-----------:|-----------:|-----------:|----------:|----------:|----------:|----------:|-----------:|----------:|----------:|-----------:|-------------:|-------------:|
|  0 | -1.3487   |  0.946897  |          0 |          1 |          1 |         0 |         0 |         0 |         0 |          1 |         0 |         1 |          0 |            0 |            1 |
|  1 | -0.382058 |  0.836335  |          1 |          0 |          0 |         1 |         0 |         0 |         0 |          0 |         0 |         1 |          0 |            0 |            1 |
|  2 | -0.158986 | -0.573641  |          0 |          1 |          1 |         0 |         0 |         0 |         0 |          0 |         0 |         0 |          0 |            0 |            1 |
|  3 |  2.14609  | -0.0979794 |          0 |          0 |          0 |         0 |         0 |         0 |         0 |          0 |         0 |         0 |          0 |            0 |            1 |
|  4 |  1.84866  | -0.111344  |          1 |          0 |          1 |         1 |         0 |         0 |         0 |          0 |         0 |         0 |          0 |            0 |            1 |


Vanilla models were trained to set baseline Accuracy scores, which was the evaluation metric here. GridSearchCV was applied here to test and tune hyperparameters for each model in determination of the best model as the one with the highest Accuracy score. The best model was deployed in the next section.

The following models were built:
- K-Neighbors Classifier
- Decision Tree Classifier
- Logistic Regression
- Support Vector Classification (SVC)
- Random Forest Classifier
- Gradient Boosting Classifier. 

The Accuracy of the vanilla models closely ranged from 0.778 to 0.815. 
|                            |   Accuracy |
|:---------------------------|-----------:|
| GradientBoostingClassifier |   0.815318 |
| SVC                        |   0.810842 |
| RandomForestClassifier     |   0.805193 |
| LogisticRegression         |   0.80405  |
| KNeighborsClassifier       |   0.784955 |
| DecisionTreeClassifier     |   0.778157 |


The following table indicates the hyperparameters tuned for each model. 
|                                 | models                                      | parameters                                                                                                                                   |
|:--------------------------------|:--------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| ('KNeighborsClassifier',)       | KNeighborsClassifier()                      | {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}                                                                                             |
| ('DecisionTreeClassifier',)     | DecisionTreeClassifier(random_state=42)     | {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}                                                                                               |
| ('LogisticRegression',)         | LogisticRegression(random_state=42)         | {'C': [0.0001, 0.001, 0.1, 1, 10, 50, 100]}                                                                                                  |
| ('SVC',)                        | SVC(random_state=42)                        | {'C': [0.0001, 0.001, 0.1, 1, 10, 50, 100], 'kernel': ['linear', 'rbf']}                                                                     |
| ('RandomForestClassifier',)     | RandomForestClassifier(random_state=42)     | {'n_estimators': [50, 100, 150, 200], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}                                                          |
| ('GradientBoostingClassifier',) | GradientBoostingClassifier(random_state=42) | {'n_estimators': [50, 100, 150, 200], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]} |

The following table shows the best models by highest Accuracy (using GridSearchCV), for each model after tuning hyperparameters. 

|                            | Best_model_params                                            |   Best_model_score |
|:---------------------------|:-------------------------------------------------------------|-------------------:|
| GradientBoostingClassifier | {'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 150} |           0.832203 |
| RandomForestClassifier     | {'max_depth': 5, 'n_estimators': 200}                        |           0.82882  |
| DecisionTreeClassifier     | {'max_depth': 4}                                             |           0.818701 |
| LogisticRegression         | {'C': 100}                                                   |           0.814188 |
| SVC                        | {'C': 1, 'kernel': 'rbf'}                                    |           0.810842 |
| KNeighborsClassifier       | {'n_neighbors': 3}                                           |           0.790599 |

**Gradient Boosting Classifier** *(learning_rate = 0.1, max_depth = 6, n_estimators = 50)* had the highest Accuracy score of 0.823 amongst all the best models. This model will be deployed to predict the test data.

<insert feature_importances> 

The key feature importance variables in relation to survival were: gender (Sex_male), passenger fare (Fare), age (Age) and the Third Ticket Class (Pclass=3).

## Deploying solution

The same process was used on test_data to prepare the data for the survival predictions, which includes:
- Proxies for the missing Fare and Age data* 
- One Hot Encoding for Pclass, SibSp, Parch and Embarked variables
- Standardised Age and Fare data.

* Note: one individual did not have an age proxy for missing data, given that she was the only individual without an age shown with Pclass=3, SibSp=0 and Parch=2+. In turn, the proxy from train_data. 

After the data preparation, the model was deployed for submission. The submission file, 'submission.csv' contained two columns: 
- PassengerId (sorted in any order)
- Survived (contains your binary predictions: 1 for survived, 0 for deceased). 
