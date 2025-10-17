# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("D:\Data Science\\CSV files\\bmi.csv")

df.dropna(inplace=True)  

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

features = ['Gender', 'Height', 'Weight']
target = 'Index'

X = df[features]
y = df[target]

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'MaxAbsScaler': MaxAbsScaler(),
    'RobustScaler': RobustScaler()
}
scaled_dfs = {}
for name, scaler in scalers.items():
    scaled = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled, columns=[f"{col}_{name}" for col in features])
    scaled_dfs[name] = scaled_df

final_df = pd.concat([df, *scaled_dfs.values()], axis=1)

selector = SelectKBest(score_func=f_classif, k=3)
selected_features = selector.fit_transform(X, y)

selected_df = pd.DataFrame(selected_features, columns=['Selected1', 'Selected2', 'Selected3'])
final_df = pd.concat([final_df, selected_df], axis=1)

final_df.to_csv("bmi_scaled_selected.csv", index=False)
print("Data cleaned, scaled, selected, and saved to 'bmi_scaled_selected.csv'")
```
Data cleaned, scaled, selected, and saved to 'bmi_scaled_selected.csv'
<img width="1468" height="798" alt="DS Ex4 - Output 1" src="https://github.com/user-attachments/assets/a1c8251f-12f0-4e71-8d25-e4bc022798d2" />
<img width="1467" height="793" alt="DS Ex4 - Output 2" src="https://github.com/user-attachments/assets/da16cabc-d8be-4068-a031-e8ee92a045a6" />

# RESULT:
We have performed Feature Scaling and Feature Selection processes and saved the changes to a dataset csv file 'bmi_scaled_selected.csv'. 
