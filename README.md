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

# CODING AND OUTPUT

```
import pandas as pd
from scipy import stats
import numpy as np
df = pd.read_csv("/content/bmi.csv")
df
```

<img width="401" height="563" alt="image" src="https://github.com/user-attachments/assets/8a486155-7521-4da3-916b-0610d32cec09" />

```
df.head()
```

<img width="350" height="258" alt="image" src="https://github.com/user-attachments/assets/e85d674b-08d4-4e95-adcb-1d5bab9610e9" />

```
df.dropna()
```

<img width="382" height="558" alt="image" src="https://github.com/user-attachments/assets/b650b934-74f5-4036-b41a-9802debf219a" />


```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Hs','Ws']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="517" height="438" alt="image" src="https://github.com/user-attachments/assets/0c63353c-03e0-4cee-b3db-4676c697a243" />

```
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df[['Hm','Wm']] = sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="529" height="438" alt="image" src="https://github.com/user-attachments/assets/eaf68bed-5651-418a-9c46-7e1306a40173" />

```
import pandas as pd
df = pd.read_csv("/content/titanic_dataset.csv")
df.columns
```


<img width="530" height="53" alt="image" src="https://github.com/user-attachments/assets/57a9608c-90b5-443b-b140-3fc33ec4fe12" />

```
from sklearn.feature_selection import SelectKBest
X = df[['PassengerId','Pclass','Age','SibSp','Parch','Fare']]
y = df['Survived']
from sklearn.feature_selection import f_classif
selector = SelectKBest(score_func=f_classif, k=4)
X_new = selector.fit_transform(X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features =X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```


<img width="522" height="42" alt="image" src="https://github.com/user-attachments/assets/26536feb-b163-4122-a10e-3fdfa6b05759" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```


<img width="520" height="225" alt="image" src="https://github.com/user-attachments/assets/b988b5b3-0a8e-4f0a-b710-87ff629c4068" />

```
tips.time.unique()
```


<img width="486" height="65" alt="image" src="https://github.com/user-attachments/assets/a6dadcba-6d89-4cac-99c5-bd80980ad712" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```


<img width="248" height="105" alt="image" src="https://github.com/user-attachments/assets/6c7f58b1-a662-4452-bb7c-d76f7929123b" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```


<img width="439" height="58" alt="image" src="https://github.com/user-attachments/assets/80d27ac1-f4c1-4113-aa4c-a2af1d47dada" />








# RESULT:
       Thus, Feature selection and Feature scaling has been used on the given dataset.
