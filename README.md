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
import pandas as pd
import numpy as np
df=pd.read_csv("bmi.csv")
df
<img width="486" height="626" alt="image" src="https://github.com/user-attachments/assets/fd117c20-e758-4ea7-87d6-7c18e914e00a" />
<img width="456" height="636" alt="image" src="https://github.com/user-attachments/assets/0c00175f-475c-4261-9c2d-947a54215312" />
df.dropna()
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
<img width="960" height="53" alt="image" src="https://github.com/user-attachments/assets/63704ee5-b885-416e-bea1-d88117576042" />
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
<img width="456" height="556" alt="image" src="https://github.com/user-attachments/assets/ceee2ab4-dbc1-4cb8-8e22-0de58061cc46" />
df1=pd.read_csv("bmi.csv")
df2=pd.read_csv("bmi.csv")
df3=pd.read_csv("bmi.csv")
df4=pd.read_csv("bmi.csv")
df5=pd.read_csv("bmi.csv")
<img width="506" height="644" alt="image" src="https://github.com/user-attachments/assets/98a2d2b2-a346-46e7-9017-6de80e6a62dc" />
df1
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
<img width="464" height="543" alt="image" src="https://github.com/user-attachments/assets/a9708607-fe71-4a2e-8058-9024a3200b42" />
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
<img width="492" height="636" alt="image" src="https://github.com/user-attachments/assets/85934eb1-40fc-4bd1-bbc4-d4597267b46c" />
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
<img width="511" height="642" alt="image" src="https://github.com/user-attachments/assets/6658c09e-c203-421e-b3df-bd9714acb311" />
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
<img width="498" height="627" alt="image" src="https://github.com/user-attachments/assets/df38e847-c145-4199-bceb-c0c331ac14cf" />
df4
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv("income(1) (1).csv")
data
<img width="996" height="290" alt="image" src="https://github.com/user-attachments/assets/6c13373c-d224-471f-8b83-b25ca6205f10" />
data1=pd.read_csv('/content/titanic_dataset (1).csv')
data1
<img width="868" height="316" alt="image" src="https://github.com/user-attachments/assets/2e15b61c-d39b-4f24-a567-6d36a99e2096" />
data1=data1.dropna()
x=data1.drop(['Survived','Name','Ticket'],axis=1)
y=data1['Survived']
data1['Sex']=data1['Sex'].astype('category')
data1['Cabin']=data1['Cabin'].astype('category')
data1['Embarked']=data1['Embarked'].astype('category')
data1['Sex']=data1['Sex'].cat.codes
data1['Cabin']=data1['Cabin'].cat.codes
data1['Embarked']=data1['Embarked'].cat.codes
data1
<img width="886" height="329" alt="image" src="https://github.com/user-attachments/assets/05cbcb00-7dc2-4f21-979a-13b6a3c0cbd8" />
k=5
selector=SelectKBest(score_func=chi2,k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
x_encoded=pd.get_dummies(x)
selector=SelectKBest(score_func=chi2,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
<img width="886" height="56" alt="image" src="https://github.com/user-attachments/assets/f76f114e-715e-40a1-9910-0ea543c58350" />
from sklearn.feature_selection import SelectKBest,f_regression
import pandas as pd
selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
<img width="879" height="64" alt="image" src="https://github.com/user-attachments/assets/c579a9aa-d86a-49a2-8c43-fa23b6119da9" />
from sklearn.feature_selection import SelectKBest,mutual_info_classif
import pandas as pd
selector=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
<img width="882" height="69" alt="image" src="https://github.com/user-attachments/assets/1fb5936a-451d-4f88-b097-8462301f6b45" />
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
<img width="850" height="118" alt="image" src="https://github.com/user-attachments/assets/735ceb3f-7f00-4692-9e1b-edb94bc6ed20" />
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_selection=model.feature_importances_ threshold=0.1
selected_features=x.columns[feature_selection>threshold]
print("Selected Features:")
print(selected_features)
<img width="810" height="65" alt="image" src="https://github.com/user-attachments/assets/748f7d34-14b9-4213-8b44-24d4f21aa3e8" />
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance=model.feature_importances_ threshold=0.15
selected_features=x.columns[feature_importance>threshold]
print("Selected Features:")
print(selected_features)
<img width="458" height="76" alt="image" src="https://github.com/user-attachments/assets/225bba2e-6d9f-4a5d-9fa7-3933d799787d" />












# RESULT:
 Thus  performed Feature Scaling and Feature Selection process and saved the data file.
