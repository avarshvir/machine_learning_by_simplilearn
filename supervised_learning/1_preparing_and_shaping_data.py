import pandas as pd
import numpy as np
df = pd.read_csv('../datasets/titanic.csv')
print(df.head())
print(df.columns)
print(df.describe())
print(df.info())

#creating new feature
df['Travelalone'] = np.where((df['SibSp'] + df['Parch']) > 0,0,1).astype('uint8')
df1 = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis = 1)
print(df.isna().sum())
print("for df1 : \n",df1.isna().sum())

df['Age'].fillna(df1['Age'].median(skipna=True), inplace =True)
print(df1.head())

df_titanic = pd.get_dummies(df1, columns=['Pclass','Embarked','Sex'],drop_first=True)
print(df_titanic.head())

X = df_titanic.drop(['Survived'],axis=1)
y = df_titanic['Survived']

from sklearn.preprocessing import MinMaxScaler,StandardScaler
trans_MM = MinMaxScaler()
trans_SS = StandardScaler()

df_MM =  trans_MM.fit_transform(X)
print(pd.DataFrame(df_MM))

df_SS = trans_SS.fit_transform(X)
print(pd.DataFrame(df_SS))

