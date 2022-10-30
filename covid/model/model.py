import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

icu_data = pd.read_csv("model/ICU.csv",index_col = False)
icu_data.drop("WINDOW",axis=1,inplace = True)

icu_data.dropna(inplace=True)
icu_data.drop_duplicates(keep='last',inplace=True)

y = icu_data["ICU"]
x = icu_data.drop("ICU",axis=1)

x.drop("PATIENT_VISIT_IDENTIFIER",axis=1,inplace=True)

x["AGE_PERCENTIL"] = x["AGE_PERCENTIL"].str.replace("th","")
x["AGE_PERCENTIL"] = x["AGE_PERCENTIL"].str.replace("Above 90","100")

x["AGE_PERCENTIL"] = x["AGE_PERCENTIL"].astype(str).astype(float).astype(int)

np.random.seed(10)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

clf = RandomForestClassifier()
clf.fit(x_train,y_train)


print(clf.score(x_test,y_test))

# save model to disc
filename = 'final_model.sav'
joblib.dump(clf,filename)