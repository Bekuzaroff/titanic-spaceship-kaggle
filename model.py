import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv("train.csv")

# ================ testing-set, we don't tounch it before we have fully-ready ml model
test_data = pd.read_csv("test.csv") 

 # ============== look for data and datatypes all the statistics
print(train_data.head())
print(train_data.info())
print(train_data.describe())








