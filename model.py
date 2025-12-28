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

# ============ data preparation

# delete non-needed attrs
y_train_data = train_data["Transported"]

train_data = train_data.drop("Transported", axis=1)
train_data = train_data.drop("Name", axis=1)

train_data['Deck'] = train_data['Cabin'].str.split('/').str[0]
train_data['CabinNumber'] = train_data['Cabin'].str.split('/').str[1]
train_data['Side'] = train_data['Cabin'].str.split('/').str[2] 

train_data = train_data.drop("Cabin", axis=1)
train_data["CabinNumber"] = train_data["CabinNumber"].astype("float64")
# spents = [train_data["RoomService"], 
#           train_data["ShoppingMall"], 
#           train_data["FoodCourt"],train_data["VRDeck"], train_data["Spa"]]

# train_data["spent-money"] = sum(spents)

# print(train_data.info())
# prepare data: filling empty fields, encoding non-int attrs, scaling attrs.
number_attrs = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNumber"]
cat_attrs = ["VIP", "CryoSleep", "HomePlanet", "Destination", "Side", "Deck"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, number_attrs),
    ("cat", OneHotEncoder(), cat_attrs)
])

train_data_prepared = full_pipeline.fit_transform(train_data, y_train_data)

print(train_data_prepared)






