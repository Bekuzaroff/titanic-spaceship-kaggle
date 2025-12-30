import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


data_set  = pd.read_csv("train.csv")


 # ============== look for data and datatypes all the statistics
# print(data_set.head())
# print(data_set.info())
# print(data_set.describe())

# ============ data preparation

# delete non-needed attrs
encoder = OrdinalEncoder()
y_data_set = data_set[["Transported"]]
y_data_set = encoder.fit_transform(y_data_set)
y_data_set = y_data_set.ravel()

data_set["Transported"] = y_data_set
y_data_set = data_set["Transported"]


data_set = data_set.drop("Transported", axis=1)
data_set = data_set.drop("Name", axis=1)

data_set['Deck'] = data_set['Cabin'].str.split('/').str[0]
data_set['CabinNumber'] = data_set['Cabin'].str.split('/').str[1]
data_set['Side'] = data_set['Cabin'].str.split('/').str[2] 

data_set = data_set.drop("Cabin", axis=1)
data_set["CabinNumber"] = data_set["CabinNumber"].astype("float64")
spents = ["RoomService", "ShoppingMall", "FoodCourt", "VRDeck", "Spa"]

data_set["spent-money"] = data_set[spents].sum(axis=1)

ids = data_set["PassengerId"].str.split("_")
data_set["long_id"] = ids.str[0]
data_set["short_id"] = ids.str[1]

data_set["max_place_money_spent"] = data_set[spents].max(axis=1)

data_set["TotalSpent_log"] = np.log1p(data_set["spent-money"]) 

group_sizes = data_set["long_id"].value_counts()
data_set["GroupSize"] = data_set["long_id"].map(group_sizes) 
data_set["IsAlone"] = (data_set["GroupSize"] == 1).astype(int)  


# prepare data: filling empty fields, encoding non-int attrs, scaling attrs.
number_attrs = ["IsAlone", "GroupSize", "TotalSpent_log","long_id", "short_id", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNumber", "spent-money", "max_place_money_spent"]
cat_attrs = ["VIP", "CryoSleep", "HomePlanet", "Destination", "Side", "Deck"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, number_attrs),
    ("cat", cat_pipeline, cat_attrs)
])

train_data, test_data = train_test_split(data_set, test_size=0.1, random_state=42)
y_train_data, y_test_data = train_test_split(y_data_set, test_size=0.1, random_state=42)

train_data_prepared = full_pipeline.fit_transform(train_data)

tree_reg = RandomForestClassifier(n_estimators=200, max_depth=10, max_features="sqrt", bootstrap=True)
tree_reg.fit(train_data_prepared, y_train_data)

# param_grid = {
#     "n_estimators": [50, 100, 150, 200],
#     "max_features": ["sqrt", "log2", "auto"],
#     "max_depth": [5, 10, 20, 30],
#     "bootstrap": [False, True]
# }
# gr_search = RandomizedSearchCV(tree_reg, param_grid)
# gr_search.fit(train_data_prepared, y_train_data)

# print(gr_search.best_estimator_)

# print(gr_search.best_params_)


# testing with test data-set


some_data_test = test_data.iloc[150:160]
some_data_test_prepared = full_pipeline.transform(some_data_test)

print("test predicts: ", tree_reg.predict(some_data_test_prepared))
print("test actual labels: ", np.array(y_test_data.iloc[150:160]))

full_data_prepared = full_pipeline.transform(test_data)
full_predictions = tree_reg.predict(full_data_prepared)
full_labels = np.array(y_test_data)

mae = mean_squared_error(full_labels, full_predictions)
rmse = np.sqrt(mae)

print("mae: ", mae)
print("rmse: ", rmse)











