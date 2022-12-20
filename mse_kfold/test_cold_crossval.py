import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from random import sample
import random
import cold_crossval

np.random.seed(29)
a = cold_crossval.data_reduction("../pokedex.csv", ["status"])

shake_data = a[["name", "status", "type_number", 
        "height_m", "weight_kg", "abilities_number", 
        "total_points", "hp", "attack", 
        "defense", "sp_attack", "sp_defense", 
        "speed", "egg_type_number"]].dropna(axis = 0).to_numpy()
train_split = round(0.9*shake_data.shape[0])
train_inds = random.sample(list(range(shake_data.shape[0])),train_split)
train_data = shake_data[train_inds,:]
test_inds = list(set(range(shake_data.shape[0])).difference(set(train_inds)))
test_data = shake_data[test_inds,:]

tree = DecisionTreeClassifier(ccp_alpha = 0.001, max_depth = 3)
tree.fit(train_data[:, 2:6], train_data[:, 1].astype(int))
preds = tree.predict(test_data[:, 2:6])

def test_overall_shape():
	out = cold_crossval.data_reduction("../pokedex.csv", ["status"])
	assert out.shape == (1028, 51)

def test_overall_type():
	out = cold_crossval.data_reduction("../pokedex.csv", ["status"])
	assert type(out) == pd.DataFrame

def test_mse_type():
    out = cold_crossval.cold_classification(test_data[:, 1], preds)
    assert type(out) == np.float64

def test_kfold_type():
    out = cold_crossval.cold_kfoldcv(data = shake_data,
         col_names = ["name", "status", "type_number", "height_m", "weight_kg", "abilities_number", "total_points", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "egg_type_number"], 
         inputs = ["total_points", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed"], 
         output = "status", 
         k = 10)
    assert type(out) == np.float64