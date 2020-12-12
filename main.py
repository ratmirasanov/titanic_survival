import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# Data Preprocessing.
# Remove "PassengerId" column.
train_df = train_df.drop(["PassengerId"], axis=1)

# Missing Data. Remove "Cabin", but create a new feature "Deck".
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset["Cabin"] = dataset["Cabin"].fillna("U0")
    dataset["Deck"] = dataset["Cabin"].map(
        lambda x: re.compile("([a-zA-Z]+)").search(x).group()
    )
    dataset["Deck"] = dataset["Deck"].map(deck)
    dataset["Deck"] = dataset["Deck"].fillna(0)
    dataset["Deck"] = dataset["Deck"].astype(int)

train_df = train_df.drop(["Cabin"], axis=1)
test_df = test_df.drop(["Cabin"], axis=1)
print(train_df.head(10))
