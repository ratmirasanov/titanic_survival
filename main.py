import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style


test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")
train_df.head(8)
