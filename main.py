import re
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# Data Preprocessing.
# Remove "PassengerId" column.
train_df = train_df.drop(["PassengerId"], axis=1)

# Remove "Ticket" column.
train_df = train_df.drop(["Ticket"], axis=1)
test_df = test_df.drop(["Ticket"], axis=1)


# Missing Data.
# Remove "Cabin", but create a new feature "Deck".
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

# Fill the missing "Age" column's values (feature).
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # Compute random numbers between the mean, std and is_null.
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # Fill NaN values in "Age" column with random values generated.
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

train_df["Age"].isnull().sum()

# Fill the missing "Embarked" column's values (feature).
common_value = "S"
data = [train_df, test_df]

for dataset in data:
    dataset["Embarked"] = dataset["Embarked"].fillna(common_value)


# Converting Features.
# The "Fare" column (feature).
data = [train_df, test_df]

for dataset in data:
    dataset["Fare"] = dataset["Fare"].fillna(0)
    dataset["Fare"] = dataset["Fare"].astype(int)

# Remove "Name", but create a new feature "Title".
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # Extract titles.
    dataset["Title"] = dataset.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    # Replace titles with a more common title or as Rare.
    dataset["Title"] = dataset["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")
    # Convert titles into numbers.
    dataset["Title"] = dataset["Title"].map(titles)
    # Filling NaN with 0, to get safe.
    dataset["Title"] = dataset["Title"].fillna(0)

train_df = train_df.drop(["Name"], axis=1)
test_df = test_df.drop(["Name"], axis=1)

# The "Sex" column (feature).
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset["Sex"] = dataset["Sex"].map(genders)

# The "Embarked" column (feature).
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset["Embarked"] = dataset["Embarked"].map(ports)


# Creating Categories.
# The "Age" column (feature).
data = [train_df, test_df]

for dataset in data:
    dataset["Age"] = dataset["Age"].astype(int)
    dataset.loc[dataset["Age"] <= 11, "Age"] = 0
    dataset.loc[(dataset["Age"] > 11) & (dataset["Age"] <= 18), "Age"] = 1
    dataset.loc[(dataset["Age"] > 18) & (dataset["Age"] <= 22), "Age"] = 2
    dataset.loc[(dataset["Age"] > 22) & (dataset["Age"] <= 27), "Age"] = 3
    dataset.loc[(dataset["Age"] > 27) & (dataset["Age"] <= 33), "Age"] = 4
    dataset.loc[(dataset["Age"] > 33) & (dataset["Age"] <= 40), "Age"] = 5
    dataset.loc[(dataset["Age"] > 40) & (dataset["Age"] <= 66), "Age"] = 6
    dataset.loc[dataset["Age"] > 66, "Age"] = 6

# The "Fare" column (feature).
data = [train_df, test_df]

for dataset in data:
    dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 14.454), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 31), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 99), "Fare"] = 3
    dataset.loc[(dataset["Fare"] > 99) & (dataset["Fare"] <= 250), "Fare"] = 4
    dataset.loc[dataset["Fare"] > 250, "Fare"] = 5
    dataset["Fare"] = dataset["Fare"].astype(int)


# Creating new Features.
# The "Age_Class" column (feature).
data = [train_df, test_df]
for dataset in data:
    dataset["Age_Class"] = dataset["Age"] * dataset["Pclass"]

# The "Relatives" column (feature).
data = [train_df, test_df]
for dataset in data:
    dataset["Relatives"] = dataset["SibSp"] + dataset["Parch"]
    dataset.loc[dataset["Relatives"] > 0, "Not_Alone"] = 0
    dataset.loc[dataset["Relatives"] == 0, "Not_Alone"] = 1
    dataset["Not_Alone"] = dataset["Not_Alone"].astype(int)

# The "Fare_Per_Person" column (feature).
for dataset in data:
    dataset["Fare_Per_Person"] = dataset["Fare"] / (dataset["Relatives"] + 1)
    dataset["Fare_Per_Person"] = dataset["Fare_Per_Person"].astype(int)

# Building Machine Learning Models.
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Stochastic Gradient Descent (SGD).
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Random Forest.
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# Logistic Regression.
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# K Nearest Neighbor.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes.
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron.
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear Support Vector Machine.
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Decision Tree.
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

results = pd.DataFrame(
    {
        "Model": [
            "Support Vector Machines",
            "KNN",
            "Logistic Regression",
            "Random Forest",
            "Naive Bayes",
            "Perceptron",
            "Stochastic Gradient Decent",
            "Decision Tree",
        ],
        "Score": [
            acc_linear_svc,
            acc_knn,
            acc_log,
            acc_random_forest,
            acc_gaussian,
            acc_perceptron,
            acc_sgd,
            acc_decision_tree,
        ],
    }
)
result_df = results.sort_values(by="Score", ascending=False)
result_df = result_df.set_index("Score")
result_df.head(9)
