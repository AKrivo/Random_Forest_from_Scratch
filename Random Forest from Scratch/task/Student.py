import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(52)


def plot(accuracy_history: list, filename='plot_RF'):
    # function to visualize learning process at stage 4

    n_trees = len(accuracy_history)

    plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    plt.plot(accuracy_history)

    plt.xlabel('Number of Trees')
    plt.ylabel('Acc')
    plt.xticks(np.arange(0, n_trees, 60))
    plt.title('Effect of Number of Trees on Model Accuracy')
    plt.grid()

    # plt.subplot(1, 2, 2)
    # plt.plot(accuracy_history)
    #
    # plt.xlabel('Epoch number')
    # plt.ylabel('Accuracy')
    # plt.xticks(np.arange(0, n_epochs, 4))
    # plt.title('Accuracy on test dataframe from epoch')
    # plt.grid()

    plt.savefig(f'{filename}.png')


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = \
        train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    # Make your code here...

# Stage 1/6
DT = DecisionTreeClassifier(random_state=52)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_val)
task_1 = accuracy_score(y_val, y_pred).round(3)


# print(task_1)

# Stage 2/6
def create_bootstrap(X, y):
    mask = []
    for i in range(len(X_train)):
        mask.append(np.random.choice(range(len(X_train))))
    return X[mask], y[mask]


X_masked, y_masked = create_bootstrap(X_train, y_train)


# print(y_masked[:10].tolist())


# Stage 3/6

class RandomForestClassifier():
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error

        self.forest = []
        self.is_fit = False

    def fit(self, X_train, y_train):

        # Your code for Stage 3 here
        for _ in range(self.n_trees):
            DT = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features='sqrt',
                min_impurity_decrease=self.min_error)
            X, y, = create_bootstrap(X_train, y_train)
            self.forest.append(DT.fit(X, y))

        self.is_fit = True

    def predict(self, X_test):

        if not self.is_fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')

        y = np.zeros(shape=(self.n_trees, X_test.shape[0]))
        for i in range(len(self.forest)):
            y[i] = (self.forest[i].predict(X_test))
        final = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=y)

        return final


# x = RandomForestClassifier()
# x.fit(X_train, y_train)
# pred = x.predict(X_val)
# print(pred.tolist())

# Stage 5/6

# task_5 = accuracy_score(y_val, pred).round(3)


# print(task_5)


# Stage 6/6


def eval(X, y, n_trees, X_val, y_val):
    pred_list = []
    for i in tqdm(range(1, n_trees)):
        RF = RandomForestClassifier(i, np.iinfo(np.int64).max, 1e-6)
        RF.fit(X, y)
        temp = RF.predict(X_val)
        res = accuracy_score(y_val, temp).round(3)
        pred_list.append(res)
    return pred_list


task_6 = eval(X_train, y_train, 600, X_val, y_val)
plot(task_6, filename='plot_RF')

print(task_6[:20])
