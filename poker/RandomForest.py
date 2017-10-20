import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as sc
import numpy as np
from .DecisionTree import Tree
class Forest:
    def __init__(self, x_train, y_train, default, tree_count=10):
        self.x_train = x_train
        self.y_train = y_train
        self.default = default
        self.tree_count = tree_count

    def train(self):
        self.tree_list = []
        x_train, x_test, y_train, y_test = train_test_split(self.x_train, self.y_train, test_size=0.5)
        for num in range(0, self.tree_count):
            tree = Tree(x_train, y_train, 'p')
            tree.train()
            self.tree_list.append(tree)


    def predict(self, vector):
        res = []
        for tree in self.tree_list:
            res.append(tree.predict(vector))
        return max(set(res), key=res.count)


def main():
    mushroom_df = pd.read_csv('MushroomDataSet.csv')
    y_input = mushroom_df.values[:, 1]
    x_input = mushroom_df.values[:, 2:]
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.2)
    forest = Forest(x_train=x_train, y_train=y_train, default='p')
    forest.train()

    correct_pred = 0
    total = 0
    for x, y in zip(x_test, y_test):
        actual = forest.predict(x)
        total += 1
        print (actual, y)
        if actual == y:
            correct_pred +=1
    print (1.0*correct_pred/total)

if __name__ == '__main__': main()