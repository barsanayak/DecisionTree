import pandas as pd
from sklearn.model_selection import train_test_split
import scipy as sc
import numpy as np


class Attribute:
    # Value is one of the possibilities that the attribute can have.
    def __init__(self, id, parent_attribute_value):
        self.id = id
        self.parent_attribute_value = parent_attribute_value

    def set_classification(self, classification):
        self.classification = classification

class Node:
    def __init__(self, attribute, child_nodes):
        self.attribute = attribute
        self.child_nodes = child_nodes


class Tree:
    def __init__(self, x_train, y_train, default):
        self.x_train = x_train
        self.y_train = y_train
        self.default = default

    def train(self):
        # add a column y_train, at the front:
        result = np.insert(self.x_train, 0, self.y_train, axis=1)
        self.root = self._compute_node(pd.DataFrame(data=result))

    def predict(self, vector):
        node = self.root
        while node.attribute.id is not None:
            column_index = node.attribute.id - 1
            next_level = False
            for child in node.child_nodes:
                if vector[column_index] == child.attribute.parent_attribute_value:
                    node = child
                    next_level = True
                    break
            if not next_level:
                return self.default
        return node.attribute.classification

    def _compute_node(self, df, parent_attribute_value=None):
        if df.empty:
            return None
        # 0 is for classification values.
        best_column = 1
        max_entropy = self._attribute_entropy(df, best_column)
        for column_index in range(2, df.shape[1]):
            entropy = self._attribute_entropy(df, column_index)
            if max_entropy < entropy:
                best_column = column_index
                max_entropy = entropy

        gb = df.groupby(best_column, axis=0)
        child_nodes = np.array([])
        for attribute_value, index_values in gb.groups.items():
            y_value = df[df[best_column] == attribute_value][0]
            if y_value.empty:
                leaf_attribute = Attribute(None, attribute_value)
                leaf_attribute.set_classification(self.default)
                child_nodes = np.append(child_nodes, Node(leaf_attribute, None))
            elif y_value.unique().size == 1:
                leaf_attribute = Attribute(None, attribute_value)
                leaf_attribute.set_classification(y_value.unique()[0])
                child_nodes = np.append(child_nodes, Node(leaf_attribute, None))
            else :
                child_nodes = np.append(child_nodes, self._compute_node(df.loc[index_values], attribute_value))

        return Node(Attribute(best_column, parent_attribute_value), child_nodes)

    def _attribute_entropy(self, df, column_index):
        gb = df.groupby(column_index, axis=0)
        total_rows = len(df.index)
        total_entropy = 0

        for unique_value in gb.groups:
            y_value = df[df[column_index] == unique_value][0]
            y_entropy = self._entropy(y_value)
            total_entropy -= y_entropy * y_value.size/total_rows
        return total_entropy


    def _entropy(self, pd_column):
        # calculates the probabilities
        p_data = pd_column.value_counts() / len(pd_column)

        # input probabilities to get the entropy
        entropy = sc.stats.entropy(p_data)
        return entropy


def main():
    mushroom_df = pd.read_csv('MushroomDataSet.csv')
    y_input = mushroom_df.values[:, 1]
    x_input = mushroom_df.values[:, 2:]
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.9)
    tree = Tree(x_train=x_train, y_train=y_train, default='p')
    tree.train()
    correct_pred = 0
    total = 0
    for x, y in zip(x_test, y_test):
        actual = tree.predict(x)
        total += 1
        print (actual, y)
        if actual == y:
            correct_pred +=1
    print (1.0*correct_pred/total)

if __name__ == '__main__': main()