import pandas as pd
from sklearn.model_selection import train_test_split

from poker.DecisionTree import Tree


def main():
    poker_df = pd.read_csv('poker-hand-training.csv')
    y_input = poker_df.values[:, -1]
    x_input = poker_df.values[:, 0:-1]

    x_train, x_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.2)
    tree = Tree(x_train=x_train, y_train=y_train, default=0)
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