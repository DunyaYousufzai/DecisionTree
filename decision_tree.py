import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

class decision_tree:
    def __init__(self, file):
        self.file = file

    def data_selection(self, a, b,c):
       global rx, ry
       data = pd.read_csv(self.file)
       rx = data.iloc[:,a:b].values
       ry = data.iloc[:,c].values

    def trainig(self):
        global reg
        reg = DecisionTreeRegressor(random_state = 0)
        reg.fit(rx, ry)
    
    def plot(self, title,xlabel,ylabel):
        gridX = np.arange(min(rx), max(rx), 0.01)
        gridX = gridX.reshape((len(gridX), 1))
        plt.scatter(rx, ry, color = "blue")
        plt.plot(gridX, reg.predict(gridX), color = "brown")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def predict(self, f):
        prediction_y = reg.predict([[f]])
        print(prediction_y)

poly = decision_tree("decision_tree.csv")
poly.data_selection(1,2,2)
poly.trainig()
poly.plot("polynomial Model", "position","salary")
poly.predict(8)