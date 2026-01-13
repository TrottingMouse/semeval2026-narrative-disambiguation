from sklearn.linear_model import LinearRegression
import csv
import json

# Define a simple metaclassifier using logistic regression using the data from everything.csv

class MetaClassifier:
    def __init__(self, data_file='everything.csv'):
        self.data_file = data_file
        self.model = LinearRegression()
        self.X = []
        self.y = []
        self._load_data()
        self._train_model()

    def _load_data(self):
        with open(self.data_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                features = list(map(float, row[1:]))  # All columns except first are features
                label = int(row[0])  # Last column is the label
                self.X.append(features)
                self.y.append(label)

    def _train_model(self):
        self.model.fit(self.X, self.y)

    def predict(self, features):
        return self.model.predict([features])[0]



