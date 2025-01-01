import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None, criterion='gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None
    

    def fit(self, X, y):
        data = np.c_[X, y]  
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        X, y = data[:, :-1], data[:, -1]
        num_samples, num_features = X.shape

        # Stopping criteria
        if num_samples == 0 or len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return self._create_leaf(y)

        # Find the best split
        best_split = self._find_best_split(data, num_features)
        if best_split['gain'] == 0:
            return self._create_leaf(y)

        # Split the dataset
        left_data = data[data[:, best_split['feature']] <= best_split['threshold']]
        right_data = data[data[:, best_split['feature']] > best_split['threshold']]

        # Create subtree
        return {
            'feature': best_split['feature'],
            'threshold': best_split['threshold'],
            'left': self._build_tree(left_data, depth + 1),
            'right': self._build_tree(right_data, depth + 1)
        }

    def _create_leaf(self, y):
        values, counts = np.unique(y, return_counts=True)
        return {'leaf': True, 'prediction': values[np.argmax(counts)]}

    def _find_best_split(self, data, num_features):
        X, y = data[:, :-1], data[:, -1]
        best_split = {'feature': None, 'threshold': None, 'gain': 0}
        base_impurity = self._impurity(y)

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_impurity = self._impurity(y[left_indices])
                right_impurity = self._impurity(y[right_indices])
                weighted_avg_impurity = (
                    len(y[left_indices]) / len(y) * left_impurity +
                    len(y[right_indices]) / len(y) * right_impurity
                )

                gain = base_impurity - weighted_avg_impurity
                if gain > best_split['gain']:
                    best_split = {'feature': feature, 'threshold': threshold, 'gain': gain}

        return best_split

    def _impurity(self, y):
        if self.criterion == 'gini':
            values, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            values, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            return -np.sum(probabilities * np.log2(probabilities + 1e-9))
        else:
            raise ValueError("Unsupported criterion: " + self.criterion)

    def predict(self, X):
        return np.array([self._predict_single(row, self.tree) for row in X])

    def _predict_single(self, row, tree):
        if 'leaf' in tree:
            return tree['prediction']
        if row[tree['feature']] <= tree['threshold']:
            return self._predict_single(row, tree['left'])
        else:
            return self._predict_single(row, tree['right'])

# Example Usage
if __name__ == "__main__":
    # Create a simple dataset
    data = {
        'Feature1': [2, 3, 10, 19, 8],
        'Feature2': [1, 4, 9, 15, 6],
        'Label': [0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)

    X = df[['Feature1', 'Feature2']].values
    y = df['Label'].values

    # Initialize and train the decision tree using Gini Impurity
    tree_gini = DecisionTree(max_depth=None, criterion='gini')  # No predefined max depth
    tree_gini.fit(X, y)
    predictions_gini = tree_gini.predict(X)
    print("Predictions (Gini):", predictions_gini)
    print("Decision Tree (Gini):", tree_gini.tree)

    # Initialize and train the decision tree using Entropy
    tree_entropy = DecisionTree(max_depth=None, criterion='entropy')  # No predefined max depth
    tree_entropy.fit(X, y)
    predictions_entropy = tree_entropy.predict(X)
    print("Predictions (Entropy):", predictions_entropy)
    print("Decision Tree (Entropy):", tree_entropy.tree)
