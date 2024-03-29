import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy


X_Y_DIMENSION_ERROR = "x.shape[0] != y.shape[0]! x and y must has the same first dimension."


# Nonparametric Bayes
class BayesianNonparametricalModel:
    _kernel_class: KernelDensity
    _kernel_params: dict
    _kernel_instances: dict[str: KernelDensity]
    _klass_probabilities: np.array
    _classes: np.array


    def __init__(self, kernel_params: dict) -> None:
        self._kernel_class = KernelDensity
        self._kernel_params = kernel_params
        self._kernel_instances = {}

    def fit(self, x: np.array, y: np.array) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(X_Y_DIMENSION_ERROR)
        
        self._classes, counts = np.unique(y, return_counts=True)
        klass_probs = counts / np.sum(counts)
        self._klass_probabilities = {self._classes[i]: klass_probs[i] for i in range(len(klass_probs))}

        for klass in self._classes:
            klass_indexes = np.where(y == klass)
            x_k = x[klass_indexes]
            
            kernel_k = self._kernel_class(**self._kernel_params)
            kernel_k.fit(x_k)
            self._kernel_instances[klass] = kernel_k
            
    def predict_proba(self, x: np.array) -> np.array:
        probs_arr = []
        for klass in self._classes:
            f_xk = np.exp(self._kernel_instances[klass].score_samples(x))
            f_k = self._klass_probabilities[klass]
            p_k = f_xk * f_k
            probs_arr.append(p_k)
        probs_arr = np.array(probs_arr).T # [[first element], ... , [last element]]
        norm_values = probs_arr.sum(axis=1)
        norm_values[norm_values == 0] = 1
        probs = probs_arr / norm_values[:, None]
        return probs

    def predict(self, x: np.array) -> np.array:
        probs = self.predict_proba(x)
        indexes_max_elements = probs.argmax(axis=1)
        return self._classes[indexes_max_elements]
    
    @property
    def classes(self):
        return copy.copy(self._classes)
    
    @property
    def klass_probabilities(self):
        return copy.copy(self._klass_probabilities)
    
    @property
    def kernel_params(self):
        return copy.copy(self._kernel_params)


if __name__ == "__main__":
    df = pd.read_csv("data/Dry_Bean_Dataset_cleaned.csv", index_col=0)
    x = df.drop(["Class"], axis=1).to_numpy()
    y = df["Class"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    kernel_params = {
        "kernel": "linear",
        "bandwidth": 0.2
    }

    bnm = BayesianNonparametricalModel(kernel_params)
    bnm.fit(X_train, y_train)
    pred = bnm.predict(X_test)
    print(accuracy_score(y_test, pred))
