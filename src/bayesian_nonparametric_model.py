import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyod.models.abod import ABOD
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy


# Nonparametric Bayes
class BayesianNonparametricalModel:
    _kernel_class: KernelDensity
    _kernel_params: dict
    _kernel_instances: dict[str: KernelDensity]
    _klass_probabilities: np.array
    _classes: np.array


    def __init__(self, kernel: KernelDensity, kernel_params: dict) -> None:
        self._kernel_class = kernel
        self._kernel_params = kernel_params
        self._kernel_instances = {}

    def fit(self, x, y) -> None:
        if x.shape[0] != y.shape[0]:
            raise Exception
        
        self._classes, counts = np.unique(y, return_counts=True)
        klass_probs = counts / np.sum(counts)
        self._klass_probabilities = {self._classes[i]: klass_probs[i] for i in range(len(klass_probs))}

        # fitting kernels for each class
        for klass in self._classes:
            klass_indexes = np.where(y == klass)
            x_k = x[klass_indexes]            
            kernel_k = self._kernel_class(**self._kernel_params)
            kernel_k.fit(x_k)
            self._kernel_instances[klass] = kernel_k
            
    def predict_proba(self, x) -> np.array:
        if len(x.shape) != 2:
            raise Exception
        
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

    def predict(self, x) -> np.array:
        probs = self.predict_proba(x)
        indexes_max_elements = probs.argmax(axis=1)
        return self._classes[indexes_max_elements]
    
    @property
    def classes(self):
        return copy(self._classes)
    
    @property
    def klass_probabilities(self):
        return copy(self._klass_probabilities)
    
    @property
    def kernel_params(self):
        return copy(self._kernel_params)


if __name__ == "__main__":
    df = pd.read_csv("data/Dry_Bean_Dataset_cleaned.csv", index_col=0)
    x = df.drop(["Class"], axis=1).to_numpy()
    y = df["Class"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    kernel = KernelDensity
    kernel_params = {
        "kernel": "linear",
        "bandwidth": 0.2
    }
    bnm = BayesianNonparametricalModel(kernel, kernel_params)
    bnm.fit(X_train, y_train)
    pred = bnm.predict(X_test)
    print(accuracy_score(y_test, pred))
