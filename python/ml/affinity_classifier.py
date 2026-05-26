from typing import List, Dict
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.preprocessing import StandardScaler
import numpy as np


class AffinityClassifier:
    def __init__(
            self,
            distance_metric: str = 'euclidean',
            affinity_method: str = 'inverse',
            preprocess: bool = False
    ) -> None:
        self.distance_metric = distance_metric
        self.affinity_method = affinity_method
        self.preprocess = preprocess
        self.feature_names = None
        self.trained = False
        self.centroids = None
        self.classes = None
        self.data_preprocessor = None

    def is_trained(self) -> bool:
        return self.trained

    def get_params(self) -> Dict:
        return {'distance_metric': self.distance_metric, 'preprocess': self.preprocess}

    def set_params(self, **params) -> None:
        if 'distance_metric' in params:
            self.distance_metric = params['distance_metric']
        if 'preprocess' in params:
            self.preprocess = params['preprocess']

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        self.data_preprocessor: StandardScaler | object = StandardScaler().fit(x)
        return self.data_preprocessor.transform(x)

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        if self.is_trained():
            if self.preprocess:
                x = self._preprocess(x)
            if self.distance_metric == 'euclidean':
                distances = euclidean_distances(x, self.centroids)
            elif self.distance_metric == 'manhattan':
                distances = manhattan_distances(x, self.centroids)
            elif self.distance_metric == 'cosine':
                distances = cosine_distances(x, self.centroids)
            else:
                raise ValueError('Error !!! Invalid distance_metric. Correct values = {euclidean, manhattan, cosine} ... ')
            return distances
        raise ValueError('Error !!! Model not trained ...')

    def _compute_affinity_confidence(self, x: np.ndarray) -> np.ndarray:
        if self.preprocess:
            x = self._preprocess(x)
        distances = self._compute_distances(x)
        if self.affinity_method == 'inverse':
            affinity_confidence = 1 / (1 + distances)
        elif self.affinity_method == 'rb_gaussian':
            gamma = np.sqrt(x.shape[1])
            affinity_confidence = np.exp(-gamma * np.power(distances, 2))
        else:
            raise ValueError('Error !!! Invalid affinity_method. Correct values = {inverse, rb_gaussian} ... ')
        return affinity_confidence

    def fit(self, data: np.ndarray, target: np.ndarray, feature_names: List[str] = None) -> None:
        self.trained = False
        self.feature_names = feature_names
        self.classes = np.unique(target)
        if self.preprocess:
            x = self._preprocess(data)
        else:
            x = data
        centroids = []
        for cls in self.classes:
            x_cls = x[target == cls]
            centroids.append(np.mean(x_cls, axis=0).tolist())
        self.centroids = np.array(centroids)
        self.trained = True

    def predict_distance(self, data: np.ndarray) -> np.ndarray:
        return self._compute_distances(data)

    def predict_affinity_confidence(self, data: np.ndarray) -> np.ndarray:
        return self._compute_affinity_confidence(data)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        if self.preprocess:
            data = self._preprocess(data)
        # distance = self.predict_distance(data)
        affinity_confidence = self.predict_affinity_confidence(data)
        # proba = softmax(distance, axis=1)
        proba = softmax(affinity_confidence, axis=1)
        return proba

    def predict(self, data: np.ndarray) -> np.ndarray:
        distances = self._compute_distances(data)
        indices = np.argmin(distances, axis=1)
        return self.classes[indices]

