from typing import List, Dict, Tuple
from .ml_utils import get_cluster_labels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
from copy import deepcopy


class GatingModel:
    def __init__(
            self,
            e_id: str,
            data: np.ndarray,
            target: np.ndarray,
            lower_quantile_value: float,
            upper_quantile_value: float,
            k: int = 5,
            feature_names: List[str] = None,
            do_shuffle: bool = True
    ) -> None:
        self.estimator_id = e_id
        self.n_folds = k
        self.lower_quantile_value, self.upper_quantile_value = lower_quantile_value, upper_quantile_value
        self.data = data
        self.regression_target = target
        self.target = get_cluster_labels(self.regression_target, self.lower_quantile_value, self.upper_quantile_value)
        if do_shuffle:
            self.data = shuffle(self.data, random_state=42)
            self.target = shuffle(self.target, random_state=42)
            self.regression_target = shuffle(self.regression_target, random_state=42)
        self.feature_names = feature_names
        self.trained_total = False
        self.trained_cv = False
        self.gating_model = None
        self.gating_model_cv = None
        self.train_indices = None
        self.test_indices = None
        self._create_preprocessors()
        self._split_cv_folds()

    def is_trained_total(self) -> bool:
        return self.trained_total

    def is_trained_cv(self) -> bool:
        return self.trained_cv

    def _create_preprocessors(self) -> None:
        self.data_preprocessor: StandardScaler | object = StandardScaler().fit(self.data)
        # self.data_preprocessor: Normalizer | object = Normalizer().fit(self.data)

    def _split_cv_folds(self) -> None:
        if self.n_folds < 1:
            raise ValueError('Error !!! Invalid number of folds. Correct values are greater than or equal to 1 ... ')
        if self.n_folds == 1:
            self.train_indices = [range(self.data.shape[0])]
            self.test_indices = [range(self.data.shape[0])]
        else:
            self.train_indices = []
            self.test_indices = []
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(self.data, self.target):
                self.train_indices.append(train_index)
                self.test_indices.append(test_index)

    def _create_estimator(self, params: Dict) -> RF:
        estimator = None
        if self.estimator_id == 'RF':
            estimator = RF(random_state=42, n_jobs=8, bootstrap=True)
        else:
            raise ValueError('Error !!! Invalid name. Correct values = {RF} ... ')
        estimator.set_params(**params)
        return estimator

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        return self.data_preprocessor.transform(x)

    def _learn_without_cv(self, params: Dict, preprocess: bool = False) -> None:
        self.trained_total = False
        self.gating_model = None
        total_estimator = self._create_estimator(params)
        if preprocess:
            total_estimator.fit(self._preprocess(self.data), self.target)
        else:
            total_estimator.fit(self.data, self.target)
        self.gating_model = deepcopy(total_estimator)
        self.trained_total = True

    def _learn_with_cv(self, params: Dict, preprocess: bool = False) -> None:
        self.trained_cv = False
        self.gating_model_cv = []
        for f in range(self.n_folds):
            estimator = self._create_estimator(params)
            if preprocess:
                estimator.fit(self._preprocess(self.data[self.train_indices[f]]), self.target[self.train_indices[f]])
            else:
                estimator.fit(self.data[self.train_indices[f]], self.target[self.train_indices[f]])
            self.gating_model_cv.append(deepcopy(estimator))
        self.trained_cv = True

    def learn(self, params: Dict, preprocess: bool = False) -> None:
        self._learn_without_cv(params, preprocess=preprocess)
        self._learn_with_cv(params, preprocess=preprocess)

    def _predict(self, estimator, x_test: np.ndarray, preprocess: bool = False) -> np.ndarray:
        if preprocess:
            y_pred = estimator.predict(self.data_preprocessor.transform(x_test))
        else:
            y_pred = estimator.predict(x_test)
        return y_pred

    def _predict_probabilities(self, estimator, x_test: np.ndarray, preprocess: bool = False) -> np.ndarray:
        if preprocess:
            y_proba = estimator.predict_proba(self.data_preprocessor.transform(x_test))
        else:
            y_proba = estimator.predict_proba(x_test)
        return y_proba

    def predict_with_cv(self, preprocess: bool = False) \
            -> Tuple[float, float, float, float, float, float, float, float, float]:
        if self.trained_cv:
            recall_micro = []
            recall_macro = []
            recall_weighted = []
            precision_micro = []
            precision_macro = []
            precision_weighted = []
            f1_micro = []
            f1_macro = []
            f1_weighted = []
            for f in range(self.n_folds):
                y_test = self.target[self.test_indices[f]]
                y_pred = self._predict(self.gating_model_cv[f], self.data[self.test_indices[f]], preprocess=preprocess)
                recall_micro.append(recall_score(y_test, y_pred, average='micro'))
                recall_macro.append(recall_score(y_test, y_pred, average='macro'))
                recall_weighted.append(recall_score(y_test, y_pred, average='weighted'))
                precision_micro.append(precision_score(y_test, y_pred, average='micro'))
                precision_macro.append(precision_score(y_test, y_pred, average='macro'))
                precision_weighted.append(precision_score(y_test, y_pred, average='weighted'))
                f1_micro.append(f1_score(y_test, y_pred, average='micro'))
                f1_macro.append(f1_score(y_test, y_pred, average='macro'))
                f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))
                # print(classification_report(y_test, y_pred))
            return (
                round(float(np.mean(recall_micro)), 3), round(float(np.mean(recall_macro)), 3), round(float(np.mean(recall_weighted)), 3),
                round(float(np.mean(precision_micro)), 3), round(float(np.mean(precision_macro)), 3), round(float(np.mean(precision_weighted)), 3),
                round(float(np.mean(f1_micro)), 3), round(float(np.mean(f1_macro)), 3), round(float(np.mean(f1_weighted)), 3),
            )
        raise ValueError('Error !!! Model not trained ...')

    def predict_blind_data(
            self,
            b_data: np.ndarray,
            b_target: np.ndarray,
            preprocess: bool = False
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        if self.trained_cv:
            recall_micro = []
            recall_macro = []
            recall_weighted = []
            precision_micro = []
            precision_macro = []
            precision_weighted = []
            f1_micro = []
            f1_macro = []
            f1_weighted = []
            b_clusters = get_cluster_labels(b_target, self.lower_quantile_value, self.upper_quantile_value)
            for f in range(self.n_folds):
                y_pred = self._predict(self.gating_model_cv[f], b_data, preprocess=preprocess)
                recall_micro.append(recall_score(b_clusters, y_pred, average='micro'))
                recall_macro.append(recall_score(b_clusters, y_pred, average='macro'))
                recall_weighted.append(recall_score(b_clusters, y_pred, average='weighted'))
                precision_micro.append(precision_score(b_clusters, y_pred, average='micro'))
                precision_macro.append(precision_score(b_clusters, y_pred, average='macro'))
                precision_weighted.append(precision_score(b_clusters, y_pred, average='weighted'))
                f1_micro.append(f1_score(b_clusters, y_pred, average='micro'))
                f1_macro.append(f1_score(b_clusters, y_pred, average='macro'))
                f1_weighted.append(f1_score(b_clusters, y_pred, average='weighted'))
            return (
                round(float(np.mean(recall_micro)), 3), round(float(np.mean(recall_macro)), 3), round(float(np.mean(recall_weighted)), 3),
                round(float(np.mean(precision_micro)), 3), round(float(np.mean(precision_macro)), 3), round(float(np.mean(precision_weighted)), 3),
                round(float(np.mean(f1_micro)), 3), round(float(np.mean(f1_macro)), 3), round(float(np.mean(f1_weighted)), 3),
            )
        raise ValueError('Error !!! Model not trained ...')

    def predict_blind_without_cv(
            self,
            b_data: np.ndarray,
            b_target: np.ndarray,
            preprocess: bool = False
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        if self.trained_total:
            b_clusters = get_cluster_labels(b_target, self.lower_quantile_value, self.upper_quantile_value)
            y_pred = self._predict(self.gating_model, b_data, preprocess=preprocess)
            recall_micro = recall_score(b_clusters, y_pred, average='micro')
            recall_macro = recall_score(b_clusters, y_pred, average='macro')
            recall_weighted = recall_score(b_clusters, y_pred, average='weighted')
            precision_micro = precision_score(b_clusters, y_pred, average='micro')
            precision_macro = precision_score(b_clusters, y_pred, average='macro')
            precision_weighted = precision_score(b_clusters, y_pred, average='weighted')
            f1_micro = f1_score(b_clusters, y_pred, average='micro')
            f1_macro = f1_score(b_clusters, y_pred, average='macro')
            f1_weighted = f1_score(b_clusters, y_pred, average='weighted')
            return (
                round(float(recall_micro), 3), round(float(recall_macro), 3), round(float(recall_weighted), 3),
                round(float(precision_micro), 3), round(float(precision_macro), 3), round(float(precision_weighted), 3),
                round(float(f1_micro), 3), round(float(f1_macro), 3), round(float(f1_weighted), 3),
            )
        raise ValueError('Error !!! Model not trained ...')

    def predict_probability_blind_without_cv(
            self,
            b_data: np.ndarray,
            preprocess: bool = False
    ) -> np.ndarray:
        if self.trained_total:
            return self._predict_probabilities(self.gating_model, b_data, preprocess=preprocess)
        raise ValueError('Error !!! Model not trained ...')

