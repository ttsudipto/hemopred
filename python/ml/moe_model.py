from typing import List, Dict, Tuple
from .gating_model import GatingModel
from .ml_utils import mean_fishers, get_quantiles, get_cluster_labels
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.linear_model import QuantileRegressor as QR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import ElasticNet as EN
import numpy as np
from copy import deepcopy


class MOEModel:

    def __init__(
            self,
            e_id: Tuple[str, str, str],
            data: np.ndarray,
            target: np.ndarray,
            clusters: np.ndarray,
            gating_model: GatingModel,
            quantiles: Tuple[float, float],
            k: int = 5,
            feature_names: List[str] = None,
            do_shuffle: bool = True
    ) -> None:
        self.lower_eid, self.mid_eid, self.upper_eid = e_id
        self.imputer_id = 'KNN'
        self.n_folds = k
        if do_shuffle:
            self.data = shuffle(data, random_state=42)
            self.target = shuffle(target, random_state=42)
            # self.clusters = shuffle(clusters, random_state=42)
        else:
            self.data = data
            self.target = target
            # self.clusters = clusters
        self.feature_names = feature_names
        self.gating_model = gating_model
        self.lq, self.uq = quantiles
        self.trained_total = False
        self.trained_cv = False
        self.imputers = []
        self.cv_lq_experts = []
        self.cv_uq_experts = []
        self.cv_mid_experts = []
        self.total_imputer = None
        self.total_lq_expert = None
        self.total_uq_expert = None
        self.total_mid_expert = None
        self.train_indices = []
        self.test_indices = []
        self._create_preprocessors()
        self._split_cv_folds()

    def is_trained_total(self) -> bool:
        return self.trained_total

    def is_trained_cv(self) -> bool:
        return self.trained_cv

    def _create_preprocessors(self) -> None:
        if self.target.ndim > 1:
            self.data_preprocessor: StandardScaler | object = StandardScaler().fit(self.data)
            self.target_preprocessor: StandardScaler | object = StandardScaler().fit(self.target)
        else:
            self.data_preprocessor: StandardScaler | object = StandardScaler().fit(self.data)
            self.target_preprocessor: StandardScaler | object = StandardScaler().fit(self.target.reshape(-1, 1))

    def _split_cv_folds(self) -> None:
        if self.n_folds < 1:
            raise ValueError('Error !!! Invalid number of folds. Correct values are greater than or equal to 1 ... ')
        if self.n_folds == 1:
            self.train_indices = [range(self.data.shape[0])]
            self.test_indices = [range(self.data.shape[0])]
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            lower_quantile_value, upper_quantile_value = get_quantiles(self.target, self.lq * 100, self.uq * 100)
            local_clusters = get_cluster_labels(self.target, lower_quantile_value, upper_quantile_value)
            for train_index, test_index in skf.split(self.data, local_clusters):
                self.train_indices.append(train_index)
                self.test_indices.append(test_index)

    def _create_imputer(self):
        if self.imputer_id == 'KNN':
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='distance')
        else:
            raise ValueError('Error !!! Invalid imputer. Correct values = {KNN} ... ')
        return imputer

    def _create_quantile_estimator(self, q_type: str, params: Dict) -> GBR | HGBR | QR:
        if q_type == 'lower':
            eid = self.lower_eid
            quantile = self.lq
        elif q_type == 'upper':
            eid = self.upper_eid
            quantile = self.uq
        else:
            raise ValueError('Error !!! Invalid type. Correct values = {lower, upper} ... ')
        estimator = None
        if eid == 'GBR':
            estimator = GBR(loss='quantile', alpha=quantile, random_state=42, max_depth=5)
        elif eid == 'HGBR':
            estimator = HGBR(loss='quantile', quantile=quantile, random_state=42, max_iter=1000)
        elif eid == 'QR':
            estimator = QR(quantile=quantile)
        else:
            raise ValueError('Error !!! Invalid name. Correct values = {GBR, HGBR, QR} ... ')
        estimator.set_params(**params)
        return estimator

    def _create_mid_estimator(self, params: Dict) -> SVR | RFR | MLPR | LR | EN:
        estimator = None
        if self.mid_eid == 'SVR':
            estimator = SVR(max_iter=10000)
        elif self.mid_eid == 'RFR':
            estimator = RFR(n_jobs=8, bootstrap=True, random_state=42)
        elif self.mid_eid == 'ETR':
            estimator = ETR(n_jobs=8, bootstrap=True, random_state=42)
        elif self.mid_eid == 'MLPR':
            estimator = MLPR(learning_rate='constant', solver='adam', max_iter=10000, early_stopping=True, random_state=42)
        elif self.mid_eid == 'LR':
            estimator = LR(n_jobs=8)
        elif self.mid_eid == 'EN':
            estimator = EN(max_iter=10000, selection='random', random_state=42)
        else:
            raise ValueError('Error !!! Invalid name. Correct values = {SVR, RFR, MLP, LR, EN} ... ')
        estimator.set_params(**params)
        return estimator

    def _preprocess(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data = self.data_preprocessor.transform(x)
        target = self.target_preprocessor.transform(y.reshape(-1, 1)).ravel()
        return data, target

    def _learn_without_cv(self, params: List[Dict], preprocess: bool = False) -> None:
        self.total_imputer = None
        self.total_lq_expert = None
        self.total_uq_expert = None
        self.total_mid_expert = None
        imputer = self._create_imputer()
        data = imputer.fit_transform(self.data)
        lower_quantile_value, upper_quantile_value = get_quantiles(self.target, self.lq * 100, self.uq * 100)
        local_clusters = get_cluster_labels(self.target, lower_quantile_value, upper_quantile_value)
        if preprocess:
            data, target = self._preprocess(data, self.target)
        else:
            data, target = data, self.target
        lower_estimator = self._create_quantile_estimator('lower', params[0])
        lower_estimator.fit(data[local_clusters == 1], target[local_clusters == 1])
        mid_estimator = self._create_mid_estimator(params[1])
        mid_estimator.fit(data[local_clusters == 2], target[local_clusters == 2])
        upper_estimator = self._create_quantile_estimator('upper', params[2])
        upper_estimator.fit(data[local_clusters == 3], target[local_clusters == 3])
        self.total_imputer = deepcopy(imputer)
        self.total_lq_expert = deepcopy(lower_estimator)
        self.total_mid_expert = deepcopy(mid_estimator)
        self.total_uq_expert = deepcopy(upper_estimator)
        self.trained_total = True

    def _learn_with_cv(self, params: List[Dict], preprocess: bool = False) -> None:
        self.imputers = []
        self.cv_lq_experts = []
        self.cv_mid_experts = []
        self.cv_uq_experts = []
        for f in range(self.n_folds):
            imputer = self._create_imputer()
            data = imputer.fit_transform(self.data[self.train_indices[f]])
            self.imputers.append(deepcopy(imputer))
            lower_quantile_value, upper_quantile_value = get_quantiles(self.target[self.train_indices[f]], self.lq * 100, self.uq * 100)
            cv_clusters = get_cluster_labels(self.target[self.train_indices[f]], lower_quantile_value, upper_quantile_value)
            if preprocess:
                data, target = self._preprocess(data, self.target[self.train_indices[f]])
            else:
                target = self.target[self.train_indices[f]]
            # cv_clusters = self.clusters[self.train_indices[f]]
            lower_estimator = self._create_quantile_estimator('lower', params[0])
            lower_estimator.fit(data[cv_clusters == 1], target[cv_clusters == 1])
            self.cv_lq_experts.append(deepcopy(lower_estimator))
            mid_estimator = self._create_mid_estimator(params[1])
            mid_estimator.fit(data[cv_clusters == 2], target[cv_clusters == 2])
            self.cv_mid_experts.append(deepcopy(mid_estimator))
            upper_estimator = self._create_quantile_estimator('upper', params[2])
            upper_estimator.fit(data[cv_clusters == 3], target[cv_clusters == 3])
            self.cv_uq_experts.append(deepcopy(upper_estimator))
        self.trained_cv = True

    def learn(self, params: List[Dict], preprocess: bool = False) -> None:
        self._learn_without_cv(params, preprocess=preprocess)
        self._learn_with_cv(params, preprocess=preprocess)

    def _predict(self, experts: List, x_test: np.ndarray, preprocess: bool = False) -> np.ndarray:
        cluster_probs = self.gating_model.predict_probability_blind_without_cv(x_test, preprocess=preprocess)
        if preprocess:
            data = self.data_preprocessor.transform(x_test)
        else:
            data = x_test
        y_pred_preprocessed = np.zeros(shape=(data.shape[0],))
        for i in range(len(experts)):
            y_pred_preprocessed += cluster_probs[:, i] * experts[i].predict(data)
        if preprocess:
            y_pred = self.target_preprocessor.inverse_transform(y_pred_preprocessed.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_preprocessed
        return y_pred

    def predict_with_cv(self, preprocess: bool = False) -> Tuple[float, float, float, float, float]:
        # if self.trained_cv and self.trained_gating:
        if self.trained_cv:
            r2score = []
            mse = []
            mae = []
            pcc = []
            rmse = []
            for f in range(self.n_folds):
                y_test = self.target[self.test_indices[f]]
                x_test = self.imputers[f].transform(self.data[self.test_indices[f]])
                y_pred = self._predict(
                    [self.cv_lq_experts[f], self.cv_mid_experts[f], self.cv_uq_experts[f]],
                    x_test,
                    preprocess=preprocess
                )
                r2score.append(r2_score(y_test, y_pred))
                mse.append(mean_squared_error(y_test, y_pred))
                mae.append(mean_absolute_error(y_test, y_pred))
                pcc.append(pearsonr(y_test, y_pred)[0])
                rmse.append(root_mean_squared_error(y_test, y_pred))
            return (
                round(float(np.mean(mae)), 4), round(float(np.mean(mse)), 4),
                round(float(np.mean(rmse)), 4), round(float(np.mean(pcc)), 4),
                round(float(np.mean(r2score)), 4)
            )
        raise ValueError('Error !!! Model not trained ...')

    def predict_blind_data(self, b_data: np.ndarray, b_target: np.ndarray, preprocess: bool = False) \
            -> Tuple[float, float, float, float, float]:
        if self.trained_cv:
            r2score = []
            mse = []
            mae = []
            pcc = []
            rmse = []
            for f in range(self.n_folds):
                x_test = self.imputers[f].transform(b_data)
                y_pred = self._predict(
                    [self.cv_lq_experts[f], self.cv_mid_experts[f], self.cv_uq_experts[f]],
                    x_test,
                    preprocess=preprocess
                )
                r2score.append(r2_score(b_target, y_pred))
                mse.append(mean_squared_error(b_target, y_pred))
                mae.append(mean_absolute_error(b_target, y_pred))
                pcc.append(mean_fishers(pearsonr(b_target, y_pred)[0]))
                rmse.append(root_mean_squared_error(b_target, y_pred))
            return (
                round(float(np.mean(mae)), 4), round(float(np.mean(mse)), 4),
                round(float(np.mean(rmse)), 4), round(float(np.mean(pcc)), 4),
                round(float(np.mean(r2score)), 4)
            )
        raise ValueError('Error !!! Model not trained ...')

    def predict_blind_without_cv(self, b_data: np.ndarray, b_target: np.ndarray, preprocess: bool = False) \
            -> Tuple[float, float, float, float, float]:
        if self.trained_total:
            x_test = self.total_imputer.transform(b_data)
            y_pred = self._predict(
                [self.total_lq_expert, self.total_mid_expert, self.total_uq_expert],
                x_test,
                preprocess=preprocess
            )
            r2score = r2_score(b_target, y_pred)
            mse = mean_squared_error(b_target, y_pred)
            mae = mean_absolute_error(b_target, y_pred)
            pcc = mean_fishers(pearsonr(b_target, y_pred)[0])
            rmse = root_mean_squared_error(b_target, y_pred)
            return (
                round(float(mae), 4), round(float(mse), 4), round(float(rmse), 4),
                round(float(pcc), 4), round(float(r2score), 4)
            )
        raise ValueError('Error !!! Model not trained ...')

    def predict_labels_blind_without_cv(self, b_data: np.ndarray, preprocess: bool = False) -> np.ndarray:
    # mainly used by the explainer
        x_test = self.total_imputer.transform(b_data)
        if self.trained_total:
            return self._predict(
                [self.total_lq_expert, self.total_mid_expert, self.total_uq_expert],
                x_test,
                preprocess=preprocess
            )
        raise ValueError('Error !!! Model not trained ...')

