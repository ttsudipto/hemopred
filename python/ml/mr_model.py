from typing import List, Dict, Tuple
from .ml_utils import mean_fishers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import ElasticNet as EN
from sklearn.multioutput import RegressorChain
import numpy as np
from copy import deepcopy


class MRModel:

    estimators_to_be_chained = ['SVR']

    def __init__(
            self,
            e_id: str,
            data: np.ndarray,
            target: np.ndarray,
            k: int = 5,
            feature_names: List[str] = None,
            do_shuffle: bool = True
    ) -> None:
        self.estimator_id = e_id
        self.n_folds = k
        self.feature_names = feature_names
        if do_shuffle:
            self.data = shuffle(data, random_state=42)
            self.target = shuffle(target, random_state=42)
        else:
            self.data = data
            self.target = target
        self.trained_total = False
        self.trained_cv = False
        self.estimators = []
        self.total_estimator = None
        self.train_indices = []
        self.test_indices = []
        self._create_preprocessors()
        self._split_cv_folds()

    def is_trained_total(self) -> bool:
        return self.trained_total

    def is_trained_cv(self) -> bool:
        return self.trained_cv

    def _is_chaining_required(self) -> bool:
        return self.estimator_id in MRModel.estimators_to_be_chained

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
            skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(self.data, self.target):
                self.train_indices.append(train_index)
                self.test_indices.append(test_index)

    def _create_estimator(self, params: Dict) -> RegressorChain | SVR | RFR | MLPR | LR | EN:
        estimator = None
        if self.estimator_id == 'SVR':
            estimator = SVR(max_iter=10000)
        elif self.estimator_id == 'RFR':
            estimator = RFR(n_jobs=8, bootstrap=True, random_state=42)
        elif self.estimator_id == 'ETR':
            estimator = ETR(n_jobs=8, bootstrap=True, random_state=42)
        elif self.estimator_id == 'MLPR':
            estimator = MLPR(learning_rate='constant', solver='adam', max_iter=10000, early_stopping=True,
                            random_state=42)
        elif self.estimator_id == 'LR':
            estimator = LR(n_jobs=8)
        elif self.estimator_id == 'EN':
            estimator = EN(max_iter=10000, selection='random', random_state=42)
        else:
            raise ValueError('Error !!! Invalid name. Correct values = {SVR, RFR, MLP, LR, EN} ... ')
        estimator.set_params(**params)
        if self.target.ndim > 1 and self._is_chaining_required():
            estimator = RegressorChain(estimator, random_state=42)
        return estimator

    def _learn_without_cv(self, params: Dict, preprocess: bool = False) -> None:
        estimator = self._create_estimator(params)
        if preprocess:
            if self.target.ndim > 1:
                estimator.fit(
                    self.data_preprocessor.transform(self.data),
                    self.target_preprocessor.transform(self.target)
                )
            else:
                estimator.fit(
                    self.data_preprocessor.transform(self.data),
                    self.target_preprocessor.transform(self.target.reshape(-1, 1)).ravel()
                )
        else:
            estimator.fit(self.data, self.target)
        self.total_estimator = deepcopy(estimator)
        self.trained_total = True

    def _learn_with_cv(self, params: Dict, preprocess: bool = False) -> None:
        self.estimators = []
        for f in range(self.n_folds):
            estimator = self._create_estimator(params)
            if preprocess:
                if self.target.ndim > 1:
                    estimator.fit(
                        self.data_preprocessor.transform(self.data[self.train_indices[f]]),
                        self.target_preprocessor.transform(self.target[self.train_indices[f]])
                    )
                else:
                    estimator.fit(
                        self.data_preprocessor.transform(self.data[self.train_indices[f]]),
                        self.target_preprocessor.transform(self.target[self.train_indices[f]].reshape(-1, 1)).ravel()
                    )
            else:
                estimator.fit(self.data[self.train_indices[f]], self.target[self.train_indices[f]])
            self.estimators.append(deepcopy(estimator))
        self.trained_cv = True

    def learn(self, params: Dict, preprocess: bool = False) -> None:
        self._learn_without_cv(params, preprocess=preprocess)
        self._learn_with_cv(params, preprocess=preprocess)

    def _predict(self, estimator: SVR | RFR | MLPR | LR | EN, x_test: np.ndarray, preprocess: bool = False) -> np.ndarray:
        if preprocess:
            x_test_preprocessed = self.data_preprocessor.transform(x_test)
            y_pred_processed = estimator.predict(x_test_preprocessed)
            if self.target.ndim > 1:
                y_pred = self.target_preprocessor.inverse_transform(y_pred_processed)
            else:
                y_pred = self.target_preprocessor.inverse_transform(y_pred_processed.reshape(-1, 1)).ravel()
        else:
            y_pred = estimator.predict(x_test)
        return y_pred

    def predict_with_cv(self, preprocess: bool = False) \
            -> (Tuple[float, float, float, float, float] |
                Tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray]):
        if self.trained_cv:
            r2score = []
            mse = []
            mae = []
            pcc = []
            rmse = []
            r2score_raw = []
            mse_raw = []
            mae_raw = []
            pcc_raw = []
            rmse_raw = []
            for f in range(self.n_folds):
                y_test = self.target[self.test_indices[f]]
                y_pred = self._predict(self.estimators[f], self.data[self.test_indices[f]], preprocess=preprocess)
                if self.target.ndim > 1:
                    target_variances = np.var(y_test, axis=0)
                    r2score.append(r2_score(y_test, y_pred, multioutput='variance_weighted'))
                    mse.append(mean_squared_error(y_test, y_pred, multioutput=target_variances))
                    mae.append(mean_absolute_error(y_test, y_pred, multioutput='uniform_average'))
                    # pcc.append(np.mean(pearsonr(y_test, y_pred)[0]))
                    pcc.append(mean_fishers(pearsonr(y_test, y_pred)[0]))
                    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred, multioutput=target_variances)))
                    r2score_raw.append(r2_score(y_test, y_pred, multioutput='raw_values'))
                    mse_raw.append(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
                    mae_raw.append(mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
                    pcc_raw.append(pearsonr(y_test, y_pred)[0])
                    rmse_raw.append(np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values')))
                else:
                    r2score.append(r2_score(y_test, y_pred))
                    mse.append(mean_squared_error(y_test, y_pred))
                    mae.append(mean_absolute_error(y_test, y_pred))
                    pcc.append(pearsonr(y_test, y_pred)[0])
                    rmse.append(root_mean_squared_error(y_test, y_pred))
            if self.target.ndim > 1:
                return (
                    round(float(np.mean(mae)), 3), np.round(np.mean(mae_raw, axis=0), 3),
                    round(float(np.mean(mse)), 3), np.round(np.mean(mse_raw, axis=0), 3),
                    round(float(np.mean(rmse)), 3), np.round(np.mean(rmse_raw, axis=0), 3),
                    # round(float(np.mean(pcc)), 3), np.round(np.mean(pcc_raw, axis=0), 3),
                    round(float(mean_fishers(np.array(pcc))), 3),
                    np.round(mean_fishers(np.array(pcc_raw), axis=0), 3),
                    round(float(np.mean(r2score)), 3), np.round(np.mean(r2score_raw, axis=0), 3)
                )
            else:
                return (
                    round(float(np.mean(mae)), 3), round(float(np.mean(mse)), 3),
                    round(float(np.mean(rmse)), 3), round(float(np.mean(pcc)), 3),
                    round(float(np.mean(r2score)), 3)
                )
        raise ValueError('Error !!! Model not trained ...')

    def predict_blind_data(self, b_data: np.ndarray, b_target: np.ndarray, preprocess: bool = False) \
            -> (Tuple[float, float, float, float, float] |
                Tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray]):
        if self.trained_cv:
            r2score = []
            mse = []
            mae = []
            pcc = []
            rmse = []
            r2score_raw = []
            mse_raw = []
            mae_raw = []
            pcc_raw = []
            rmse_raw = []
            for f in range(self.n_folds):
                y_pred = self._predict(self.estimators[f], b_data, preprocess=preprocess)
                if self.target.ndim > 1:
                    target_variances = np.var(b_target, axis=0)
                    r2score.append(r2_score(b_target, y_pred, multioutput='variance_weighted'))
                    mse.append(mean_squared_error(b_target, y_pred, multioutput=target_variances))
                    mae.append(mean_absolute_error(b_target, y_pred, multioutput='uniform_average'))
                    # pcc.append(np.mean(pearsonr(b_target, y_pred)[0]))
                    pcc.append(mean_fishers(pearsonr(b_target, y_pred)[0]))
                    rmse.append(np.sqrt(mean_squared_error(b_target, y_pred, multioutput=target_variances)))
                    r2score_raw.append(r2_score(b_target, y_pred, multioutput='raw_values'))
                    mse_raw.append(mean_squared_error(b_target, y_pred, multioutput='raw_values'))
                    mae_raw.append(mean_absolute_error(b_target, y_pred, multioutput='raw_values'))
                    pcc_raw.append(pearsonr(b_target, y_pred)[0])
                    rmse_raw.append(np.sqrt(mean_squared_error(b_target, y_pred, multioutput='raw_values')))
                else:
                    r2score.append(r2_score(b_target, y_pred))
                    mse.append(mean_squared_error(b_target, y_pred))
                    mae.append(mean_absolute_error(b_target, y_pred))
                    pcc.append(mean_fishers(pearsonr(b_target, y_pred)[0]))
                    rmse.append(root_mean_squared_error(b_target, y_pred))
            if self.target.ndim > 1:
                return (
                    round(float(np.mean(mae)), 3), np.round(np.mean(mae_raw, axis=0), 3),
                    round(float(np.mean(mse)), 3), np.round(np.mean(mse_raw, axis=0), 3),
                    round(float(np.mean(rmse)), 3), np.round(np.mean(rmse_raw, axis=0), 3),
                    # round(float(np.mean(pcc)), 3), np.round(np.mean(pcc_raw, axis=0), 3),
                    round(float(mean_fishers(np.array(pcc))), 3), np.round(mean_fishers(np.array(pcc_raw), axis=0), 3),
                    round(float(np.mean(r2score)), 3), np.round(np.mean(r2score_raw, axis=0), 3)
                )
            else:
                return (
                    round(float(np.mean(mae)), 3), round(float(np.mean(mse)), 3),
                    round(float(np.mean(rmse)), 3), round(float(np.mean(pcc)), 3),
                    round(float(np.mean(r2score)), 3)
                )
        raise ValueError('Error !!! Model not trained ...')

    def predict_blind_without_cv(self, b_data: np.ndarray, b_target: np.ndarray, preprocess: bool = False) \
            -> (Tuple[float, float, float, float, float] |
                Tuple[float, np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray, float, np.ndarray]):
        if self.trained_total:
            y_pred = self._predict(self.total_estimator, b_data, preprocess=preprocess)
            if self.target.ndim > 1:
                target_variances = np.var(b_target, axis=0)
                r2score = r2_score(b_target, y_pred, multioutput='variance_weighted')
                mse = mean_squared_error(b_target, y_pred, multioutput=target_variances)
                mae = mean_absolute_error(b_target, y_pred, multioutput='uniform_average')
                # pcc = np.mean(pearsonr(b_target, y_pred)[0])
                pcc = mean_fishers(pearsonr(b_target, y_pred)[0])
                rmse = np.sqrt(mean_squared_error(b_target, y_pred, multioutput=target_variances))
                r2score_raw = r2_score(b_target, y_pred, multioutput='raw_values')
                mse_raw = mean_squared_error(b_target, y_pred, multioutput='raw_values')
                mae_raw = mean_absolute_error(b_target, y_pred, multioutput='raw_values')
                pcc_raw = pearsonr(b_target, y_pred)[0]
                rmse_raw = np.sqrt(mean_squared_error(b_target, y_pred, multioutput='raw_values'))
                return (
                    round(mae, 3), np.round(mae_raw, 3),
                    round(mse, 3), np.round(mse_raw, 3),
                    round(rmse, 3), np.round(rmse_raw, 3),
                    round(pcc, 3), np.round(pcc_raw, 3),
                    round(r2score, 3), np.round(r2score_raw, 3)
                )
            else:
                r2score = r2_score(b_target, y_pred)
                mse = mean_squared_error(b_target, y_pred)
                mae = mean_absolute_error(b_target, y_pred)
                pcc = mean_fishers(pearsonr(b_target, y_pred)[0])
                rmse = root_mean_squared_error(b_target, y_pred)
                return (
                    round(mae, 3), round(mse, 3), round(rmse, 3),
                    round(pcc, 3), round(r2score, 3)
                )
        raise ValueError('Error !!! Model not trained ...')

    def predict_labels_blind_without_cv(self, b_data: np.ndarray, preprocess: bool = False) -> np.ndarray:  # mainly used by the explainer
        if self.trained_total:
            return self._predict(self.total_estimator, b_data, preprocess=preprocess)
        raise ValueError('Error !!! Model not trained ...')

