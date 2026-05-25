from copy import deepcopy
from typing import Tuple,List
from ..io.resource import DataResource
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
import numpy as np


def split_cv_blind_data(
        res: DataResource,
        test_size: float = 0.2,
        return_resource: bool = False,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[DataResource, DataResource]:
    cv_data, blind_data, cv_target, blind_target = train_test_split(
        res.data,
        res.target,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=res.target_labels
    )
    if verbose:
        print(
            'CV data: ' + str(cv_data.shape) + ' | '
            'CV target: ' + str(cv_target.shape) + ' | '
            'Blind data: ' + str(blind_data.shape) + ' | '
            'Blind target: ' + str(blind_target.shape) + '\n'
        )
    if return_resource:
        _, _, cv_target_labels, blind_target_labels = train_test_split(
            res.data,
            res.target_labels,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=res.target_labels
        )
        _, _, cv_target_labels_original, blind_target_labels_original = train_test_split(
            res.data,
            res.target_labels_original,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=res.target_labels
        )
        _, _, cv_patient_ids, blind_patient_ids = train_test_split(
            res.data,
            res.patient_ids,
            test_size=test_size,
            random_state=42,
            shuffle=True,
            stratify=res.target_labels
        )
        cv_res = deepcopy(res)
        cv_res.data, cv_res.target, cv_res.target_labels, cv_res.target_labels_original, cv_res.patient_ids = \
            cv_data, cv_target, cv_target_labels, cv_target_labels_original, cv_patient_ids
        blind_res = deepcopy(res)
        blind_res.data, blind_res.target, blind_res.target_labels, blind_res.target_labels_original, blind_res.patient_ids = \
            blind_data, blind_target, blind_target_labels, blind_target_labels_original, blind_patient_ids
        return cv_res, blind_res

    return cv_data, blind_data, cv_target, blind_target

def univariate(
        data: np.ndarray,
        target: np.ndarray,
        features: List[str],
        preprocess:bool = True,
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if preprocess:
        data = StandardScaler().fit_transform(data)
        target = StandardScaler().fit_transform(target.reshape(-1, 1)).ravel()
    f_val, p_val = f_regression(data, target)
    if verbose:
        print('\t'.join(('Feature', 'F-value', 'P-value')))
        for i in range(len(features)):
            print('\t'.join((features[i], str(f_val[i]), str(p_val[i]))))
    return f_val, p_val

def mean_fishers(values: float | np.ndarray, axis: int | None = None) -> float | np.ndarray:
    if isinstance(values, float):
        return values
    z = np.arctanh(values)
    if axis is None:
        z_bar = np.mean(z.ravel())
    else:
        z_bar = np.mean(z, axis=axis)
    return np.tanh(z_bar)

def mean_variance_weighted(values: float | np.ndarray, true_values: np.ndarray, axis: int | None = None) -> float:
    if isinstance(values, float):
        return values
    variances = np.var(true_values, axis=axis)
    weights = variances / np.sum(variances)
    return np.sum(weights * values)

def get_quantiles(target, q_lower, q_upper) -> Tuple[float, float]:
    lower_quantile, upper_quantile = np.percentile(target, [q_lower, q_upper])
    return lower_quantile, upper_quantile

def get_cluster_labels(target, lower_quantile_value, upper_quantile_value, verbose: bool = False) -> np.ndarray:
    labels = np.select(
        [target <= lower_quantile_value, target >= upper_quantile_value],
        [1, 3],
        default=2
    )
    if verbose:
        for i in range(target.shape[0]):
            print(target[i], labels[i])
    return labels

