from copy import deepcopy
from typing import Tuple,List
from ..io.resource import DataResource
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, \
    classification_report, top_k_accuracy_score
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

def compute_binary_metrics_with_ci(y_true, y_pred, n_bootstraps=500, ci_level=0.95):
    rng = np.random.default_rng(42)
    n_samples = len(y_true)

    bootstrapped_acc = []
    bootstrapped_sens = []
    bootstrapped_spec = []
    bootstrapped_f1s = []
    bootstrapped_ba = []

    for _ in range(n_bootstraps):
        # Resample indices with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        resampled_true = y_true[indices]
        resampled_pred = y_pred[indices]

        # Calculate metrics for the bootstrap sample
        acc = accuracy_score(resampled_true, resampled_pred)
        sens = recall_score(resampled_true, resampled_pred)
        cm = confusion_matrix(resampled_true, resampled_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = (tn/1.0) / (tn+fp)
        f1s = f1_score(resampled_true, resampled_pred)
        ba = balanced_accuracy_score(resampled_true, resampled_pred)

        bootstrapped_acc.append(acc)
        bootstrapped_sens.append(sens)
        bootstrapped_spec.append(spec)
        bootstrapped_f1s.append(f1s)
        bootstrapped_ba.append(ba)

    pt_acc = accuracy_score(y_true, y_pred)
    pt_sens = recall_score(y_true, y_pred)
    pt_cm = confusion_matrix(y_true, y_pred)
    pt_tn, pt_fp, pt_fn, pt_tp = pt_cm.ravel()
    pt_spec = (pt_tn / 1.0) / (pt_tn + pt_fp)
    pt_f1s = f1_score(y_true, y_pred)
    pt_ba = balanced_accuracy_score(y_true, y_pred)

    pt_metrics = {
        'accuracy': pt_acc,
        'sensitivity': pt_sens,
        'specificity': pt_spec,
        'f1-score': pt_f1s,
        'balanced accuracy': pt_ba
    }

    # Calculate percentiles for the boundaries
    lower_pct = (1.0 - ci_level) / 2.0 * 100
    upper_pct = (1.0 + ci_level) / 2.0 * 100

    ci_metrics = {
        'accuracy': np.percentile(bootstrapped_acc, [lower_pct, upper_pct]),
        'sensitivity': np.percentile(bootstrapped_sens, [lower_pct, upper_pct]),
        'specificity': np.percentile(bootstrapped_spec, [lower_pct, upper_pct]),
        'f1-score': np.percentile(bootstrapped_f1s, [lower_pct, upper_pct]),
        'balanced accuracy': np.percentile(bootstrapped_ba, [lower_pct, upper_pct])
    }

    return pt_metrics, ci_metrics

def compute_multiclass_metrics_with_ci(y_true, y_pred, y_proba, labels=None, n_bootstraps=500, ci_level=0.95):
    rng = np.random.default_rng(42)
    n_samples = len(y_true)

    ci_metrics = {
        'top_1': [],
        'top_2': [],
        'top_3': [],
        'macro_precision': [],
        'macro_recall': [],
        'macro_f1': [],
        'weighted_precision': [],
        'weighted_recall': [],
        'weighted_f1': []
    }

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        resampled_true = y_true[indices]
        resampled_pred = y_pred[indices]
        resampled_proba = y_proba[indices]

        # Guard against a boot sample missing an entire class, which breaks metric matrices
        if len(np.unique(resampled_true)) < len(labels):
            continue

        top_1_acc = top_k_accuracy_score(resampled_true, resampled_proba, k=1, labels=labels)
        top_2_acc = top_k_accuracy_score(resampled_true, resampled_proba, k=2, labels=labels)
        top_3_acc = top_k_accuracy_score(resampled_true, resampled_proba, k=3, labels=labels)
        report = classification_report(resampled_true, resampled_pred, labels=labels, output_dict=True, zero_division=0)
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1 = report['weighted avg']['f1-score']

        ci_metrics['top_1'].append(top_1_acc)
        ci_metrics['top_2'].append(top_2_acc)
        ci_metrics['top_3'].append(top_3_acc)
        ci_metrics['macro_precision'].append(macro_precision)
        ci_metrics['macro_recall'].append(macro_recall)
        ci_metrics['macro_f1'].append(macro_f1)
        ci_metrics['weighted_precision'].append(weighted_precision)
        ci_metrics['weighted_recall'].append(weighted_recall)
        ci_metrics['weighted_f1'].append(weighted_f1)

    pt_report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pt_metrics = {
        'top_1': top_k_accuracy_score(y_true, y_proba, k=1, labels=labels),
        'top_2': top_k_accuracy_score(y_true, y_proba, k=2, labels=labels),
        'top_3': top_k_accuracy_score(y_true, y_proba, k=3, labels=labels),
        'macro_precision': pt_report['macro avg']['precision'],
        'macro_recall': pt_report['macro avg']['recall'],
        'macro_f1': pt_report['macro avg']['f1-score'],
        'weighted_precision': pt_report['weighted avg']['precision'],
        'weighted_recall': pt_report['weighted avg']['recall'],
        'weighted_f1': pt_report['weighted avg']['f1-score']
    }

    # Extract boundaries using percentiles
    lower_pct = (1.0 - ci_level) / 2.0 * 100
    upper_pct = (1.0 + ci_level) / 2.0 * 100

    intervals = {}
    for metric_name, values in ci_metrics.items():
        intervals[metric_name] = np.percentile(values, [lower_pct, upper_pct])

    return pt_metrics, intervals
