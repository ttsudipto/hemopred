from typing import Tuple, Dict
from sklearn.model_selection import ParameterGrid
from .gating_model import GatingModel
import numpy as np


rf_param_grid = {
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', 'balanced_subsample'],
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                     110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                     210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
                     310, 320, 330, 340, 350, 360, 370, 380, 390, 400],
    'max_depth': [None],
    'max_features': [0.25, 0.5, 0.75, None]
}


def print_gs_gating_output_header() -> None:
        print('\t'.join(
            (
                'params', 'Recall_micro', 'Recall_macro', 'Recall_weighted',
                'Precision_micro', 'Precision_macro', 'Precision_weighted',
                'F1_score_micro', 'F1_score_macro', 'F1_score_weighted',
            )
        ))


def print_gs_gating_output_row(param: Dict, result_cv: Tuple) -> None:
    (
        recall_micro, recall_macro, recall_weighted,
        precision_micro, precision_macro, precision_weighted,
        f1_micro, f1_macro, f1_weighted,
    ) = result_cv
    print('\t'.join(
        (
            str(param),
            str(recall_micro), str(recall_macro), str(recall_weighted),
            str(precision_micro), str(precision_macro), str(precision_weighted),
            str(f1_micro), str(f1_macro), str(f1_weighted),
        )
    ))


def gs_gating_rf_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        lower_quantile_value: float,
        upper_quantile_value: float,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    print_gs_gating_output_header()
    for param in ParameterGrid(rf_param_grid):
        model = GatingModel(
            e_id='RF',
            data=data,
            target=target,
            lower_quantile_value=lower_quantile_value,
            upper_quantile_value=upper_quantile_value,
            k=k,
            do_shuffle=True
        )
        model.learn(param, preprocess)
        result_cv = model.predict_with_cv(preprocess=preprocess)
        print_gs_gating_output_row(param, result_cv)
    return
