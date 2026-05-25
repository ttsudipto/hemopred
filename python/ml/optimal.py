from typing import Dict, List
from .gating_model import GatingModel
from .grid_search_gating import print_gs_gating_output_row
import numpy as np

optimal_params_gating = {
    'HbA2': {
        'RF': {
            'class_weight': 'balanced_subsample',
            'n_estimators': 190,
            'max_depth': None,
            'criterion': 'gini',
            'max_features': 0.25
        }
    }, 'HbF': {
        'RF': {
            'class_weight': 'balanced',
            'n_estimators': 300,
            'max_depth': None,
            'criterion': 'entropy',
            'max_features': 0.25
        }
    }
}


def get_optimal_gating_params(
        target_id: str,
        e_id: str,
        verbose: bool = False
) -> Dict:
    param = optimal_params_gating[target_id][e_id]
    if verbose:
        print(param)
    return param

def create_optimal_gating_model(
        target_id: str,
        e_id: str,
        data: np.ndarray,
        target: np.ndarray,
        lower_quantile_value: float,
        upper_quantile_value: float,
        k: int = 5,
        feature_names: List[str] = None,
        preprocess: bool = False,
        verbose: bool = False
) -> GatingModel:
    param = optimal_params_gating[target_id][e_id]
    model = GatingModel(
        e_id=e_id,
        data=data,
        target=target,
        lower_quantile_value=lower_quantile_value,
        upper_quantile_value=upper_quantile_value,
        k=k,
        feature_names=feature_names,
        do_shuffle=True
    )
    model.learn(param, preprocess)
    if verbose:
        result_cv = model.predict_with_cv(preprocess=preprocess)
        print_gs_gating_output_row(param, result_cv)
    return model

