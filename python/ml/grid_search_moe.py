from typing import Tuple, Dict, List
from .gating_model import GatingModel
from .moe_model import MOEModel
from sklearn.model_selection import ParameterGrid
import numpy as np


svr_rbf_param_grid = {
    'kernel': ['rbf'],
    'C': [1, 5, 10, 15],
    'gamma': [0.1, 0.01, 0.001, 0.0001],
    'epsilon': [0.01, 0.1, 0.25, 0.5, 0.75],
}

svr_poly_param_grid = {
    'kernel': ['poly'],
    'C': [1, 5, 10],
    'degree': [2, 3],
    'gamma': [0.1, 0.01, 0.001, 0.0001],
    'coef0': [0, 1, 2, 3],
    'epsilon': [0.01, 0.1, 0.25, 0.5, 0.75],
}

svr_linear_param_grid = {
    'kernel': ['linear'],
    'C': [1, 5, 10, 15],
    'epsilon': [0.01, 0.1, 0.25, 0.5, 0.75],
}

rfr_param_grid = {
    # 'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None],
    'max_features': [0.25, 0.5, 0.75, None]
}

etr_param_grid = {
    # 'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None],
    'max_features': [0.25, 0.5, 0.75, None]
}

mlpr_param_grid = {
    'activation': ['relu', 'logistic'],
    'hidden_layer_sizes':
        []
        + [(50,), (100,), (150,), (200,), (250,), (300,)]
        # + [(50, 10), (100, 20), (150, 30), (200, 40), (250, 50), (300, 60)]
        + [(50, 25), (100, 50), (150, 75), (200, 100), (250, 125), (300, 150)]
        + [(50, 50), (100, 100), (150, 150), (200, 200), (250, 250), (300, 300)]
        + [],
    'learning_rate_init': [0.01, 0.001, 0.0001]
}

en_param_grid = {
    'alpha': list(np.linspace(0.01, 1, 99)),
    'l1_ratio': list(np.linspace(0.01, 1, 99)),
}

gbr_param_grid = {
    'n_estimators': [50, 100, 150, 200],
}

hgbr_param_grid = {
    'learning_rate': [0.2, 0.4, 0.6, 0.8, 1.0],
    'l2_regularization': [0, 0.25, 0.5, 0.75],
}

qr_param_grid = {
    'alpha': [0, 0.1, 1, 10],
}

eid_map = {
    'SVR_rbf': 'SVR',
    'SVR_poly': 'SVR',
    'SVR_linear': 'SVR',
    'RFR': 'RFR',
    'ETR': 'ETR',
    'MLPR': 'MLPR',
    'EN': 'EN',
    'GBR': 'GBR',
    'HGBR': 'HGBR',
    'QR': 'QR'
}

param_grid_map = {
    'SVR_rbf': svr_rbf_param_grid,
    'SVR_poly': svr_poly_param_grid,
    'SVR_linear': svr_linear_param_grid,
    'RFR': rfr_param_grid,
    'ETR': etr_param_grid,
    'MLPR': mlpr_param_grid,
    'EN': en_param_grid,
    'GBR': gbr_param_grid,
    'HGBR': hgbr_param_grid,
    'QR': qr_param_grid,
}


def print_gs_moe_output_header() -> None:
    print('\t'.join(
        (
            'params', 'MAE', 'MSE', 'RMSE', 'PCC', 'R2_score',
            'MAE_blind', 'MSE_blind', 'RMSE_blind', 'PCC_blind', 'R2_score_blind',
            'MAE_blind_cv', 'MSE_blind_cv', 'RMSE_blind_cv', 'PCC_blind_cv', 'R2_score_blind_cv',
        )
    ))


def print_gs_moe_output_row(param: List[Dict], result_cv: Tuple, result_blind: Tuple, result_blind_cv: Tuple) -> None:
    mae, mse, rmse, pcc, r2score = result_cv
    mae_blind, mse_blind, rmse_blind, pcc_blind, r2score_blind = result_blind
    mae_blind_cv, mse_blind_cv, rmse_blind_cv, pcc_blind_cv, r2score_blind_cv = result_blind_cv
    print('\t'.join(
        (
            str(param),
            str(round(mae, 4)), str(round(mse, 4)), str(round(rmse, 4)), str(round(pcc, 4)), str(round(r2score, 4)),
            str(round(mae_blind, 4)), str(round(mse_blind, 4)), str(round(rmse_blind, 4)), str(round(pcc_blind, 4)), str(round(r2score_blind, 4)),
            str(round(mae_blind_cv, 4)), str(round(mse_blind_cv, 4)), str(round(rmse_blind_cv, 4)), str(round(pcc_blind_cv, 4)), str(round(r2score_blind_cv, 4)),
        )
    ))


def gs_with_cv(
        e_ids: Tuple[str, str],
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        clusters: np.ndarray,
        gating_model: GatingModel,
        quantiles: Tuple[float, float],
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')

    q_eid = eid_map[e_ids[0]]
    q_param_grid = ParameterGrid(param_grid_map[e_ids[0]])
    mid_eid = eid_map[e_ids[1]]
    mid_param_grid = ParameterGrid(param_grid_map[e_ids[1]])

    print_gs_moe_output_header()
    for q_param in q_param_grid:
        for mid_param in mid_param_grid:
            model = MOEModel(
                e_id=(q_eid, mid_eid, q_eid),
                data=data,
                target=target,
                clusters=clusters,
                gating_model=gating_model,
                quantiles=quantiles,
                k=k,
                do_shuffle=True
            )
            model.learn([q_param, mid_param, q_param], preprocess=preprocess)
            result_cv = model.predict_with_cv(preprocess=preprocess)
            result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
            result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
            print_gs_moe_output_row([q_param, mid_param], result_cv, result_blind, result_blind_cv)

