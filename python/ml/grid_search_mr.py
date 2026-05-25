from typing import Tuple, Dict
from .mr_model import MRModel
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
    'criterion': ['squared_error'],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None],
    'max_features': [0.25, 0.5, 0.75, None]
}

etr_param_grid = {
    # 'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'criterion': ['squared_error'],
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


def print_gs_mr_output_header(show_raw: bool) -> None:
    if show_raw:
        print('\t'.join(
            (
                'params', 'MAE', 'MAE_raw', 'MSE', 'MSE_raw', 'RMSE', 'RMSE_raw', 'PCC', 'PCC_raw', 'R2_score', 'R2_score_raw',
                'MAE_blind', 'MAE_blind_raw', 'MSE_blind', 'MSE_blind_raw', 'RMSE_blind', 'RMSE_blind_raw', 'PCC_blind', 'PCC_blind_raw', 'R2_score_blind', 'R2_score_blind_raw',
                'MAE_blind_cv', 'MAE_blind_cv_raw', 'MSE_blind_cv', 'MSE_blind_cv_raw', 'RMSE_blind_cv', 'RMSE_blind_cv_raw', 'PCC_blind_cv', 'PCC_blind_cv_raw', 'R2_score_blind_cv', 'R2_score_blind_cv_raw',
            )
        ))
    else:
        print('\t'.join(
            (
                'params', 'MAE', 'MSE', 'RMSE', 'PCC', 'R2_score',
                'MAE_blind', 'MSE_blind', 'RMSE_blind', 'PCC_blind', 'R2_score_blind',
                'MAE_blind_cv', 'MSE_blind_cv', 'RMSE_blind_cv', 'PCC_blind_cv', 'R2_score_blind_cv',
            )
        ))


def print_gs_mr_output_row(param: Dict, result_cv: Tuple, result_blind: Tuple, result_blind_cv: Tuple, show_raw: bool) -> None:
    if show_raw:
        mae, mae_raw, mse, mse_raw, rmse, rmse_raw, pcc, pcc_raw, r2score, r2score_raw = result_cv
        mae_blind, mae_blind_raw, mse_blind, mse_blind_raw, rmse_blind, rmse_blind_raw, pcc_blind, pcc_blind_raw, r2score_blind, r2score_blind_raw = result_blind
        mae_blind_cv, mae_blind_cv_raw, mse_blind_cv, mse_blind_cv_raw, rmse_blind_cv, rmse_blind_cv_raw, pcc_blind_cv, pcc_blind_cv_raw, r2score_blind_cv, r2score_blind_cv_raw = result_blind_cv
        print('\t'.join(
            (
                str(param),
                str(round(mae, 3)), str(np.round(mae_raw, 3)), str(round(mse, 3)), str(np.round(mse_raw, 3)), str(round(rmse, 3)), str(np.round(rmse_raw, 3)), str(round(pcc, 3)), str(np.round(pcc_raw, 3)), str(round(r2score, 3)), str(np.round(r2score_raw, 3)),
                str(round(mae_blind, 3)), str(np.round(mae_blind_raw, 3)), str(round(mse_blind, 3)), str(np.round(mse_blind_raw, 3)), str(round(rmse_blind, 3)), str(np.round(rmse_blind_raw, 3)), str(round(pcc_blind, 3)), str(np.round(pcc_blind_raw, 3)), str(round(r2score_blind, 3)), str(np.round(r2score_blind_raw, 3)),
                str(round(mae_blind_cv, 3)), str(np.round(mae_blind_cv_raw, 3)), str(round(mse_blind_cv, 3)), str(np.round(mse_blind_cv_raw, 3)), str(round(rmse_blind_cv, 3)), str(np.round(rmse_blind_cv_raw, 3)), str(round(pcc_blind_cv, 3)), str(np.round(pcc_blind_cv_raw, 3)), str(round(r2score_blind_cv, 3)), str(np.round(r2score_blind_cv_raw, 3)),
            )
        ))
    else:
        mae, mse, rmse, pcc, r2score = result_cv
        mae_blind, mse_blind, rmse_blind, pcc_blind, r2score_blind = result_blind
        mae_blind_cv, mse_blind_cv, rmse_blind_cv, pcc_blind_cv, r2score_blind_cv = result_blind_cv
        print('\t'.join(
            (
                str(param),
                str(round(mae, 3)), str(round(mse, 3)), str(round(rmse, 3)), str(round(pcc, 3)), str(round(r2score, 3)),
                str(round(mae_blind, 3)), str(round(mse_blind, 3)), str(round(rmse_blind, 3)), str(round(pcc_blind, 3)), str(round(r2score_blind, 3)),
                str(round(mae_blind_cv, 3)), str(round(mse_blind_cv, 3)), str(round(rmse_blind_cv, 3)), str(round(pcc_blind_cv, 3)), str(round(r2score_blind_cv, 3)),
            )
        ))


def gs_svr_rbf_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    if target.ndim > 1:
        print_gs_mr_output_header(show_raw=True)
    else:
        print_gs_mr_output_header(show_raw=False)
    for kernel in svr_rbf_param_grid['kernel']:
        for c in svr_rbf_param_grid['C']:
            for gamma in svr_rbf_param_grid['gamma']:
                for epsilon in svr_rbf_param_grid['epsilon']:
                    param = {
                        'kernel': kernel,
                        'C': c,
                        'gamma': gamma,
                        'epsilon': epsilon,
                    }
                    model = MRModel(e_id='SVR', data=data, target=target, k=k, do_shuffle=True)
                    model.learn(param, preprocess)
                    result_cv = model.predict_with_cv(preprocess=preprocess)
                    result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                    result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                    if target.ndim > 1:
                        print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                    else:
                        print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return


def gs_svr_poly_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    if target.ndim > 1:
        print_gs_mr_output_header(show_raw=True)
    else:
        print_gs_mr_output_header(show_raw=False)
    for kernel in svr_poly_param_grid['kernel']:
        for c in svr_poly_param_grid['C']:
            for degree in svr_poly_param_grid['degree']:
                for gamma in svr_poly_param_grid['gamma']:
                    for coef0 in svr_poly_param_grid['coef0']:
                        for epsilon in svr_poly_param_grid['epsilon']:
                            param = {
                                'kernel': kernel,
                                'C': c,
                                'degree': degree,
                                'gamma': gamma,
                                'coef0': coef0,
                                'epsilon': epsilon,
                            }
                            model = MRModel(e_id='SVR', data=data, target=target, k=k, do_shuffle=True)
                            model.learn(param, preprocess)
                            result_cv = model.predict_with_cv(preprocess=preprocess)
                            result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                            result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                            if target.ndim > 1:
                                print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                            else:
                                print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return


def gs_svr_linear_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    if target.ndim > 1:
        print_gs_mr_output_header(show_raw=True)
    else:
        print_gs_mr_output_header(show_raw=False)
    for kernel in svr_linear_param_grid['kernel']:
        for c in svr_linear_param_grid['C']:
            for epsilon in svr_linear_param_grid['epsilon']:
                param = {
                    'kernel': kernel,
                    'C': c,
                    'epsilon': epsilon,
                }
                model = MRModel(e_id='SVR', data=data, target=target, k=k, do_shuffle=True)
                model.learn(param, preprocess)
                result_cv = model.predict_with_cv(preprocess=preprocess)
                result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                if target.ndim > 1:
                    print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                else:
                    print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return


def gs_rfr_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    if target.ndim > 1:
        print_gs_mr_output_header(show_raw=True)
    else:
        print_gs_mr_output_header(show_raw=False)
    for criterion in rfr_param_grid['criterion']:
        for n_estimators in rfr_param_grid['n_estimators']:
            for max_depth in rfr_param_grid['max_depth']:
                for max_features in rfr_param_grid['max_features']:
                    param = {
                        'criterion': criterion,
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'max_features': max_features,
                    }
                    model = MRModel(e_id='RFR', data=data, target=target, k=k, do_shuffle=True)
                    model.learn(param, preprocess)
                    result_cv = model.predict_with_cv(preprocess=preprocess)
                    result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                    result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                    if target.ndim > 1:
                        print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                    else:
                        print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return


def gs_etr_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    if target.ndim > 1:
        print_gs_mr_output_header(show_raw=True)
    else:
        print_gs_mr_output_header(show_raw=False)
    for criterion in rfr_param_grid['criterion']:
        for n_estimators in rfr_param_grid['n_estimators']:
            for max_depth in rfr_param_grid['max_depth']:
                for max_features in rfr_param_grid['max_features']:
                    param = {
                        'criterion': criterion,
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'max_features': max_features,
                    }
                    model = MRModel(e_id='ETR', data=data, target=target, k=k, do_shuffle=True)
                    model.learn(param, preprocess)
                    result_cv = model.predict_with_cv(preprocess=preprocess)
                    result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                    result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                    if target.ndim > 1:
                        print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                    else:
                        print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return


def gs_mlpr_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    if target.ndim > 1:
        print_gs_mr_output_header(show_raw=True)
    else:
        print_gs_mr_output_header(show_raw=False)
    for activation in mlpr_param_grid['activation']:
        for hls in mlpr_param_grid['hidden_layer_sizes']:
            for lri in mlpr_param_grid['learning_rate_init']:
                param = {
                    'activation': activation,
                    'hidden_layer_sizes': hls,
                    'learning_rate_init': lri
                }
                model = MRModel(e_id='MLPR', data=data, target=target, k=k, do_shuffle=True)
                model.learn(param, preprocess)
                result_cv = model.predict_with_cv(preprocess=preprocess)
                result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                if target.ndim > 1:
                    print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                else:
                    print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return


def gs_en_with_cv(
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        k: int = 5,
        preprocess: bool = False,
        ignore_warn: bool = True
) -> None:
    if ignore_warn:
        import warnings
        warnings.filterwarnings('ignore')
    print('\t'.join(
        (
            'params', 'MAE', 'MSE', 'RMSE', 'PCC', 'R2_score',
            'MAE_blind', 'MSE_blind', 'RMSE_blind', 'PCC_blind', 'R2_score_blind',
            'MAE_blind_cv', 'MSE_blind_cv', 'RMSE_blind_cv', 'PCC_blind_cv', 'R2_score_blind_cv',
        )
    ))
    for alpha in en_param_grid['alpha']:
            for l1_ratio in en_param_grid['l1_ratio']:
                param = {
                    'alpha': float(alpha),
                    'l1_ratio': float(l1_ratio),
                }
                model = MRModel(e_id='EN', data=data, target=target, k=k, do_shuffle=True)
                model.learn(param, preprocess)
                result_cv = model.predict_with_cv(preprocess=preprocess)
                result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
                result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
                if target.ndim > 1:
                    print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=True)
                else:
                    print_gs_mr_output_row(param, result_cv, result_blind, result_blind_cv, show_raw=False)
    return

