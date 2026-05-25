from typing import Dict, Tuple, List
from .gating_model import GatingModel
from .moe_model import MOEModel
from .grid_search_gating import print_gs_gating_output_row
from .grid_search_moe import eid_map, print_gs_moe_output_row
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

optimal_params_moe = {
    'HbA2': {
        'GBR_SVR_rbf': [ # 0.162	0.095	0.308	0.917	0.838	0.151	0.084	0.289	0.928	0.861	0.153	0.084	0.29	0.928	0.861
            {'n_estimators': 50},
            {'C': 1, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'rbf'}
        ], 'GBR_SVR_poly': [ # 0.162	0.095	0.308	0.917	0.839	0.151	0.084	0.289	0.928	0.861	0.153	0.084	0.29	0.928	0.861
            {'n_estimators': 50},
            {'C': 1, 'coef0': 2, 'degree': 2, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'poly'}
        ], 'GBR_SVR_linear': [ # 0.162	0.096	0.308	0.917	0.838	0.151	0.084	0.29	0.928	0.861	0.153	0.084	0.29	0.928	0.86
            {'n_estimators': 50},
            {'C': 1, 'epsilon': 0.25, 'kernel': 'linear'}
        ], 'GBR_RFR': [ # 0.162	0.095	0.308	0.917	0.838	0.151	0.084	0.289	0.928	0.861	0.154	0.084	0.29	0.928	0.86
            {'n_estimators': 50},
            {'max_depth': None, 'max_features': 0.5, 'n_estimators': 50}
        ], 'GBR_ETR': [ # 0.162	0.095	0.308	0.917	0.838	0.151	0.084	0.289	0.928	0.861	0.153	0.084	0.29	0.928	0.861
            {'n_estimators': 50},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 150}
        ], 'GBR_MLPR': [ # 0.162	0.095	0.308	0.917	0.839	0.151	0.084	0.289	0.928	0.861	0.153	0.084	0.29	0.928	0.861
            {'n_estimators': 50},
            {'activation': 'relu', 'hidden_layer_sizes': (300, 150), 'learning_rate_init': 0.01}
        ], 'QR_SVR_rbf': [ # 0.205	0.129	0.359	0.896	0.783	0.199	0.121	0.347	0.901	0.8	0.199	0.12	0.347	0.901	0.8
            {'alpha': 0},
            {'C': 1, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'rbf'}
        ], 'QR_SVR_poly': [ # 0.205	0.129	0.359	0.896	0.783	0.198	0.12	0.347	0.901	0.8	0.198	0.12	0.347	0.901	0.8
            {'alpha': 0},
            {'C': 1, 'coef0': 1, 'degree': 3, 'epsilon': 0.01, 'gamma': 0.01, 'kernel': 'poly'}
        ], 'QR_SVR_linear': [ # 0.205	0.129	0.359	0.896	0.783	0.2	0.121	0.347	0.9	0.8	0.199	0.121	0.347	0.9	0.8
            {'alpha': 0},
            {'C': 1, 'epsilon': 0.01, 'kernel': 'linear'}
        ], 'QR_RFR': [ # 0.205	0.129	0.359	0.896	0.782	0.199	0.121	0.348	0.901	0.799	0.199	0.121	0.347	0.901	0.8
            {'alpha': 0},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 100}
        ], 'QR_ETR': [ # 0.205	0.129	0.359	0.896	0.782	0.199	0.121	0.348	0.901	0.8	0.199	0.121	0.347	0.901	0.8
            {'alpha': 0},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 50}
        ], 'QR_MLPR': [ # 0.204	0.129	0.359	0.896	0.783	0.198	0.12	0.347	0.901	0.8	0.198	0.12	0.347	0.901	0.8
            {'alpha': 0},
            {'activation': 'logistic', 'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.0001}
        ],
    }, 'HbF': {
        'GBR_SVR_rbf': [ # 0.138	0.075	0.273	0.768	0.587	0.143	0.081	0.285	0.765	0.583	0.142	0.082	0.286	0.764	0.581
            {'n_estimators': 50},
            {'C': 1, 'epsilon': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}
        ], 'GBR_SVR_poly': [ # 0.138	0.075	0.273	0.768	0.587	0.143	0.081	0.285	0.766	0.583	0.142	0.082	0.286	0.764	0.581
            {'n_estimators': 50},
            {'C': 5, 'coef0': 1, 'degree': 3, 'epsilon': 0.1, 'gamma': 0.0001, 'kernel': 'poly'}
        ], 'GBR_SVR_linear': [ # 0.138	0.075	0.273	0.768	0.586	0.142	0.081	0.285	0.765	0.583	0.142	0.082	0.286	0.764	0.581
            {'n_estimators': 50},
            {'C': 1, 'epsilon': 0.1, 'kernel': 'linear'}
        ], 'GBR_RFR': [ # 0.141	0.075	0.274	0.767	0.584	0.145	0.082	0.286	0.765	0.581	0.145	0.082	0.287	0.763	0.578
            {'n_estimators': 50},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 50}
        ], 'GBR_ETR': [ # 0.14	0.075	0.274	0.767	0.584	0.145	0.082	0.286	0.765	0.581	0.144	0.082	0.287	0.763	0.579
            {'n_estimators': 50},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 100}
        ], 'GBR_MLPR': [ # 0.139	0.075	0.273	0.768	0.586	0.143	0.081	0.285	0.766	0.583	0.143	0.082	0.286	0.764	0.58
            {'n_estimators': 50},
            {'activation': 'logistic', 'hidden_layer_sizes': (300, 150), 'learning_rate_init': 0.001}
        ], 'QR_SVR_rbf': [ # 0.139	0.072	0.267	0.782	0.606	0.144	0.077	0.277	0.781	0.606	0.144	0.077	0.277	0.781	0.606
            {'alpha': 0.1},
            {'C': 1, 'epsilon': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}
        ], 'QR_SVR_poly': [ # 0.139	0.072	0.267	0.782	0.606	0.144	0.077	0.277	0.781	0.606	0.144	0.077	0.277	0.781	0.606
            {'alpha': 0.1},
            {'C': 1, 'coef0': 1, 'degree': 2, 'epsilon': 0.1, 'gamma': 0.001, 'kernel': 'poly'}
        ], 'QR_SVR_linear': [ #0.139	0.071	0.267	0.781	0.606	0.143	0.077	0.277	0.781	0.606	0.143	0.077	0.277	0.781	0.606
            {'alpha': 0.1},
            {'C': 1, 'epsilon': 0.1, 'kernel': 'linear'}
        ], 'QR_RFR': [ # 0.142	0.072	0.268	0.781	0.602	0.146	0.077	0.278	0.78	0.603	0.147	0.077	0.278	0.78	0.603
            {'alpha': 0.1},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 50}
        ], 'QR_ETR': [ # 0.142	0.072	0.268	0.781	0.603	0.146	0.077	0.278	0.781	0.604	0.146	0.077	0.278	0.781	0.604
            {'alpha': 0.1},
            {'max_depth': None, 'max_features': 0.25, 'n_estimators': 50}
        ], 'QR_MLPR': [ # 0.141	0.072	0.267	0.782	0.604	0.144	0.077	0.278	0.781	0.606	0.145	0.077	0.278	0.781	0.605
            {'alpha': 0.1},
            {'activation': 'logistic', 'hidden_layer_sizes': (100, 100), 'learning_rate_init': 0.001}
        ],
    }
}

optimal_eids_moe = {
    'HbA2': ('GBR', 'SVR_poly'),
    'HbF': ('QR', 'SVR_rbf')
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

def get_optimal_moe_estimator_ids(target_id: str) -> Tuple[str, str]:
    return optimal_eids_moe[target_id]

def get_optimal_moe_params(
        target_id: str,
        e_ids: Tuple[str, str],
        verbose: bool = False
) -> List[Dict]:
    param = optimal_params_moe[target_id]['_'.join(e_ids)]
    if verbose:
        print(param)
    return param

def create_optimal_moe_model(
        target_id: str,
        e_ids: Tuple[str, str] | None,
        data: np.ndarray,
        target: np.ndarray,
        b_data: np.ndarray,
        b_target: np.ndarray,
        clusters: np.ndarray,
        gating_model: GatingModel,
        quantiles: Tuple[float, float],
        k: int = 5,
        feature_names: List[str] = None,
        preprocess: bool = False,
        verbose: bool = False
) -> MOEModel:
    if e_ids is None:
        e_ids = get_optimal_moe_estimator_ids(target_id)
    q_eid = eid_map[e_ids[0]]
    mid_eid = eid_map[e_ids[1]]
    q_param = optimal_params_moe[target_id]['_'.join(e_ids)][0]
    mid_param = optimal_params_moe[target_id]['_'.join(e_ids)][1]
    model = MOEModel(
        e_id=(q_eid, mid_eid, q_eid),
        data=data,
        target=target,
        clusters=clusters,
        gating_model=gating_model,
        quantiles=quantiles,
        k=k,
        feature_names=feature_names,
        do_shuffle=True
    )
    model.learn([q_param, mid_param, q_param], preprocess=preprocess)
    if verbose:
        result_cv = model.predict_with_cv(preprocess=preprocess)
        result_blind = model.predict_blind_without_cv(b_data, b_target, preprocess=preprocess)
        result_blind_cv = model.predict_blind_data(b_data, b_target, preprocess=preprocess)
        print_gs_moe_output_row([q_param, mid_param], result_cv, result_blind, result_blind_cv)
    return model

