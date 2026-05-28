from ..io.resource import read_data
from ..ml.ml_utils import split_cv_blind_data, get_quantiles, get_cluster_labels
from ..ml.optimal import create_optimal_gating_model, create_optimal_moe_model
from ..pickler.pickler import load_moe_model_from_file
from ..uncertainty.moe_wrapper import MOEModelWrapper
from ..uncertainty.mapie import MAPIEModel
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score
import numpy as np

input_file_name = 'PGIMER_HPLC_2024_CBC_Data_v2.tsv'
q_lower, q_upper = 20, 80


##############################  HbA2  ##############################

###########
#  Input  #
###########
res_hba2 = read_data(input_file_name, n_features=10, targets='HbA2', verbose=True)
cv_res_hba2, blind_res_hba2 = split_cv_blind_data(res_hba2, test_size=0.2, return_resource=True, verbose=True)
lower_quantile_value, upper_quantile_value = get_quantiles(res_hba2.target, q_lower, q_upper)
res_hba2_labels = get_cluster_labels(res_hba2.target, lower_quantile_value, upper_quantile_value)
cv_res_hba2_labels = get_cluster_labels(cv_res_hba2.target, lower_quantile_value, upper_quantile_value)

#########
#  MOE  #
#########
# gating_model_hba2 = create_optimal_gating_model('HbA2', 'RF', res_hba2.data, res_hba2.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)
# optimal_hba2_moe_model = create_optimal_moe_model('HbA2', None, cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model_hba2, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
optimal_hba2_moe_model = load_moe_model_from_file('HbA2', without_data=True)
y_pred = optimal_hba2_moe_model.predict_labels_blind_without_cv(res_hba2.data, preprocess=True)

###########
#  MAPIE  #
###########
optimal_hba2_moe_model_wrapper = MOEModelWrapper(optimal_hba2_moe_model, preprocess=True)
mapie_hba2_model = MAPIEModel(optimal_hba2_moe_model_wrapper, blind_res_hba2.data, blind_res_hba2.target)
mapie_hba2_model.learn({'confidence_level': 0.95, 'conformity_score': 'absolute'})
y_pred_mapie, y_pis = mapie_hba2_model.predict(res_hba2.data)

print('\t'.join(('Patient ID', 'Label', 'HbA2', 'Predicted HbA2', 'PI_low', 'PI_high')))
for i in range(res_hba2.data.shape[0]):
    print('\t'.join((res_hba2.patient_ids[i], res_hba2.target_labels[i], str(res_hba2.target[i]), str(y_pred[i]), str(y_pis[i][0][0]), str(y_pis[i][1][0]))))
print(regression_coverage_score(res_hba2.target, y_pis)[0])
print(regression_mean_width_score(y_pis))
print()


##############################  HbF  ##############################

###########
#  Input  #
###########
res_hbf = read_data(input_file_name, n_features=10, targets='HbF', verbose=True)
cv_res_hbf, blind_res_hbf = split_cv_blind_data(res_hbf, test_size=0.2, return_resource=True, verbose=True)
lower_quantile_value, upper_quantile_value = get_quantiles(res_hbf.target, q_lower, q_upper)
res_hbf_labels = get_cluster_labels(res_hbf.target, lower_quantile_value, upper_quantile_value)
cv_res_hbf_labels = get_cluster_labels(cv_res_hbf.target, lower_quantile_value, upper_quantile_value)

#########
#  MOE  #
#########
# gating_model_hbf = create_optimal_gating_model('HbF', 'RF', res_hbf.data, res_hbf.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)
# optimal_hbf_moe_model = create_optimal_moe_model('HbF', None, cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model_hbf, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
optimal_hbf_moe_model = load_moe_model_from_file('HbF', without_data=True)
y_pred = optimal_hbf_moe_model.predict_labels_blind_without_cv(res_hbf.data, preprocess=True)

###########
#  MAPIE  #
###########
optimal_hbf_moe_model_wrapper = MOEModelWrapper(optimal_hbf_moe_model, preprocess=True)
mapie_hbf_model = MAPIEModel(optimal_hbf_moe_model_wrapper, blind_res_hbf.data, blind_res_hbf.target)
mapie_hbf_model.learn({'confidence_level': 0.95, 'conformity_score': 'absolute'})
y_pred_mapie, y_pis = mapie_hbf_model.predict(res_hbf.data)

print('\t'.join(('Patient ID', 'Label', 'HbF', 'Predicted HbF', 'PI_low', 'PI_high')))
for i in range(res_hbf.data.shape[0]):
    print('\t'.join((res_hbf.patient_ids[i], res_hbf.target_labels[i], str(res_hbf.target[i]), str(y_pred[i]), str(y_pis[i][0][0]), str(y_pis[i][1][0]))))
print(regression_coverage_score(res_hbf.target, y_pis)[0])
print(regression_mean_width_score(y_pis))
print()
