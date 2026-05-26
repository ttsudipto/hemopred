from ..io.resource import read_data
from ..ml.ml_utils import split_cv_blind_data, get_quantiles, get_cluster_labels
from ..ml.optimal import create_optimal_gating_model, create_optimal_moe_model, create_optimal_mr_model
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
# print(np.unique(res_hba2_labels))
# print(np.unique(cv_res_hba2_labels))
gating_model = create_optimal_gating_model('HbA2', 'RF', res_hba2.data, res_hba2.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)

#########
#  MOE  #
#########
optimal_hba2_moe_model = create_optimal_moe_model('HbA2', None, cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
y_pred = optimal_hba2_moe_model.predict_labels_blind_without_cv(res_hba2.data, preprocess=True)
print('\t'.join(('Patient ID', 'Label', 'HbA2', 'Predicted HbA2')))
for i in range(res_hba2.data.shape[0]):
    print('\t'.join((res_hba2.patient_ids[i], res_hba2.target_labels[i], str(res_hba2.target[i]), str(y_pred[i]))))
print()

########
#  MR  #
########
optimal_hba2_mr_model = create_optimal_mr_model('HbA2', None, cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, k=5, preprocess=True, verbose=True)
y_pred = optimal_hba2_mr_model.predict_labels_blind_without_cv(res_hba2.data, preprocess=True)
print('\t'.join(('Patient ID', 'Label', 'HbA2', 'Predicted HbA2')))
for i in range(res_hba2.data.shape[0]):
    print('\t'.join((res_hba2.patient_ids[i], res_hba2.target_labels[i], str(res_hba2.target[i]), str(y_pred[i]))))
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
# print(np.unique(res_hbf_labels))
# print(np.unique(cv_res_hbf_labels))
gating_model = create_optimal_gating_model('HbF', 'RF', res_hbf.data, res_hbf.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)

#########
#  MOE  #
#########
optimal_hbf_moe_model = create_optimal_moe_model('HbF', None, cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
y_pred = optimal_hbf_moe_model.predict_labels_blind_without_cv(res_hbf.data, preprocess=True)
print('\t'.join(('Patient ID', 'Label', 'HbF', 'Predicted HbF')))
for i in range(res_hbf.data.shape[0]):
    print('\t'.join((res_hbf.patient_ids[i], res_hbf.target_labels[i], str(res_hbf.target[i]), str(y_pred[i]))))
print()

########
#  MR  #
########
optimal_hbf_mr_model = create_optimal_mr_model('HbF', None, cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, k=5, preprocess=True, verbose=True)
y_pred = optimal_hbf_mr_model.predict_labels_blind_without_cv(res_hbf.data, preprocess=True)
print('\t'.join(('Patient ID', 'Label', 'HbF', 'Predicted HbF')))
for i in range(res_hbf.data.shape[0]):
    print('\t'.join((res_hbf.patient_ids[i], res_hbf.target_labels[i], str(res_hbf.target[i]), str(y_pred[i]))))
print()
