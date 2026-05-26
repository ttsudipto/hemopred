from ..io.resource import read_data
from ..ml.ml_utils import split_cv_blind_data, get_quantiles, get_cluster_labels
from ..ml.grid_search_moe import gs_with_cv
from ..ml.optimal import create_optimal_gating_model, create_optimal_moe_model
from ..pickler.pickler import dump_moe_model_to_file, load_moe_model_from_file
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
print(np.unique(res_hba2_labels))
print(np.unique(cv_res_hba2_labels))
gating_model = create_optimal_gating_model('HbA2', 'RF', res_hba2.data, res_hba2.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)

########
#  GS  #
########
# gs_with_cv(('GBR', 'SVR_rbf'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'SVR_rbf'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'SVR_rbf'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'SVR_poly'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'SVR_poly'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'SVR_poly'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'SVR_linear'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'SVR_linear'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'SVR_linear'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'RFR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'RFR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'RFR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'ETR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'ETR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'ETR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'MLPR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'MLPR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'MLPR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)

####################
#  Optimal models  #
####################
# create_optimal_moe_model('HbA2', ('GBR', 'SVR_rbf'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('GBR', 'SVR_poly'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('GBR', 'SVR_linear'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('GBR', 'RFR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('GBR', 'ETR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('GBR', 'MLPR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('QR', 'SVR_rbf'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('QR', 'SVR_poly'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('QR', 'SVR_linear'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('QR', 'RFR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('QR', 'ETR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbA2', ('QR', 'MLPR'), cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)

#############################
#  Optimal model (overall)  #
#############################
optimal_model_hba2 = create_optimal_moe_model('HbA2', None, cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)

################
#  Save model  #
################
# dump_moe_model_to_file(optimal_model_hba2, 'HbA2')
# dump_moe_model_to_file(optimal_model_hba2, 'HbA2', without_data=True)

################
#  Load model  #
################
optimal_model_hba2_loaded = load_moe_model_from_file('HbA2')
result_blind = optimal_model_hba2_loaded.predict_blind_without_cv(blind_res_hba2.data, blind_res_hba2.target, preprocess=True)
print(result_blind)
# optimal_model_hba2_loaded_wo_data = load_moe_model_from_file('HbA2', without_data=True)
# result_blind = optimal_model_hba2_loaded_wo_data.predict_blind_without_cv(blind_res_hba2.data, blind_res_hba2.target, preprocess=True)
# print(result_blind)


##############################  HbF  ##############################

###########
#  Input  #
###########
res_hbf = read_data(input_file_name, n_features=10, targets='HbF', verbose=True)
cv_res_hbf, blind_res_hbf = split_cv_blind_data(res_hbf, test_size=0.2, return_resource=True, verbose=True)
lower_quantile_value, upper_quantile_value = get_quantiles(res_hbf.target, q_lower, q_upper)
res_hbf_labels = get_cluster_labels(res_hbf.target, lower_quantile_value, upper_quantile_value)
cv_res_hbf_labels = get_cluster_labels(cv_res_hbf.target, lower_quantile_value, upper_quantile_value)
print(np.unique(res_hbf_labels))
print(np.unique(cv_res_hbf_labels))
gating_model = create_optimal_gating_model('HbF', 'RF', res_hbf.data, res_hbf.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)

########
#  GS  #
########
# gs_with_cv(('GBR', 'SVR_rbf'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'SVR_rbf'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'SVR_rbf'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'SVR_poly'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'SVR_poly'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'SVR_poly'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'SVR_linear'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'SVR_linear'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'SVR_linear'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'RFR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'RFR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'RFR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'ETR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'ETR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'ETR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
#
# gs_with_cv(('GBR', 'MLPR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('QR', 'MLPR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)
# gs_with_cv(('HGBR', 'MLPR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, ignore_warn=False)

####################
#  Optimal models  #
####################
# create_optimal_moe_model('HbF', ('GBR', 'SVR_rbf'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('GBR', 'SVR_poly'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('GBR', 'SVR_linear'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('GBR', 'RFR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('GBR', 'ETR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('GBR', 'MLPR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('QR', 'SVR_rbf'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('QR', 'SVR_poly'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('QR', 'SVR_linear'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('QR', 'RFR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('QR', 'ETR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)
# create_optimal_moe_model('HbF', ('QR', 'MLPR'), cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)

#############################
#  Optimal model (overall)  #
#############################
optimal_model_hbf = create_optimal_moe_model('HbF', None, cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, preprocess=True, verbose=True)

################
#  Save model  #
################
# dump_moe_model_to_file(optimal_model_hbf, 'HbF')
# dump_moe_model_to_file(optimal_model_hbf, 'HbF', without_data=True)

################
#  Load model  #
################
optimal_model_hbf_loaded = load_moe_model_from_file('HbF')
result_blind = optimal_model_hbf_loaded.predict_blind_without_cv(blind_res_hbf.data, blind_res_hbf.target, preprocess=True)
print(result_blind)
# optimal_model_hbf_loaded_wo_data = load_moe_model_from_file('HbF', without_data=True)
# result_blind = optimal_model_hbf_loaded_wo_data.predict_blind_without_cv(blind_res_hbf.data, blind_res_hbf.target, preprocess=True)
# print(result_blind)

