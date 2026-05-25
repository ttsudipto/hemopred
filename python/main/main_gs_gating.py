from ..io.resource import read_data
from ..ml.ml_utils import get_quantiles, get_cluster_labels
from ..ml.grid_search_gating import gs_gating_rf_with_cv
import numpy as np


input_file_name = 'PGIMER_HPLC_2024_CBC_Data_v2.tsv'
q_lower, q_upper = 20, 80

##############################  HbA2  ##############################

###########
#  Input  #
###########
# res_hba2 = read_data(input_file_name, n_features=10, targets='HbA2', verbose=True)
# lower_quantile_value, upper_quantile_value = get_quantiles(res_hba2.target, q_lower, q_upper)
# res_hba2_labels = get_cluster_labels(res_hba2.target, lower_quantile, upper_quantile)
# print(np.unique(res_hba2_labels))

########
#  GS  #
########
# gs_gating_rf_with_cv(res_hba2.data, res_hba2.target, lower_quantile_value=lower_quantile_value, upper_quantile_value=upper_quantile_value, preprocess=True, ignore_warn=False)


##############################  HbF  ##############################

###########
#  Input  #
###########
# res_hbf = read_data(input_file_name, n_features=10, targets='HbF', verbose=True)
# lower_quantile_value, upper_quantile_value = get_quantiles(res_hbf.target, q_lower, q_upper)
# res_hbf_labels = get_cluster_labels(res_hbf.target, lower_quantile, upper_quantile)
# print(np.unique(res_hbf_labels))

########
#  GS  #
########
# gs_gating_rf_with_cv(res_hbf.data, res_hbf.target, lower_quantile_value=lower_quantile_value, upper_quantile_value=upper_quantile_value, preprocess=True, ignore_warn=False)

