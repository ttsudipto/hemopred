from ..io.resource import read_data
from ..ml.ml_utils import split_cv_blind_data
from ..ml.grid_search_mr import gs_svr_rbf_with_cv
from ..ml.grid_search_mr import gs_svr_poly_with_cv
from ..ml.grid_search_mr import gs_svr_linear_with_cv
from ..ml.grid_search_mr import gs_rfr_with_cv
from ..ml.grid_search_mr import gs_etr_with_cv
from ..ml.grid_search_mr import gs_en_with_cv
from ..ml.grid_search_mr import gs_mlpr_with_cv
from ..ml.optimal import create_optimal_mr_model


input_file_name = 'PGIMER_HPLC_2024_CBC_Data_v2.tsv'

##############################  HbA2  ##############################

###########
#  Input  #
###########
res_hba2 = read_data(input_file_name, n_features=10, targets='HbA2', verbose=True)
cv_data_hba2, blind_data_hba2, cv_target_hba2, blind_target_hba2 = split_cv_blind_data(res_hba2, verbose=True)

########
#  GS  #
########
# gs_svr_rbf_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)
# gs_svr_poly_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)
# gs_svr_linear_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)
# gs_rfr_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)
# gs_etr_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)
# gs_en_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)
# gs_mlpr_with_cv(cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, preprocess=True, ignore_warn=False)

####################
#  Optimal models  #
####################
create_optimal_mr_model('HbA2', 'SVR_rbf', cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbA2', 'SVR_poly', cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbA2', 'SVR_linear', cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbA2', 'RFR', cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbA2', 'ETR', cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbA2', 'MLPR', cv_data_hba2, cv_target_hba2, blind_data_hba2, blind_target_hba2, k=5, preprocess=True, verbose=True)


##############################  HbF  ##############################

###########
#  Input  #
###########
res_hbf = read_data(input_file_name, n_features=10, targets='HbF', verbose=True)
cv_data_hbf, blind_data_hbf, cv_target_hbf, blind_target_hbf = split_cv_blind_data(res_hbf, verbose=True)

########
#  GS  #
########
# gs_svr_rbf_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)
# gs_svr_poly_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)
# gs_svr_linear_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)
# gs_rfr_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)
# gs_etr_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)
# gs_en_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)
# gs_mlpr_with_cv(cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, preprocess=True, ignore_warn=False)

####################
#  Optimal models  #
####################
create_optimal_mr_model('HbF', 'SVR_rbf', cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbF', 'SVR_poly', cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbF', 'SVR_linear', cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbF', 'RFR', cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbF', 'ETR', cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, k=5, preprocess=True, verbose=True)
create_optimal_mr_model('HbF', 'MLPR', cv_data_hbf, cv_target_hbf, blind_data_hbf, blind_target_hbf, k=5, preprocess=True, verbose=True)

