from ..io.resource import read_data
from ..ml.ml_utils import split_cv_blind_data, get_quantiles, get_cluster_labels
from ..ml.optimal import create_optimal_gating_model, create_optimal_moe_model
from ..explanation.explainer import ExplainerMOEModel
import matplotlib
import pickle

matplotlib.use("TkAgg")

input_file_name = 'PGIMER_HPLC_2024_CBC_Data_v2.tsv'
q_lower, q_upper = 20, 80

friendly_label_name_map = {
    'Beta Thalassemia Trait': 'Beta Thalassemia\nTrait',
    'Normal': 'Normal',
    'HbD Punjab Trait': 'HbD Punjab Trait',
    'Sickle Cell Trait': 'Sickle Cell Trait',
    'Borderline high HbA2': 'Borderline HbA2',
    'Compound or heterogeneous': 'HbD Punjab/Beta\nThalassemia (double\nheterozygous)',
}


##############################  HbA2  ##############################

###########
#  Input  #
###########

res_hba2 = read_data(input_file_name, n_features=10, targets='HbA2', verbose=True)
cv_res_hba2, blind_res_hba2 = split_cv_blind_data(res_hba2, test_size=0.2, return_resource=True, verbose=True)
lower_quantile_value, upper_quantile_value = get_quantiles(res_hba2.target, q_lower, q_upper)
res_hba2_labels = get_cluster_labels(res_hba2.target, lower_quantile_value, upper_quantile_value)
cv_res_hba2_labels = get_cluster_labels(cv_res_hba2.target, lower_quantile_value, upper_quantile_value)
gating_model = create_optimal_gating_model('HbA2', 'RF', res_hba2.data, res_hba2.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)
optimal_model_hba2 = create_optimal_moe_model('HbA2', None, cv_res_hba2.data, cv_res_hba2.target, blind_res_hba2.data, blind_res_hba2.target, cv_res_hba2_labels, gating_model, (q_lower/100, q_upper/100), k=5, feature_names=cv_res_hba2.headers, preprocess=True, verbose=True)

###################
#  Fit explainer  #
###################

hba2_explainer = ExplainerMOEModel(optimal_model_hba2, preprocess=True)
hba2_explainer.fit()

#############
#  Explain  #
#############

# # shap_values_hba2 = hba2_explainer.explain(res_hba2.data, save=True, op_file_name='output/explanation/moe_hba2_shap_values.pkl')
# shap_values_hba2 = hba2_explainer.explain(res_hba2.data[:5])
# print(shap_values_hba2.values)

#############################
#  Load saved explanations  #
#############################

f = open('output/explanation/moe_hba2_shap_values.pkl', 'rb')
shap_values_hba2 = pickle.load(f)
f.close()
hba2_explainer.plot_shap_values(shap_values_hba2, plot_type='bar')
friendly_labels_hba2 = list(map(friendly_label_name_map.get, res_hba2.target_labels))
hba2_explainer.plot_shap_values(shap_values_hba2, plot_type='bar', labels=friendly_labels_hba2, plot_cohort=True)


##############################  HbF  ##############################

###########
#  Input  #
###########

res_hbf = read_data(input_file_name, n_features=10, targets='HbF', verbose=True)
cv_res_hbf, blind_res_hbf = split_cv_blind_data(res_hbf, test_size=0.2, return_resource=True, verbose=True)
lower_quantile_value, upper_quantile_value = get_quantiles(res_hbf.target, q_lower, q_upper)
res_hbf_labels = get_cluster_labels(res_hbf.target, lower_quantile_value, upper_quantile_value)
cv_res_hbf_labels = get_cluster_labels(cv_res_hbf.target, lower_quantile_value, upper_quantile_value)
gating_model = create_optimal_gating_model('HbF', 'RF', res_hbf.data, res_hbf.target, lower_quantile_value, upper_quantile_value, k=5, preprocess=True, verbose=True)
optimal_model_hbf = create_optimal_moe_model('HbF', None, cv_res_hbf.data, cv_res_hbf.target, blind_res_hbf.data, blind_res_hbf.target, cv_res_hbf_labels, gating_model, (q_lower/100, q_upper/100), k=5, feature_names=cv_res_hbf.headers, preprocess=True, verbose=True)

###################
#  Fit explainer  #
###################

hbf_explainer = ExplainerMOEModel(optimal_model_hbf, preprocess=True)
hbf_explainer.fit()

#############
#  Explain  #
#############

# # shap_values_hbf = hbf_explainer.explain(res_hbf.data, save=True, op_file_name='output/explanation/moe_hbf_shap_values.pkl')
# shap_values_hbf = hbf_explainer.explain(res_hbf.data[:5])
# print(shap_values_hbf.values)

#############################
#  Load saved explanations  #
#############################

f = open('output/explanation/moe_hbf_shap_values.pkl', 'rb')
shap_values_hbf = pickle.load(f)
f.close()
hbf_explainer.plot_shap_values(shap_values_hbf, plot_type='bar')
friendly_labels_hbf = list(map(friendly_label_name_map.get, res_hbf.target_labels))
hbf_explainer.plot_shap_values(shap_values_hbf, plot_type='bar', labels=friendly_labels_hbf, plot_cohort=True)

