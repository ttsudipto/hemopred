from ..io.resource import read_data
from ..ml.affinity_classifier import AffinityClassifier
from ..ml.ml_utils import compute_binary_metrics_with_ci, compute_multiclass_metrics_with_ci
from ..pickler.pickler import dump_affinity_classifier, load_affinity_classifier_from_file
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, log_loss, top_k_accuracy_score, recall_score, f1_score, \
    balanced_accuracy_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.pyplot import xticks
from csv import DictReader
from copy import deepcopy
import numpy as np
import matplotlib

matplotlib.use("TkAgg")


input_file_name = 'PGIMER_HPLC_2024_CBC_Data_v2.tsv'
q_lower, q_upper = 20, 80

################
#  Input true  #
################
res_hba2 = read_data(input_file_name, n_features=10, targets='HbA2', verbose=True)
res_hbf = read_data(input_file_name, n_features=10, targets='HbF', verbose=True)
data = np.vstack((res_hba2.target, res_hbf.target)).transpose()
target = res_hba2.target_labels_original
print(data.shape, target.shape)
print(np.unique(target))
cv_data, blind_data, cv_target, blind_target = train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=target
    )
print(cv_data.shape, blind_data.shape, cv_target.shape, blind_target.shape)

###############################
#  Perform CV with HPLC data  #
###############################
# ac_model_cv = AffinityClassifier(distance_metric='euclidean', affinity_method='inverse', preprocess=False)
ac_model_cv = AffinityClassifier(distance_metric='euclidean', affinity_method='rb_gaussian', preprocess=False)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(ac_model_cv, cv_data, cv_target, cv=cv)
y_proba_cv = cross_val_predict(ac_model_cv, cv_data, cv_target, cv=cv, method='predict_proba')
print(classification_report(cv_target, y_pred_cv))
mc_metrics_cv, mc_ci_cv = compute_multiclass_metrics_with_ci(cv_target, y_pred_cv, y_proba_cv, labels=np.unique(cv_target))
for k, v in mc_metrics_cv.items():
    print(k, v, mc_ci_cv[k].tolist())
# cm_cv = confusion_matrix(cv_target, y_pred_cv)
# display_labels = ['Beta\nThalassemia\nTrait', 'Borderline\nHbA2', 'HbD\nPunjab\nTrait', 'HbD Punjab/\nBeta\nThalassemia\n(double\nheterozygous)', 'Normal', 'Sickle\nCell\nTrait']
# matplotlib.pyplot.rcParams.update({'font.size': 20, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,10), layout='constrained')
# ConfusionMatrixDisplay(cm_cv, display_labels=display_labels).plot(cmap='Purples', xticks_rotation=0.0, ax=ax, colorbar=False)
# matplotlib.pyplot.show()

binary_target_cv = np.array(deepcopy(cv_target))
binary_target_cv[cv_target == 'Normal'] = 0
binary_target_cv[cv_target != 'Normal'] = 1
binary_target_cv = binary_target_cv.astype(int)
binary_pred_cv = np.array(deepcopy(y_pred_cv))
binary_pred_cv[y_pred_cv == 'Normal'] = 0
binary_pred_cv[y_pred_cv != 'Normal'] = 1
binary_pred_cv = binary_pred_cv.astype(int)
print(classification_report(binary_target_cv, binary_pred_cv))
bin_metrics_cv, bin_ci_cv = compute_binary_metrics_with_ci(binary_target_cv, binary_pred_cv)
for k, v in bin_metrics_cv.items():
    print(k, v, bin_ci_cv[k].tolist())
# cm_binary_cv = confusion_matrix(binary_target_cv, binary_pred_cv)
# ConfusionMatrixDisplay(cm_binary_cv, display_labels=('Normal', 'Disease')).plot(cmap='Purples', colorbar=False)
# matplotlib.pyplot.show()

##################################
#  Perform blind with HPLC data  #
##################################
# ac_model_total = AffinityClassifier(distance_metric='euclidean', affinity_method='inverse', preprocess=False)
ac_model_total = AffinityClassifier(distance_metric='euclidean', affinity_method='rb_gaussian', preprocess=False)
ac_model_total.fit(cv_data, cv_target)
y_pred_blind = ac_model_total.predict(blind_data)
y_proba_blind = ac_model_total.predict_proba(blind_data)
print(classification_report(blind_target, y_pred_blind))
mc_metrics_blind, mc_ci_blind = compute_multiclass_metrics_with_ci(blind_target, y_pred_blind, y_proba_blind, labels=np.unique(blind_target))
for k, v in mc_metrics_blind.items():
    print(k, v, mc_ci_blind[k].tolist())
# cm_blind = confusion_matrix(blind_target, y_pred_blind)
# display_labels = ['Beta\nThalassemia\nTrait', 'Borderline\nHbA2', 'HbD\nPunjab\nTrait', 'HbD Punjab/\nBeta\nThalassemia\n(double\nheterozygous)', 'Normal', 'Sickle\nCell\nTrait']
# matplotlib.pyplot.rcParams.update({'font.size': 20, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,10), layout='constrained')
# ConfusionMatrixDisplay(cm_blind, display_labels=display_labels).plot(cmap='Purples', xticks_rotation=0.0, ax=ax, colorbar=False)
# matplotlib.pyplot.show()

binary_target_blind = np.array(deepcopy(blind_target))
binary_target_blind[blind_target == 'Normal'] = 0
binary_target_blind[blind_target != 'Normal'] = 1
binary_target_blind = binary_target_blind.astype(int)
binary_pred_blind = np.array(deepcopy(y_pred_blind))
binary_pred_blind[y_pred_blind == 'Normal'] = 0
binary_pred_blind[y_pred_blind != 'Normal'] = 1
binary_pred_blind = binary_pred_blind.astype(int)
print(classification_report(binary_target_blind, binary_pred_blind))
bin_metrics_blind, bin_ci_blind = compute_binary_metrics_with_ci(binary_target_blind, binary_pred_blind)
for k, v in bin_metrics_blind.items():
    print(k, v, bin_ci_blind[k].tolist())
# cm_binary_blind = confusion_matrix(binary_target_blind, binary_pred_blind)
# ConfusionMatrixDisplay(cm_binary_blind, display_labels=('Normal', 'Disease')).plot(cmap='Purples', colorbar=False)
# matplotlib.pyplot.show()

######################
#  Input prediction  #
######################
file_hba2_moe_pred = open('output/prediction/prediction_hba2_moe.tsv', 'r')
reader_hba2 = DictReader(file_hba2_moe_pred, delimiter='\t')
predicted_hba2 = []
for row in reader_hba2:
    predicted_hba2.append(row['Predicted HbA2'])
file_hba2_moe_pred.close()
file_hbf_moe_pred = open('output/prediction/prediction_hbf_moe.tsv', 'r')
reader_hbf = DictReader(file_hbf_moe_pred, delimiter='\t')
predicted_hbf = []
for row in reader_hbf:
    predicted_hbf.append(row['Predicted HbF'])
file_hbf_moe_pred.close()
moe_pred_data = np.vstack((np.array(predicted_hba2), np.array(predicted_hbf))).transpose()
moe_pred_target = res_hba2.target_labels_original
print(moe_pred_data.shape, moe_pred_target.shape)
print(np.unique(moe_pred_target))
cv_moe_pred_data, blind_moe_pred_data, cv_moe_pred_target, blind_moe_pred_target = train_test_split(
        moe_pred_data,
        moe_pred_target,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=target
    )
print(cv_moe_pred_data.shape, blind_moe_pred_data.shape, cv_moe_pred_target.shape, blind_moe_pred_target.shape)

################################################
#  Perform prediction with MoE predicted data  #
################################################
y_pred_moe_pred_cv = ac_model_total.predict(cv_moe_pred_data)
y_proba_moe_pred_cv = ac_model_total.predict_proba(cv_moe_pred_data)
print(classification_report(cv_moe_pred_target, y_pred_moe_pred_cv))
mc_metrics_moe_pred_cv, mc_ci_moe_pred_cv = compute_multiclass_metrics_with_ci(cv_moe_pred_target, y_pred_moe_pred_cv, y_proba_moe_pred_cv, labels=np.unique(cv_moe_pred_target))
for k, v in mc_metrics_moe_pred_cv.items():
    print(k, v, mc_ci_moe_pred_cv[k].tolist())
# cm_moe_pred_cv = confusion_matrix(cv_moe_pred_target, y_pred_moe_pred_cv)
# display_labels = ['Beta\nThalassemia\nTrait', 'Borderline\nHbA2', 'HbD\nPunjab\nTrait', 'HbD Punjab/\nBeta\nThalassemia\n(double\nheterozygous)', 'Normal', 'Sickle\nCell\nTrait']
# matplotlib.pyplot.rcParams.update({'font.size': 20, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,10), layout='constrained')
# ConfusionMatrixDisplay(cm_moe_pred_cv, display_labels=display_labels).plot(cmap='Purples', xticks_rotation=0.0, ax=ax, colorbar=False)
# matplotlib.pyplot.show()

binary_moe_pred_target_cv = np.array(deepcopy(cv_moe_pred_target))
binary_moe_pred_target_cv[cv_moe_pred_target == 'Normal'] = 0
binary_moe_pred_target_cv[cv_moe_pred_target != 'Normal'] = 1
binary_moe_pred_target_cv = binary_moe_pred_target_cv.astype(int)
binary_pred_moe_pred_cv = np.array(deepcopy(y_pred_moe_pred_cv))
binary_pred_moe_pred_cv[y_pred_moe_pred_cv == 'Normal'] = 0
binary_pred_moe_pred_cv[y_pred_moe_pred_cv != 'Normal'] = 1
binary_pred_moe_pred_cv = binary_pred_moe_pred_cv.astype(int)
print(classification_report(binary_moe_pred_target_cv, binary_pred_moe_pred_cv))
bin_metrics_moe_pred_cv, bin_ci_moe_pred_cv = compute_binary_metrics_with_ci(binary_moe_pred_target_cv, binary_pred_moe_pred_cv)
for k, v in bin_metrics_moe_pred_cv.items():
    print(k, v, bin_ci_moe_pred_cv[k].tolist())
# cm_binary_moe_pred_cv = confusion_matrix(binary_moe_pred_target_cv, binary_pred_moe_pred_cv)
# ConfusionMatrixDisplay(cm_binary_moe_pred_cv, display_labels=('Normal', 'Disease')).plot(cmap='Purples', colorbar=False)
# matplotlib.pyplot.show()


y_pred_moe_pred_blind = ac_model_total.predict(blind_moe_pred_data)
y_proba_moe_pred_blind = ac_model_total.predict_proba(blind_moe_pred_data)
print(classification_report(blind_moe_pred_target, y_pred_moe_pred_blind))
mc_metrics_moe_pred_blind, mc_ci_moe_pred_blind = compute_multiclass_metrics_with_ci(blind_moe_pred_target, y_pred_moe_pred_blind, y_proba_moe_pred_blind, labels=np.unique(blind_moe_pred_target))
for k, v in mc_metrics_moe_pred_blind.items():
    print(k, v, mc_ci_moe_pred_blind[k].tolist())
# cm_moe_pred_blind = confusion_matrix(blind_moe_pred_target, y_pred_moe_pred_blind)
# display_labels = ['Beta\nThalassemia\nTrait', 'Borderline\nHbA2', 'HbD\nPunjab\nTrait', 'HbD Punjab/\nBeta\nThalassemia\n(double\nheterozygous)', 'Normal', 'Sickle\nCell\nTrait']
# matplotlib.pyplot.rcParams.update({'font.size': 20, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,10), layout='constrained')
# ConfusionMatrixDisplay(cm_moe_pred_blind, display_labels=display_labels).plot(cmap='Purples', xticks_rotation=0.0, ax=ax, colorbar=False)
# matplotlib.pyplot.show()

binary_moe_pred_target_blind = np.array(deepcopy(blind_moe_pred_target))
binary_moe_pred_target_blind[blind_moe_pred_target == 'Normal'] = 0
binary_moe_pred_target_blind[blind_moe_pred_target != 'Normal'] = 1
binary_moe_pred_target_blind = binary_moe_pred_target_blind.astype(int)
binary_pred_moe_pred_blind = np.array(deepcopy(y_pred_moe_pred_blind))
binary_pred_moe_pred_blind[y_pred_moe_pred_blind == 'Normal'] = 0
binary_pred_moe_pred_blind[y_pred_moe_pred_blind != 'Normal'] = 1
binary_pred_moe_pred_blind = binary_pred_moe_pred_blind.astype(int)
print(classification_report(binary_moe_pred_target_blind, binary_pred_moe_pred_blind))
bin_metrics_moe_pred_blind, bin_ci_moe_pred_blind = compute_binary_metrics_with_ci(binary_moe_pred_target_blind, binary_pred_moe_pred_blind)
for k, v in bin_metrics_moe_pred_blind.items():
    print(k, v, bin_ci_moe_pred_blind[k].tolist())
# cm_binary_moe_pred_blind = confusion_matrix(binary_moe_pred_target_blind, binary_pred_moe_pred_blind)
# ConfusionMatrixDisplay(cm_binary_moe_pred_blind, display_labels=('Normal', 'Disease')).plot(cmap='Purples', colorbar=False)
# matplotlib.pyplot.show()

#######################
#  Print predictions  #
#######################
# b_distances = ac_model_total.predict_distance(moe_pred_data)
# b_affinity_confidences = ac_model_total.predict_affinity_confidence(moe_pred_data)
# b_proba = ac_model_total.predict_proba(moe_pred_data)
# b_pred = ac_model_total.predict(moe_pred_data)
# print('\t'.join((
#     'Patient_ID', 'HbA2', 'HbF', 'Class', 'Predicted_HbA2', 'Predicted_HbF',
#     'Classes', 'Centroids', 'Distances', 'Affinity_confidences', 'Probabilities', 'Prediction'
# )))
# for i in range(moe_pred_data.shape[0]):
#     print('\t'.join((
#         str(res_hbf.patient_ids[i]), str(res_hba2.target[i]), str(res_hbf.target[i]), res_hbf.target_labels[i],
#         str(moe_pred_data[i, 0]), str(moe_pred_data[i, 1]),
#         str(ac_model_total.classes_.tolist()), str(ac_model_total.centroids.tolist()),
#         str(np.round(b_distances[i], 4)), str(np.round(b_affinity_confidences[i], 4)),
#         str(np.round(b_proba[i], 4)), str(b_pred[i]),
#     )))

##################
#   Save model   #
##################
# dump_affinity_classifier(ac_model_total)

##############################
#   Performance comparison   #
##############################
# binary_target = np.array(deepcopy(moe_pred_target))
# binary_target[moe_pred_target == 'Normal'] = 0
# binary_target[moe_pred_target != 'Normal'] = 1
# binary_target = binary_target.astype(int)
#
# b_pred = ac_model_total.predict(moe_pred_data)
# b_proba = ac_model_total.predict_proba(moe_pred_data)
# cm = confusion_matrix(moe_pred_target, b_pred)
# print(classification_report(moe_pred_target, b_pred))
# print(top_k_accuracy_score(moe_pred_target, b_proba, k=3))
# print(top_k_accuracy_score(moe_pred_target, b_proba, k=2))
# print(top_k_accuracy_score(moe_pred_target, b_proba, k=1))
# print(cm)
# binary_pred = np.array(deepcopy(b_pred))
# binary_pred[b_pred == 'Normal'] = 0
# binary_pred[b_pred != 'Normal'] = 1
# binary_pred = binary_pred.astype(int)
# cm_binary = confusion_matrix(binary_target, binary_pred)
# tn, fp, fn, tp = cm_binary.ravel()
# print(classification_report(binary_target, binary_pred))
# print(
#     accuracy_score(binary_target, binary_pred), recall_score(binary_target, binary_pred), (tn/1.0) / (tn+fp),
#     f1_score(binary_target, binary_pred), balanced_accuracy_score(binary_target, binary_pred)
# )
#
# ac_model_loaded = load_affinity_classifier_from_file()
# b_pred_loaded = ac_model_loaded.predict(moe_pred_data)
# b_proba_loaded = ac_model_loaded.predict_proba(moe_pred_data)
# cm_loaded = confusion_matrix(moe_pred_target, b_pred_loaded)
# print(classification_report(moe_pred_target, b_pred_loaded))
# print(top_k_accuracy_score(moe_pred_target, b_proba_loaded, k=3))
# print(top_k_accuracy_score(moe_pred_target, b_proba_loaded, k=2))
# print(top_k_accuracy_score(moe_pred_target, b_proba_loaded, k=1))
# print(cm_loaded)
# binary_pred_loaded = np.array(deepcopy(b_pred_loaded))
# binary_pred_loaded[b_pred_loaded == 'Normal'] = 0
# binary_pred_loaded[b_pred_loaded != 'Normal'] = 1
# binary_pred_loaded = binary_pred_loaded.astype(int)
# cm_binary_loaded = confusion_matrix(binary_target, binary_pred_loaded)
# tn, fp, fn, tp = cm_binary_loaded.ravel()
# print(classification_report(binary_target, binary_pred_loaded))
# print(
#     accuracy_score(binary_target, binary_pred_loaded), recall_score(binary_target, binary_pred_loaded),
#     (tn/1.0) / (tn+fp), f1_score(binary_target, binary_pred_loaded),
#     balanced_accuracy_score(binary_target, binary_pred_loaded)
# )



###############
#  Fit model  #
###############
# # ac_model = AffinityClassifier(distance_metric='euclidean', affinity_method='inverse', preprocess=False)
# ac_model = AffinityClassifier(distance_metric='euclidean', affinity_method='rb_gaussian', preprocess=False)
# ac_model.fit(data, target, feature_names=['HbA2', 'HbF'])

#######################
#  Model predictions  #
#######################
# b_distances = ac_model.predict_distance(moe_pred_data)
# b_affinity_confidences = ac_model.predict_affinity_confidence(moe_pred_data)
# b_proba = ac_model.predict_proba(moe_pred_data)
# b_pred = ac_model.predict(moe_pred_data)

