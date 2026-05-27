from ..io.resource import read_data
from ..ml.affinity_classifier import AffinityClassifier
from ..pickler.pickler import dump_affinity_classifier, load_affinity_classifier_from_file
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
b_data = np.vstack((np.array(predicted_hba2), np.array(predicted_hbf))).transpose()
b_target = res_hba2.target_labels_original
print(b_data.shape, b_target.shape)
print(np.unique(b_target))

###############
#  Fit model  #
###############
# ac_model = AffinityClassifier(distance_metric='euclidean', affinity_method='inverse', preprocess=False)
ac_model = AffinityClassifier(distance_metric='euclidean', affinity_method='rb_gaussian', preprocess=False)
ac_model.fit(data, target, feature_names=['HbA2', 'HbF'])

#######################
#  Model predictions  #
#######################
b_distances = ac_model.predict_distance(b_data)
b_affinity_confidences = ac_model.predict_affinity_confidence(b_data)
b_proba = ac_model.predict_proba(b_data)
b_pred = ac_model.predict(b_data)

#######################
#  Print predictions  #
#######################
# print('\t'.join((
#     'Patient_ID', 'HbA2', 'HbF', 'Class', 'Predicted_HbA2', 'Predicted_HbF',
#     'Classes', 'Centroids', 'Distances', 'Affinity_confidences', 'Probabilities', 'Prediction'
# )))
# for i in range(b_data.shape[0]):
#     print('\t'.join((
#         str(res_hbf.patient_ids[i]), str(res_hba2.target[i]), str(res_hbf.target[i]), res_hbf.target_labels[i],
#         str(b_data[i, 0]), str(b_data[i, 1]),
#         str(ac_model.classes.tolist()), str(ac_model.centroids.tolist()),
#         str(np.round(b_distances[i], 4)), str(np.round(b_affinity_confidences[i], 4)),
#         str(np.round(b_proba[i], 4)), str(b_pred[i]),
#     )))

#######################
#  Model performance  #
#######################
cm = confusion_matrix(b_target, b_pred)
print(classification_report(b_target, b_pred))
print(log_loss(b_target, b_proba))
print(top_k_accuracy_score(b_target, b_proba, k=3))
print(top_k_accuracy_score(b_target, b_proba, k=2))
print(top_k_accuracy_score(b_target, b_proba, k=1))
# display_labels = ['Beta\nThalassemia\nTrait', 'Borderline\nHbA2', 'HbD\nPunjab\nTrait', 'HbD Punjab/\nBeta\nThalassemia\n(double\nheterozygous)', 'Normal', 'Sickle\nCell\nTrait']
# matplotlib.pyplot.rcParams.update({'font.size': 20, 'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# fig, ax = matplotlib.pyplot.subplots(figsize=(10,10), layout='constrained')
# ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(cmap='Purples', xticks_rotation=0.0, ax=ax, colorbar=False)
# matplotlib.pyplot.show()

#################################
#  Model performance as binary  #
#################################
binary_target = np.array(deepcopy(target))
binary_target[target == 'Normal'] = 0
binary_target[target != 'Normal'] = 1
binary_target = binary_target.astype(int)
binary_pred = np.array(deepcopy(b_pred))
binary_pred[b_pred == 'Normal'] = 0
binary_pred[b_pred != 'Normal'] = 1
binary_pred = binary_pred.astype(int)
cm = confusion_matrix(binary_target, binary_pred)
tn, fp, fn, tp = cm.ravel()
print(classification_report(binary_target, binary_pred))
print(
    accuracy_score(binary_target, binary_pred), recall_score(binary_target, binary_pred), (tn/1.0) / (tn+fp),
    f1_score(binary_target, binary_pred), balanced_accuracy_score(binary_target, binary_pred)
)
# ConfusionMatrixDisplay(cm, display_labels=('Normal', 'Disease')).plot(cmap='Purples', colorbar=False)
# matplotlib.pyplot.show()

##################
#   Save model   #
##################
# dump_affinity_classifier(ac_model)

##################
#   Load model   #
##################
# ac_model_loaded = load_affinity_classifier_from_file()
# b_distances = ac_model_loaded.predict_distance(b_data)
# b_affinity_confidences = ac_model_loaded.predict_affinity_confidence(b_data)
# b_proba = ac_model_loaded.predict_proba(b_data)
# b_pred = ac_model_loaded.predict(b_data)

#######################
#  Loaded model performance  #
#######################
# cm = confusion_matrix(b_target, b_pred)
# print(classification_report(b_target, b_pred))
# print(log_loss(b_target, b_proba))
# print(top_k_accuracy_score(b_target, b_proba, k=3))
# print(top_k_accuracy_score(b_target, b_proba, k=2))
# print(top_k_accuracy_score(b_target, b_proba, k=1))
# ConfusionMatrixDisplay(cm, display_labels=np.unique(target)).plot(cmap='Purples', xticks_rotation=20.0, colorbar=False)
# matplotlib.pyplot.show()

