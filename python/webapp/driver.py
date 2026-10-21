from typing import Dict
import sys
import os
os.environ['MPLCONFIGDIR'] = '/tmp/'
os.environ["XDG_CACHE_HOME"] = '/tmp/'
import json
import warnings
import numpy as np
from .input_wrapper import RegressionInput
from ..pickler.pickler import load_moe_model_from_file, load_affinity_classifier_from_file, load_mapie_model_from_file
# import logging
# logging.getLogger('shap').setLevel(logging.WARNING)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

def convert_to_float(s: str) -> float:
    if s == '' :
        return 0.0
    else :
        return float(s)

def parse_json(json_string: str) :
    #print(j_string)
    json_dict = json.loads(json_string)
    wrapper = RegressionInput()
    params = wrapper.get_all_params()
    for p in params :
        wrapper.add_value(convert_to_float(json_dict[p]))
    return wrapper

def predict(data: np.ndarray) -> Dict :
    optimal_model_hba2 = load_moe_model_from_file('HbA2', without_data=True)
    optimal_model_hbf = load_moe_model_from_file('HbF', without_data=True)
    pred_hba2 = optimal_model_hba2.predict_labels_blind_without_cv(data, preprocess=True)
    pred_hbf = optimal_model_hbf.predict_labels_blind_without_cv(data, preprocess=True)
    hb_data = np.array((pred_hba2[0], pred_hbf[0])).reshape(1, -1)

    mapie_hba2_model = load_mapie_model_from_file('HbA2', without_data=True)
    _, pi_hba2 = mapie_hba2_model.predict(data)
    mapie_hbf_model = load_mapie_model_from_file('HbF', without_data=True)
    _, pi_hbf = mapie_hbf_model.predict(data)

    optimal_affinity_classifier = load_affinity_classifier_from_file()
    labels = optimal_affinity_classifier.classes_
    pred_class = optimal_affinity_classifier.predict(hb_data)
    pred_distance = optimal_affinity_classifier.predict_distance(hb_data)
    pred_affinity = optimal_affinity_classifier.predict_affinity_confidence(hb_data)
    pred_proba = optimal_affinity_classifier.predict_proba(hb_data)

    return {
        'HbA2': float(pred_hba2[0]),
        'HbF': float(pred_hbf[0]),
        'PI_HbA2': [pi_hba2[0][0][0], pi_hba2[0][1][0]],
        'PI_HbF': [pi_hbf[0][0][0], pi_hbf[0][1][0]],
        'predicted_class': pred_class[0],
        'labels': labels.tolist(),
        'distance': pred_distance[0].tolist(),
        'affinity': pred_affinity[0].tolist(),
        'proba': pred_proba[0].tolist()
    }


in_wrapper = parse_json(sys.argv[1])
output = predict(in_wrapper.get_ndarray())
print(json.dumps(output))
