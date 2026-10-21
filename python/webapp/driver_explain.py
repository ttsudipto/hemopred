from typing import Dict
import sys
import os
os.environ['MPLCONFIGDIR'] = '/tmp/'
os.environ["XDG_CACHE_HOME"] = '/tmp/'
import json
import warnings
import numpy as np
from .input_wrapper import RegressionInput
from ..pickler.pickler import load_moe_explainer_model_from_file
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

def explain(data: np.ndarray, target: str) -> Dict :
    moe_explainer = load_moe_explainer_model_from_file(target, without_data=True)
    shap_values = moe_explainer.explain(data, silent=True)
    return {
        'SHAP': shap_values.values[0].tolist(),
    }


in_wrapper = parse_json(sys.argv[1])
target_variable = sys.argv[2]
output = explain(in_wrapper.get_ndarray(), target_variable)
print(json.dumps(output))
