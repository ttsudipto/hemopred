from joblib import dump, load
from pathlib import Path
from python.ml.gating_model import GatingModel
from python.ml.moe_model import MOEModel
from python.ml.affinity_classifier import AffinityClassifier
from python.explanation.explainer import ExplainerMOEModel
from .model_metadata import GatingModelMetadata, MOEModelMetadata, AffinityClassifierModelMetadata
import pickle

gating_hba2_path_prefix = 'output/models/gating/hba2/'
gating_hbf_path_prefix = 'output/models/gating/hbf/'
moe_hba2_path_prefix = 'output/models/moe/hba2/'
moe_hbf_path_prefix = 'output/models/moe/hbf/'

gating_hba2_wo_data_path_prefix = 'output/models_wo_data/gating/hba2/'
gating_hbf_wo_data_path_prefix = 'output/models_wo_data/gating/hbf/'
moe_hba2_wo_data_path_prefix = 'output/models_wo_data/moe/hba2/'
moe_hbf_wo_data_path_prefix = 'output/models_wo_data/moe/hbf/'

ac_path_prefix = 'output/models/affinity_classifier/'


def dump_gating_model_to_file(model: GatingModel, target_type: str, without_data: bool = False) -> None:
    if model.is_trained_total():
        if target_type == 'HbA2':
            if without_data:
                path_prefix = gating_hba2_wo_data_path_prefix
            else:
                path_prefix = gating_hba2_path_prefix
        elif target_type == 'HbF':
            if without_data:
                path_prefix = gating_hbf_wo_data_path_prefix
            else:
                path_prefix = gating_hbf_path_prefix
        else:
            raise ValueError('Error !!! Invalid target_type. Correct values = {HbA2, HbF} ... ')
        if not Path(path_prefix).exists():
            Path(path_prefix).mkdir(parents=True)
        # old_rec_limit = sys.getrecursionlimit()
        # sys.setrecursionlimit(0x1000)
        dump(model.gating_model, path_prefix + 'gating_model_' + target_type.lower() + '.joblib')
        if model.data_preprocessor is not None:
            dump(model.data_preprocessor, path_prefix + 'gating_model_' + target_type.lower() + '_data_preprocessor.joblib')
        # sys.setrecursionlimit(old_rec_limit)
        md = GatingModelMetadata(model, without_data=without_data)
        metadata_file = open(path_prefix + 'gating_model_' + target_type.lower() + '_metadata.pkl', 'wb')
        pickle.dump(md, metadata_file)
        metadata_file.close()
        return
    raise ValueError('Error !!! Model not trained ...')


def dump_moe_model_to_file(model: MOEModel, target_type: str, without_data: bool = False) -> None:
    if model.is_trained_total():
        if target_type == 'HbA2':
            if without_data:
                path_prefix = moe_hba2_wo_data_path_prefix
            else:
                path_prefix = moe_hba2_path_prefix
        elif target_type == 'HbF':
            if without_data:
                path_prefix = moe_hbf_wo_data_path_prefix
            else:
                path_prefix = moe_hbf_path_prefix
        else:
            raise ValueError('Error !!! Invalid target_type. Correct values = {HbA2, HbF} ... ')
        if not Path(path_prefix).exists():
            Path(path_prefix).mkdir(parents=True)
        # old_rec_limit = sys.getrecursionlimit()
        # sys.setrecursionlimit(0x1000)
        dump_gating_model_to_file(model.gating_model, target_type, without_data=without_data)
        dump(model.total_uq_expert, path_prefix + 'moe_' + target_type.lower() + '_total_uq_expert.joblib')
        dump(model.total_mid_expert, path_prefix + 'moe_' + target_type.lower() + '_total_mid_expert.joblib')
        dump(model.total_lq_expert, path_prefix + 'moe_' + target_type.lower() + '_total_lq_expert.joblib')
        if model.data_preprocessor is not None:
            dump(model.data_preprocessor, path_prefix + 'moe_' + target_type.lower() + '_data_preprocessor.joblib')
        if model.target_preprocessor is not None:
            dump(model.target_preprocessor, path_prefix + 'moe_' + target_type.lower() + '_target_preprocessor.joblib')
        # sys.setrecursionlimit(old_rec_limit)
        md = MOEModelMetadata(model, without_data=without_data)
        metadata_file = open(path_prefix + 'moe_' + target_type.lower() + '_metadata.pkl', 'wb')
        pickle.dump(md, metadata_file)
        metadata_file.close()
        return
    raise ValueError('Error !!! Model not trained ...')


def dump_affinity_classifier(model: AffinityClassifier) -> None:
    print(model.is_trained())
    print(model.preprocess)
    if model.is_trained():
        if not Path(ac_path_prefix).exists():
            Path(ac_path_prefix).mkdir(parents=True)
        # old_rec_limit = sys.getrecursionlimit()
        # sys.setrecursionlimit(0x1000)
        if model.preprocess:
            dump(model.data_preprocessor, ac_path_prefix + 'ac_data_preprocessor.joblib')
        # sys.setrecursionlimit(old_rec_limit)
        md = AffinityClassifierModelMetadata(model)
        metadata_file = open(ac_path_prefix + 'affinity_classifier_metadata.pkl', 'wb')
        pickle.dump(md, metadata_file)
        metadata_file.close()
        return
    raise ValueError('Error !!! Model not trained ...')


def load_gating_model_from_file(target_type: str, without_data: bool = False) -> GatingModel:
    if target_type == 'HbA2':
        if without_data:
            path_prefix = gating_hba2_wo_data_path_prefix
        else:
            path_prefix = gating_hba2_path_prefix
    elif target_type == 'HbF':
        if without_data:
            path_prefix = gating_hbf_wo_data_path_prefix
        else:
            path_prefix = gating_hbf_path_prefix
    else:
        raise ValueError('Error !!! Invalid target_type. Correct values = {HbA2, HbF} ... ')
    metadata_file = open(path_prefix + 'gating_model_' + target_type.lower() + '_metadata.pkl', 'rb')  # load metadata
    md = pickle.load(metadata_file, encoding='latin1')
    metadata_file.close()
    model = md.get_model()  # `model` without estimators
    model.gating_model = load(path_prefix + 'gating_model_' + target_type.lower() + '.joblib')  # load gating_model
    try:
        model.data_preprocessor = load(path_prefix + 'gating_model_' + target_type.lower() + '_data_preprocessor.joblib')  # load data_preprocessor
    finally:
        pass
    return model


def load_moe_model_from_file(target_type: str, without_data: bool = False) -> MOEModel:
    if target_type == 'HbA2':
        if without_data:
            path_prefix = moe_hba2_wo_data_path_prefix
        else:
            path_prefix = moe_hba2_path_prefix
    elif target_type == 'HbF':
        if without_data:
            path_prefix = moe_hbf_wo_data_path_prefix
        else:
            path_prefix = moe_hbf_path_prefix
    else:
        raise ValueError('Error !!! Invalid target_type. Correct values = {HbA2, HbF} ... ')
    gating_model = load_gating_model_from_file(target_type, without_data=without_data)  # load gating_model
    metadata_file = open(path_prefix + 'moe_' + target_type.lower() + '_metadata.pkl', 'rb')  # load metadata
    md = pickle.load(metadata_file, encoding='latin1')
    metadata_file.close()
    model = md.get_model(gating_model)  # `model` without estimators
    model.total_uq_expert = load(path_prefix + 'moe_' + target_type.lower() + '_total_uq_expert.joblib')  # load total_uq_expert
    model.total_mid_expert = load(path_prefix + 'moe_' + target_type.lower() + '_total_mid_expert.joblib')  # load total_mid_expert
    model.total_lq_expert = load(path_prefix + 'moe_' + target_type.lower() + '_total_lq_expert.joblib')  # load total_lq_expert
    try:
        model.data_preprocessor = load(path_prefix + 'moe_' + target_type.lower() + '_data_preprocessor.joblib')  # load data_preprocessor
        model.target_preprocessor = load(path_prefix + 'moe_' + target_type.lower() + '_target_preprocessor.joblib')  # load target_preprocessor
    finally:
        pass
    return model


def load_affinity_classifier_from_file() -> AffinityClassifier:
    metadata_file = open(ac_path_prefix + 'affinity_classifier_metadata.pkl', 'rb')  # load metadata
    md = pickle.load(metadata_file, encoding='latin1')
    metadata_file.close()
    model = md.get_model()  # `model` without estimators
    try:
        model.data_preprocessor = load(ac_path_prefix + 'ac_data_preprocessor.joblib')  # load data_preprocessor
    except:
        pass
    return model

