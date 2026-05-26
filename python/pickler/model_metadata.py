from python.ml.moe_model import MOEModel
from python.ml.gating_model import GatingModel
from python.ml.affinity_classifier import AffinityClassifier
from python.explanation.explainer import ExplainerMOEModel
from copy import deepcopy
import numpy as np


class GatingModelMetadata:
    def __init__(self, model: GatingModel, without_data: bool = False) -> None:
        self.without_data = without_data
        self.estimator_id = model.estimator_id
        self.n_folds = model.n_folds
        self.lower_quantile_value = model.lower_quantile_value
        self.upper_quantile_value = model.upper_quantile_value
        self.feature_names = model.feature_names
        self.trained_total = model.trained_total
        self.trained_cv = model.trained_cv
        # self.gating_model = None
        # self.gating_model_cv = None
        self.train_indices = model.train_indices
        self.test_indices = model.test_indices
        if self.without_data:
            self.data = np.ones(model.data.shape)
            self.regression_target = np.ones(model.regression_target.shape)
            self.target = np.ones(model.target.shape)
        else:
            self.data = model.data
            self.regression_target = model.regression_target
            self.target = model.target

    def get_model(self) -> GatingModel:
        model = GatingModel(
            e_id=self.estimator_id,
            data=deepcopy(self.data),
            target=deepcopy(self.target),
            lower_quantile_value=self.lower_quantile_value,
            upper_quantile_value=self.upper_quantile_value,
            k=self.n_folds,
            feature_names=self.feature_names,
        )
        model.trained_total = self.trained_total
        model.trained_cv = False
        model.train_indices = deepcopy(self.train_indices)
        model.test_indices = deepcopy(self.test_indices)
        model.target = deepcopy(self.target)
        return model


class MOEModelMetadata:
    def __init__(self, model: MOEModel, without_data: bool = False) -> None:
        self.without_data = without_data
        self.lower_eid = model.lower_eid
        self.mid_eid = model.mid_eid
        self.upper_eid = model.upper_eid
        self.n_folds = model.n_folds
        self.feature_names = model.feature_names
        # self.gating_model = gating_model
        self.lq = model.lq
        self.uq = model.uq
        self.trained_total = model.trained_total
        self.trained_cv = model.trained_cv
        # self.cv_lq_experts = []
        # self.cv_uq_experts = []
        # self.cv_mid_experts = []
        # self.total_lq_expert = None
        # self.total_uq_expert = None
        # self.total_mid_expert = None
        self.train_indices = model.train_indices
        self.test_indices = model.test_indices
        if self.without_data:
            self.data = np.ones(model.data.shape)
            self.target = np.ones(model.target.shape)
            self.clusters = np.ones(model.clusters.shape)
        else:
            self.data = model.data
            self.target = model.target
            self.clusters = model.clusters

    def get_model(self, gating_model: GatingModel) -> MOEModel:
        # model = Model(self.estimator_id, deepcopy(self.data), deepcopy(self.sample_names), k=self.n_folds, do_shuffle=False)
        model = MOEModel(
            (self.lower_eid, self.mid_eid, self.upper_eid),
            deepcopy(self.data),
            deepcopy(self.target),
            deepcopy(self.clusters),
            deepcopy(gating_model),
            (self.lq, self.uq),
            self.n_folds,
            self.feature_names,
        )
        model.trained_total = self.trained_total
        model.trained_cv = False
        model.train_indices = deepcopy(self.train_indices)
        model.test_indices = deepcopy(self.test_indices)
        model.clusters = deepcopy(self.clusters)
        return model


class AffinityClassifierModelMetadata:
    def __init__(self, model: AffinityClassifier) -> None:
        self.distance_metric = model.distance_metric
        self.affinity_method = model.affinity_method
        self.preprocess = model.preprocess
        self.feature_names = model.feature_names
        self.trained = model.trained
        self.centroids = model.centroids
        self.classes = model.classes
        # self.data_preprocessor = None

    def get_model(self) -> AffinityClassifier:
        model = AffinityClassifier(
            self.distance_metric,
            self.affinity_method,
            self.preprocess
        )
        model.feature_names = deepcopy(self.feature_names)
        # print(self.trained)
        model.trained = self.trained
        model.centroids = deepcopy(self.centroids)
        model.classes = deepcopy(self.classes)
        return model


class ExplainerMOEModelMetadata:
    def __init__(self, model: ExplainerMOEModel, without_data: bool = False) -> None:
        # self.background_model = model
        self.preprocess = model.preprocess
        # self.explainer = None
        self.fitted = model.fitted
        # if without_data:
        #     self.background = np.ones(model.background.shape)
        # else:
        #     self.background = None

    def get_model(self, background_model: MOEModel) -> ExplainerMOEModel:
        model = ExplainerMOEModel(
            deepcopy(background_model),
            self.preprocess
        )
        model.fitted = self.fitted
        return model

