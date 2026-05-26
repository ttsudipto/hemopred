from typing import List
from ..ml.moe_model import MOEModel
# from shap.explainers import Permutation
from shap import Explainer
# from shap.maskers import Independent
from shap.plots import beeswarm, bar, scatter, heatmap, violin
from shap import Explanation
import numpy as np
import pickle

class ExplainerMOEModel:

    def __init__(self, model: MOEModel, preprocess: bool) -> None:
        self.background_model = model
        self.preprocess = preprocess
        self._create_background_dataset()
        self.explainer = None
        self.fitted = False

    def _create_background_dataset(self) -> None:
        self.background = self.background_model.data

    def is_fitted(self) -> bool:
        return self.fitted

    def predict_wrapper(self, test_data: np.ndarray) -> np.ndarray:
        y_pred = self.background_model.predict_labels_blind_without_cv(test_data, preprocess=self.preprocess)
        return y_pred

    def fit(self) -> None:
        self.fitted = False
        # self.explainer = Explainer(self.predict_wrapper, self.background, algorithm="permutation")
        self.explainer = Explainer(
            self.predict_wrapper,
            self.background,
            feature_names=self.background_model.feature_names,
            seed=42
        )
        self.fitted = True

    def explain(self, test_data: np.ndarray, silent: bool = False, save: bool = False, op_file_name: str = None) -> Explanation:
        if self.is_fitted():
            shap_values = self.explainer(test_data, silent=silent)
            if save:
                if op_file_name is not None:
                    f = open(op_file_name, 'wb')
                    pickle.dump(shap_values, f)
                    f.close()
                else:
                    raise ValueError('Error !!! Cannot save. The "op_file_name" must be provided for saving SHAP values ...')
            return shap_values
        raise RuntimeError('Error !!! Explainer not fitted ... ')

    def plot_shap_values(
            self,
            shap_values: Explanation,
            row_indices: List = None,
            labels: List = None,
            plot_type: str = 'bar',
            plot_cohort: bool = False
    ) -> None:
        if self.is_fitted():
            input_rows = shap_values if row_indices is None else shap_values[row_indices]
            if plot_cohort:
                if labels is not None:
                    if plot_type == 'bar':
                        bar(input_rows.cohorts(labels).abs.mean(0), max_display=50, show=True)
                    else:
                        raise ValueError('Error !!! Invalid plot_type for cohorts. Correct values = {bar} ...')
                else:
                    raise ValueError('Error !!! "labels" must be provided for cohort plot ...')
            else:
                if plot_type == 'bar':
                    bar(input_rows, max_display=50)
                elif plot_type == 'beeswarm':
                    beeswarm(input_rows)
                elif plot_type == 'scatter':
                    scatter(input_rows)
                elif plot_type == 'heatmap':
                    heatmap(input_rows)
                elif plot_type == 'violin':
                    violin(input_rows)
                else:
                    raise ValueError('Error !!! Invalid plot_type. Correct values = {bar, beeswarm, scatter, heatmap, violin} ...')
        else:
            raise RuntimeError('Error !!! Explainer not fitted ... ')
