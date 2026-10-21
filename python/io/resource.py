from typing import Sequence, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sys import maxsize
from copy import deepcopy
import numpy as np
import csv

np.set_printoptions(threshold=maxsize)


class DataResource:
    path_prefix = 'input/'
    delimiter = '\t'
    header_lines = 1
    header_offset = 3
    age_index = 3
    patient_id_index = 1
    hbf_index = 13
    hbf_target_name = 'HbF'
    hba2_index = 14
    hba2_target_name = 'HbA2'
    # target_index = 14
    target_label_index = 18
    age_filter_criteria = 18 ## 18 years
    study_labels = [
        'Normal',
        'Beta Thalassemia Trait',
        'Borderline high HbA2',
        'HbD Punjab Trait',
        'Sickle Cell Trait',
        'HbD-Punjab/Beta Thalassemia (double heterozygous)',
    ]
    compound_labels = [
        'HbS/Beta Thalassemia (double heterozygous)',
        'HbD-Punjab/Beta Thalassemia (double heterozygous)',
        'HbE/Beta Thalassemia (double heterozygous)',
        'Delta-Beta Thalassemia Trait',
        'HbQ India/Beta Thalassemia (double heterozygous)',
        'HbE Trait with coexisting alpha thalassemia',
        'HbS/HbD Punjab (double heterozygous)'
    ]

    def __init__(self, file_name: str, n_features: int, targets: str) -> None:
        self.file_name = DataResource.path_prefix + file_name
        self.n_features = n_features
        self.data = None
        self.target = None
        self.target_labels_original = None
        self.target_labels = None
        self.headers = None
        self.header_indices = None
        self.patient_ids = None
        self.scaler = None
        self.imputer = None
        self._find_target_indices(targets)

    def _find_target_indices(self, targets: str) -> None:
        if targets == 'HbA2':
            self.target_indices = (DataResource.hba2_index,)
            self.target_names = (DataResource.hba2_target_name,)
        elif targets == 'HbF':
            self.target_indices = (DataResource.hbf_index,)
            self.target_names = (DataResource.hbf_target_name,)
        elif targets == 'HbA2_HbF':
            self.target_indices = (DataResource.hba2_index, DataResource.hbf_index)
            self.target_names = (DataResource.hba2_target_name, DataResource.hbf_target_name)
        else:
            raise ValueError('Error !!! Invalid target specification. Correct values = {HbA2, HbF, HbA2_HbF} ... ')

    def _read_headers(self) -> Sequence[str]:
        csvfile = open(self.file_name, 'r')
        reader = csv.DictReader(csvfile, delimiter=DataResource.delimiter)
        headers = reader.fieldnames[DataResource.header_offset: DataResource.header_offset + self.n_features]
        csvfile.close()
        return headers

    def _create_header_indices(self) -> Dict[str, int]:
        header_indices = {header: i for i, header in enumerate(self.headers)}
        return header_indices

    def _read_data(self) -> np.ndarray:
        csvfile = open(self.file_name, 'r')
        data = np.genfromtxt(csvfile,
                             delimiter=DataResource.delimiter,
                             skip_header=DataResource.header_lines,
                             usecols=range(DataResource.header_offset,
                                           DataResource.header_offset + self.n_features),
                             dtype=np.float32)
        csvfile.close()
        return data

    def _read_patient_ids(self) -> np.ndarray:
        csvfile = open(self.file_name, 'r')
        patient_ids = np.genfromtxt(
            csvfile,
            delimiter=DataResource.delimiter,
            skip_header=DataResource.header_lines,
            usecols=(DataResource.patient_id_index,),
            dtype=str
        )
        csvfile.close()
        return patient_ids

    def _read_target(self) -> np.ndarray:
        csvfile = open(self.file_name, 'r')
        target = np.genfromtxt(
            csvfile,
            delimiter=DataResource.delimiter,
            skip_header=DataResource.header_lines,
            usecols=self.target_indices
        )
        csvfile.close()
        return target

    def _read_target_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        csvfile = open(self.file_name, 'r')
        target_labels_original = np.genfromtxt(
            csvfile,
            delimiter=DataResource.delimiter,
            skip_header=DataResource.header_lines,
            usecols=(DataResource.target_label_index,),
            dtype=str
        )
        csvfile.close()
        compound_indices = np.isin(target_labels_original, DataResource.compound_labels)
        target_labels = deepcopy(target_labels_original)
        target_labels[compound_indices] = 'Compound or heterogeneous'
        return target_labels_original, target_labels

    def _remove_unwanted_labels(self) -> None:
        label_filter = np.isin(self.target_labels_original, DataResource.study_labels)
        self.data = self.data[label_filter]
        self.target = self.target[label_filter]
        self.target_labels = self.target_labels[label_filter]
        self.target_labels_original = self.target_labels_original[label_filter]
        self.patient_ids = self.patient_ids[label_filter]

    def _age_filter(self) -> None:
        index = DataResource.age_index - DataResource.header_offset
        age_filter = np.where(self.data[:, index] >= DataResource.age_filter_criteria)
        self.data = self.data[age_filter]
        self.target = self.target[age_filter]
        self.target_labels = self.target_labels[age_filter]
        self.target_labels_original = self.target_labels_original[age_filter]
        self.patient_ids = self.patient_ids[age_filter]

    def _missing_target_filter(self) -> None:
        if self.target.ndim > 1:
            target_filter = np.full((self.target_labels.shape[0],), True, dtype=bool)
            for j in range(self.target.shape[1]):
                target_filter = target_filter & (~ np.isnan(self.target[:, j].ravel()))
        else:
            target_filter = ~ np.isnan(self.target)
        self.data = self.data[target_filter]
        self.target = self.target[target_filter]
        self.target_labels = self.target_labels[target_filter]
        self.target_labels_original = self.target_labels_original[target_filter]
        self.patient_ids = self.patient_ids[target_filter]

    def _impute_missing_values_knn(self, imputer: KNNImputer = None, scaler: StandardScaler = None) -> None:
        if imputer is None:
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='distance')
            if scaler is None:
                scaler = StandardScaler()
                imputed_data_scaled = imputer.fit_transform(scaler.fit_transform(self.data))
            else:
                imputed_data_scaled = imputer.fit_transform(scaler.transform(self.data))
        else:
            if scaler is None:
                scaler = StandardScaler()
                imputed_data_scaled = imputer.transform(scaler.fit_transform(self.data))
            else:
                imputed_data_scaled = imputer.transform(scaler.transform(self.data))
        self.data = scaler.inverse_transform(imputed_data_scaled)
        self.scaler = deepcopy(scaler)
        self.imputer = deepcopy(imputer)

    def read_from_csv(self, do_impute: bool = False, imputer: KNNImputer = None, scaler: StandardScaler = None) -> None:
        self.headers = self._read_headers()
        self.header_indices = self._create_header_indices()
        self.data = self._read_data()
        self.patient_ids = self._read_patient_ids()
        self.target = self._read_target()
        self.target_labels_original, self.target_labels = self._read_target_labels()
        self._remove_unwanted_labels()
        # print(self.data.shape, self.target.shape)
        ## Filtering ##
        self._age_filter()
        # print(self.data.shape, self.target.shape)
        self._missing_target_filter()
        # print(self.data.shape, self.target.shape)
        ## Imputation ##
        if do_impute:
            self._impute_missing_values_knn(imputer=imputer, scaler=scaler)

def print_data_summary(res: DataResource) -> None:
    print()
    print('Data - "' + res.file_name + '"')
    print('Data: ' + str(res.data.shape))
    print('Target: ' + str(res.target.shape))
    print('Patient IDs: ' + str(res.patient_ids.shape))
    print('Number of features: ' + str(len(res.headers)))
    print('Features: ' + str(res.headers))
    print()


def read_data(
        file_name: str,
        n_features: int,
        targets: str,
        do_impute: bool = True,
        imputer: KNNImputer = None,
        scaler: StandardScaler = None,
        verbose: bool = False
) -> DataResource:
    res = DataResource(file_name, n_features, targets)
    res.read_from_csv(do_impute=do_impute, imputer=imputer, scaler=scaler)
    if verbose:
        print_data_summary(res)
    return res


def data_characteristics(res: DataResource) -> None:
    heading = ["Feature"]
    for label in np.unique(res.target_labels_original):
        heading.append(label)
    print('\t'.join(heading))
    min_iqrs = []
    max_iqrs = []
    means = []
    medians = []
    for label in np.unique(res.target_labels_original):
        data = res.data[res.target_labels_original == label]
        target = res.target[res.target_labels_original == label]
        means.append(np.mean(data, axis=0).tolist() + np.mean(target, axis=0).tolist())
        medians.append(np.median(data, axis=0).tolist() + np.median(target, axis=0).tolist())
        iqr_start_data, iqr_end_data = np.percentile(data, (25, 75), axis=0)
        iqr_start_target, iqr_end_target = np.percentile(target, (25, 75), axis=0)
        min_iqrs.append(iqr_start_data.tolist() + iqr_start_target.tolist())
        max_iqrs.append(iqr_end_data.tolist() + iqr_end_target.tolist())
    # print(means)
    # print(len(means))
    for i, feature in enumerate(res.headers + list(res.target_names)):
        op_row = [feature]
        for j, label in enumerate(np.unique(res.target_labels_original)):
            op_row.append(
                'Mean=' + str(round(means[j][i], 2)) +
                ', Median=' + str(round(medians[j][i], 2)) +
                ', IQR=[' + str(round(min_iqrs[j][i], 2)) + ', ' + str(round(max_iqrs[j][i], 2)) + ']'
            )
        print('\t'.join(op_row))

