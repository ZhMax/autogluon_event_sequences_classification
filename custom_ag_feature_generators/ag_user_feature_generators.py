import os

import pandas as pd
import numpy as np

import autogluon.tabular as ag_tab
import autogluon.features.generators as ag_ftrs_g
import autogluon.common.features as ag_common_ftrs

import category_encoders as ce
from category_encoders import wrapper as ce_wrapper

import tsfresh.feature_extraction as tsf_ftrs_extrct


class FloatMemoryMinimizeFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    """
    Clips and converts dtype of float features to minimize memory usage.

    dtype_out : np.dtype, default np.float32
        dtype to clip and convert features to.
        Clipping will automatically use the correct min and max values for the dtype provided.
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, dtype_out=np.float32, **kwargs):
        super().__init__(**kwargs)
        self.dtype_out, self._clip_min, self._clip_max = self._get_dtype_clip_args(dtype_out)

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X):
        return self._minimize_numeric_memory_usage(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[ag_common_ftrs.types.R_FLOAT])

    @staticmethod
    def _get_dtype_clip_args(dtype) -> tuple((np.dtype, int, int)):
        try:
            dtype_info = np.finfo(dtype)
        except ValueError:
            dtype_info = np.finfo(dtype)
        return dtype_info.dtype, dtype_info.min, dtype_info.max

    def _minimize_numeric_memory_usage(self, X: pd.DataFrame):
        df = X.clip(self._clip_min, self._clip_max)
        df = df.astype(self.dtype_out)
        return df

    def _more_tags(self):
        return {'feature_interactions': False}
    

class DropDuplicatedRowsFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    """
    Drop duplicated rows from the dataset.

    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        X_out = self._transform(X)
        
        return X_out, self.feature_metadata_in.type_group_map_special 

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num_duplicated_rows = X.duplicated().sum()
        X_out = X.drop_duplicates().reset_index(drop=True)
        print(f"{num_duplicated_rows} duplicated rows have been dropped!")

        return X_out 

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict()  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter. 


class SortValuesByColsFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    """
    Sort values in the dataset in ascending order by the given column names

    cols_to_sort : list[str]
        list containing names of columns to sort
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, cols_to_sort, **kwargs):
        super().__init__(**kwargs)
        self.cols_to_sort = cols_to_sort

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        X_out = self._transform(X)
        
        return X_out, self.feature_metadata_in.type_group_map_special 

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.sort_values(by=self.cols_to_sort).reset_index(drop=True)

        return X_out 

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[ag_common_ftrs.types.R_INT, ag_common_ftrs.types.R_FLOAT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter. 


class StandartizationFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    """
    Standartize values in the dataset df

    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    eps = 10**(-8)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.X_train_mean = None
        self.X_train_std = None

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        self._get_standartization_parameters(X)
        X_out = self._transform(X)

        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = (X - self.X_train_mean) / (self.X_train_std + self.eps)
        return X_out

    
    def _get_standartization_parameters(self, X: pd.DataFrame):
        self.X_train_mean = X.mean()
        self.X_train_std = X.std()

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[ag_common_ftrs.types.R_FLOAT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


class BinaryGLMMencoderFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    """
    Apply Generalized Linear Mixed Model Encoder to encode columns with int values depending on
    values of binary target 

    int_cols_to_transform: list[str]
        list containing names of columns with int values to encode
    label: str
        label of the target column
    **kwargs :
        Refer to :class:`AbstractFeatureGenerator` documentation for details on valid key word arguments.
    """
    def __init__(self, int_cols_to_transform, label, **kwargs):
        super().__init__(**kwargs)
        self.int_cols_to_transform = int_cols_to_transform
        self.label = label
        
        self.encoder = ce.GLMMEncoder(cols=self.int_cols_to_transform)

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        
        self._fit_encoder(X)
        X_out = self._transform(X)
        
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_nolabel = X.drop(columns=[self.label])
        y = X[self.label]
        X_out = self.encoder.transform(X_nolabel)
        X_out[self.label] = y

        return X_out
    
    def _fit_encoder(self, X: pd.DataFrame):
        
        X_train = X.drop(columns=[self.label])
        y_train = X[self.label]

        self.encoder.fit(X_train, y_train)
        print("ce.encoder have been fitted")

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[ag_common_ftrs.types.R_INT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


class MulticlassGLMMencoderFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    def __init__(self, int_cols_to_transform, label, **kwargs):
        super().__init__(**kwargs)
        self.int_cols_to_transform = int_cols_to_transform
        self.label = label
        
        self.encoder = ce.GLMMEncoder(cols=self.int_cols_to_transform)
        self.wrapper = ce_wrapper.PolynomialWrapper(self.encoder)


    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        
        self._fit_encoder(X)
        X_out = self._transform(X)
        
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_nolabel = X.drop(columns=[self.label])
        y = X[self.label]
        X_out = self.wrapper.transform(X_nolabel)
        X_out[self.label] = y


        return X_out
    
    def _fit_encoder(self, X: pd.DataFrame):
        
        X_train = X.drop(columns=[self.label])
        y_train = X[self.label]

        self.wrapper.fit(X_train, y_train)
        print("ce.wrapper have been fitted")

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[ag_common_ftrs.types.R_INT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.
    

class TSfreshFeatureGenerator(ag_ftrs_g.AbstractFeatureGenerator):
    def __init__(self, id_col, label, tsfresh_user_features, **kwargs):
        super().__init__(**kwargs)

        self.id_col = id_col
        self.label = label
        self.settings_for_feature_extraction = tsf_ftrs_extrct.settings.MinimalFCParameters()
        self.tsfresh_user_features = tsfresh_user_features
        
    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple((pd.DataFrame, dict)):
        X_out = self._transform(X)
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = self._tsfresh_extract_features(X)

        return X_out

    def _tsfresh_extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_ftr = X.drop(columns=[self.label])
        y = X.groupby(by=self.id_col)[self.label].unique()\
             .apply(lambda x: x[0] if len(x) == 1 else x)
        
        real_cols = X_ftr.drop(columns=[self.id_col])
        kind_to_fc_parameters = {}
        for col in real_cols:
            kind_to_fc_parameters[col] = self.tsfresh_user_features

        X_out = tsf_ftrs_extrct.extract_features(X_ftr, 
                                                 column_id=self.id_col,
                                                 default_fc_parameters=self.settings_for_feature_extraction,
                                                 kind_to_fc_parameters=kind_to_fc_parameters)
        X_out[self.label] = y
        return X_out     

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[ag_common_ftrs.types.R_INT, ag_common_ftrs.types.R_FLOAT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.