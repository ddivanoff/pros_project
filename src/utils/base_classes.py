import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder         
from sklearn.feature_selection import VarianceThreshold 

from imblearn.ensemble import BalancedRandomForestClassifier

from boruta import BorutaPy


class FeatureSelection():
    
    
    """
    
    A class to perform feature selection from arbitrary data frame.
    
    
    ---------------------------------------------------------------
    
    
    Methods
    
    *******
    
    load_data(path)
    stratify_split(df, target, test_size, id_col)
    label_encoding(y_train, y_val, y_test, order)
    feature_encoding(X_train, X_val, X_test, categories)
    variance_threshold(X_train, X_val, X_test, threshold)
    run_boruta(X_train, X_val, X_test, y_train,
               n_estimators, criterion, max_iter, perc)
    pickle_export(path, file)
    pickle_import(path)
    
    ---------------------------------------------------------------
    
    Attributes
    
    ********
    
    id_train
    id_val
    id_test
    label_encoder
    feature_encoder
    variance_threshold
    clf_boruta
    boruta
    selected_features_boruta
    
    """
    
    
    def __init__(self):
        pass
    
    
    def load_data(self, path: str = None) -> pd.core.frame.DataFrame:
                
        if path is None:
            raise TypeError("Unsupported type of <<path>>. Expected - str.")
                                
        return pd.read_parquet(path)
        
        
    def stratify_split(self, df: pd.core.frame.DataFrame = None,
                             target: str = None, 
                             test_size: float = None,
                             id_col: str = None) -> (pd.core.frame.DataFrame,
                                                     pd.core.frame.DataFrame,
                                                     pd.core.frame.DataFrame,
                                                     pd.core.series.Series,
                                                     pd.core.series.Series,
                                                     pd.core.series.Series):
        
        
        """
        
        This method is to perform startify split of the initial data - df.
        
        
        ------------------------------------------------------------------
        
        Inputs
        
        *******
        
        df: pandas.core.frame.DataFrame, default = None
           Contains the raw data
           
        target: str, default = None
           The column name of the variable to predict
           
        test_size: float, default = None
           Ratio of the test size
           
        id_col: str, default = None
           The column name for unique record id
           
           
        ------------------------------------------------------------------   
        
        Returns
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame
           The splitted data frames
           
        y_train, y_val, y_test: pandas.core.series.Series
           The corresponding labels
           
        """
        
        
        if df is None:
            raise TypeError("Unsupported type of <<df>>. Expected - pandas.core.frame.DataFrame.")
            
        if not isinstance(df, pd.core.frame.DataFrame):
            raise TypeError("Unsupported type of <<df>>. Expected - pandas.core.frame.DataFrame.")
            
        if target is None:
            raise TypeError("Unsupported type of <<target>>. Expected - str.")
            
        if not isinstance(target, str):
            raise TypeError("Unsupported type of <<target>>. Expected - str.")
            
        if test_size is None:
            raise TypeError("Unsupported type of <<test_size>>. Expected - float.")
            
        if not isinstance(test_size, float):
            raise TypeError("Unsupported type of <<test_size>>. Expected - float.")
            
        if id_col is None:
            raise TypeError("Unsupported type of <<id_col>>. Expected - str.")
            
        if not isinstance(id_col, str):
            raise TypeError("Unsupported type of <<id_col>>. Expected - str.")
            
        if id_col not in df.columns:
            raise AttributeError(f"{id_col} not in {df}.")
            
        if target not in df.columns:
            raise AttributeError(f"{target} not in {df}.")
        
        
        
        X_train, X_val_test, y_train, y_val_test = train_test_split(df.drop(target, axis=1),
                                                                    df[target],
                                                                    test_size=test_size,
                                                                    random_state=42,
                                                                    stratify=df[target])
        
        X_val, X_test, y_val, y_test = train_test_split(X_val_test,
                                                y_val_test,
                                                test_size=0.5,
                                                random_state=42,
                                                stratify=y_val_test)
        
        del X_val_test, y_val_test
        
        print('\nUnique labels available:', df[target].unique())

        for label in df[target].unique():

            assert round(y_train[y_train == label].shape[0]/y_train.shape[0], 2)\
                == round(y_val[y_val == label].shape[0]/y_val.shape[0], 2)\
                == round(y_test[y_test == label].shape[0]/y_test.shape[0], 2),\
                f"Stratify crition violated for label {label}. Check the split again!"

            print(f'\nStratify criterion satisfied. Imbalance ratio for label {label}:',
                  round(y_train[y_train == label].shape[0]/y_train.shape[0], 2))
            
        assert len(
                      set(X_train[id_col].unique())\
        .intersection(set(X_val[id_col].unique()))\
        .intersection(set(X_test[id_col].unique()))
        ) == 0,\
        "Data Leackage between records from the split."
        
        assert df[id_col].value_counts().max() == 1,\
        "Presence of duplicate records. Check the data again!"
        
        id_train = X_train[id_col]
        self.id_train = id_train
        X_train.drop(id_col, axis=1, inplace=True)

        id_val = X_val[id_col]
        self.id_val = id_val
        X_val.drop(id_col, axis=1, inplace=True)

        id_test = X_test[id_col]
        self.id_test = id_test
        X_test.drop(id_col, axis=1, inplace=True)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def label_encoding(self, y_train: pd.core.series.Series = None,
                             y_val: pd.core.series.Series = None,
                             y_test: pd.core.series.Series = None,
                             order: list[str] = None) -> (pd.core.series.Series,
                                                          pd.core.series.Series,
                                                          pd.core.series.Series):
        

        """
        
        With this method one can label encode the target variable.
        
        
        ------------------------------------------------------------------
        
        Inputs
        
        *******
        
        y_train, y_val, y_test: pandas.core.series.Series
           All labels before encoding
           
        order: list[str], default = None
           Specifies the order of labelling
           
           
        ------------------------------------------------------------------   
        
        Returns
        
        *******
        
        y_train, y_val, y_test: pandas.core.series.Series
           All labels after encoding
           
        """

        
        if y_train is None or y_val is None or y_test is None:
            raise TypeError("Unsupported type for either <<y_train/val/test>>. Expected - pandas.core.series.Series.")

        if (
            not isinstance(y_train, pd.core.series.Series)
            or not isinstance(y_val, pd.core.series.Series)
            or not isinstance(y_test, pd.core.series.Series)
        ):
            raise TypeError("Unsupported type for either <<y_train/val/test>>. Expected - pandas.core.series.Series.")
        
        if order is None:
            raise TypeError("Unsupported type of <<order>>. Expected - list[str]")
                            
        if not isinstance(order, list):
            raise TypeError("Unsupported type of <<order>>. Expected - list[str]")              
        
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(order)
        self.label_encoder = label_encoder
        
        print("\nUnique labels before encoding:", y_train.unique())

        y_train = pd.Series(label_encoder.transform(y_train))
        y_val = pd.Series(label_encoder.transform(y_val))
        y_test = pd.Series(label_encoder.transform(y_test))

        print("\nUnique labels after encoding:", y_train.unique())
        
        self.y_train = y_train
        
        return y_train, y_val, y_test 
    
    def feature_encoding(self, X_train: pd.core.frame.DataFrame = None,
                               X_val: pd.core.frame.DataFrame = None,
                               X_test: pd.core.frame.DataFrame = None,
                               categories: list[str] = None) -> (pd.core.frame.DataFrame,
                                                                 pd.core.frame.DataFrame,
                                                                 pd.core.frame.DataFrame):

        
        """
        
        With this method one can perform one hot encoding of a desired feature set.
        
        
        ------------------------------------------------------------------
        
        Inputs
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame
           All features before encoding
           
        categories: list[str], default = None
           Specifies the feature subset for encoding
           
           
        ------------------------------------------------------------------   
        
        Returns
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame
           All featuers after encoding
           
        """

        
        if X_train is None or X_val is None or X_test is None:
            raise TypeError("Unsupported type for either <<X_train/val/test>>. Expected - pandas.core.frame.DataFrame.")

        if (
            not isinstance(X_train, pd.core.frame.DataFrame)
            or not isinstance(X_val, pd.core.frame.DataFrame)
            or not isinstance(X_test, pd.core.frame.DataFrame)
        ):
            raise TypeError("Unsupported type for either <<X_train/val/test>>. Expected - pandas.core.frame.DataFrame.")

        if categories is None:
            raise TypeError("Unsupported type for <<categories>>. Expected - list.")

        if not isinstance(categories, list):
            raise TypeError("Unsupported type for <<categories>>. Expected - list.")

        feature_encoder = ColumnTransformer(
            transformers=[
                (
                 "cat",
                 OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 categories
                )
            ],
            remainder="passthrough",
            n_jobs=-1
        )

        self.feature_encoder = feature_encoder

        X_train = feature_encoder.fit_transform(X_train)
        X_val = feature_encoder.transform(X_val)
        X_test = feature_encoder.transform(X_test)

        feature_names = feature_encoder.get_feature_names_out()

        feature_names_filtered = (
            [col.split("cat__")[1] for col in feature_names if "cat__" in col]
            + [col.split("remainder__")[1] for col in feature_names if "remainder__" in col]
        )

        X_train = pd.DataFrame(X_train, columns=feature_names_filtered)
        X_val = pd.DataFrame(X_val, columns=feature_names_filtered)
        X_test = pd.DataFrame(X_test, columns=feature_names_filtered)

        return X_train, X_val, X_test
  

    def variance_threshold(self, X_train: pd.core.frame.DataFrame = None,
                                 X_val: pd.core.frame.DataFrame = None,
                                 X_test: pd.core.frame.DataFrame = None,
                                 threshold: float = None) -> (pd.core.frame.DataFrame,
                                                              pd.core.frame.DataFrame,
                                                              pd.core.frame.DataFrame):

        
        """
        
        This method filters out non-informative features, based on variance threshold.
        
        
        ------------------------------------------------------------------
        
        Inputs
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame
           Initial features before filtration
           
        threshold: float, default = None
           variance threshold for filtering
           
           
        ------------------------------------------------------------------   
        
        Returns
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame
           All featuers filtration
           
        """

        
        if X_train is None or X_val is None or X_test is None:
            raise TypeError("Unsupported type for either <<X_train/val/test>>. Expected - pandas.core.frame.DataFrame.")

        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_val, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            raise TypeError("Unsupported type for either <<X_train/val/test>>. Expected - pandas.core.frame.DataFrame.")

        if threshold is None:
            raise TypeError("Unsupported type for <<threshold>>. Expected - float.")

        if not isinstance(threshold, float):
            raise TypeError("Unsupported type for <<threshold>>. Expected - float.")

        variance_threshold = VarianceThreshold(threshold=threshold)
        self.variance_threshold = variance_threshold

        features_before_filtration = X_train.columns
        
        X_train = variance_threshold.fit_transform(X_train)
        X_val = variance_threshold.transform(X_val)
        X_test = variance_threshold.transform(X_test)

        features_after_filtration = variance_threshold.get_feature_names_out()

        print(
            f"\nFiltered features with VarianceThreshold and threshold {threshold}:\n\n",
            set(features_before_filtration) - set(features_after_filtration)
        )

        X_train = pd.DataFrame(X_train, columns=features_after_filtration)
        X_val = pd.DataFrame(X_val, columns=features_after_filtration)
        X_test = pd.DataFrame(X_test, columns=features_after_filtration)

        assert len(set(X_train.columns) - set(X_val.columns) - set(X_test.columns)) == 0,\
            "Feature mismatch between splits. Check VarianceThreshold again."

        return X_train, X_val, X_test
    
    
    def run_boruta(self, X_train: pd.core.frame.DataFrame = None,
                         X_val: pd.core.frame.DataFrame = None,
                         X_test: pd.core.frame.DataFrame = None,
                         y_train: pd.core.series.Series = None, 
                         n_estimators: int = None,
                         criterion: str = None,
                         max_iter: int = None,
                         perc: (float, int) = None) -> (pd.core.frame.DataFrame,
                                                        pd.core.frame.DataFrame,
                                                        pd.core.frame.DataFrame):

        """
        
        This method uses BorutaPy to filter out features with
        insignificant feature importance. Default classifier
        is BalancedRandomForestClassifier with ressampling
        strategy set to the majority class
        
        
        ------------------------------------------------------------------
        
        Inputs
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame, default = None
           Initial features before filtration
           
        y_train: pd.core.series.Series, default = None
           
        n_estimators: int, default = None
           Number of trees for the ensemble
           
        criterion: str, default = None
           Either gini or entropy defining the split
           
        max_iter: int, default = None
           The maxium number of interations for Boruta
           
        perc: float | int, default = None
           Threshold for comparison between shadow and real features.
           The lower perc is, the more false positives will be picked
           as relevant features, but also the less relevant features
           will be left out
           
           
        ------------------------------------------------------------------   
        
        Returns
        
        *******
        
        X_train, X_val, X_test: pandas.core.frame.DataFrame
           All featuers filtration
           
        """

        if X_train is None or X_val is None or X_test is None:
            raise TypeError("Unsupported type for either <<X_train/val/test>>. Expected - pandas.core.frame.DataFrame.")

        if (
            not isinstance(X_train, pd.core.frame.DataFrame)
            or not isinstance(X_val, pd.core.frame.DataFrame)
            or not isinstance(X_test, pd.core.frame.DataFrame)
        ):
            raise TypeError("Unsupported type for either <<X_train/val/test>>. Expected - pandas.core.frame.DataFrame.")
            
        if y_train is None:
            raise TypeError("Unsupported type for <<y_train>>. Expected - pd.core.series.Series.")

        if not isinstance(y_train, pd.core.series.Series):
            raise TypeError("Unsupported type for <<y_train>>. Expected - pd.core.series.Series.")

        if n_estimators is None:
            raise TypeError("Unsupported type for <<n_estimators>>. Expected - int.")

        if not isinstance(n_estimators, int):
            raise TypeError("Unsupported type for <<n_estimators>>. Expected - int.")

        if criterion is None:
            raise TypeError("Unsupported type for <<criterion>>. Expected - str.")

        if not isinstance(criterion, str):
            raise TypeError("Unsupported type for <<criterion>>. Expected - str.")

        if max_iter is None:
            raise TypeError("Unsupported type for <<max_iter>>. Expected - int.")

        if not isinstance(max_iter, int):
            raise TypeError("Unsupported type for <<max_iter>>. Expected - int.")

        if perc is None:
            raise TypeError("Unsupported type for <<perc>>. Expected - float or int.")

        if not isinstance(perc, (int, float)):
            raise TypeError("Unsupported type for <<perc>>. Expected - float or int.")

        features_before_boruta = X_train.columns

        max_features = int(np.sqrt(len(X_train.columns)))

        if max_features % 2 == 0:
            max_features += 1

        brf = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion,
            bootstrap=True,
            oob_score=True,
            sampling_strategy="majority",
            replacement=True,
            random_state=42,
            n_jobs=-1
        )

        self.clf_boruta = brf

        boruta = BorutaPy(
            estimator=brf,
            n_estimators=n_estimators,
            perc=perc,
            max_iter=max_iter,
            random_state=42,
            verbose=2
        )

        self.boruta = boruta

        boruta.fit(X_train.values, y_train)

        selected_features_boruta = X_train.columns[boruta.support_]
        self.selected_features_boruta = selected_features_boruta

        print(
            "\nDropped features from Boruta:\n\n",
            set(features_before_boruta) - set(selected_features_boruta)
        )

        X_train = pd.DataFrame(
            boruta.transform(X_train.values),
            columns=selected_features_boruta
        )
        X_val = pd.DataFrame(
            boruta.transform(X_val.values),
            columns=selected_features_boruta
        )
        X_test = pd.DataFrame(
            boruta.transform(X_test.values),
            columns=selected_features_boruta
        )

        assert len(set(X_train.columns) - set(X_val.columns) - set(X_test.columns)) == 0,\
            "Feature mismatch between splits. Check Boruta again."

        return X_train, X_val, X_test    


    def pickle_export(self, path: str = None,
                            file = None) -> None:
                
        if path is None:
            raise TypeError("Unsupported type of <<path>>. Expected - str.")

        if file is None:
            raise TypeError("Unsupported type of <<file>>.")

        with open(path, "wb") as export_file:
            pickle.dump(file, export_file)

        return


    def pickle_import(self, path: str = None) -> object:
        
        if path is None:
            raise TypeError("Unsupported type of <<path>>. Expected - str.")

        with open(path, "rb") as import_file:
            file = pickle.load(import_file)

        return file

    