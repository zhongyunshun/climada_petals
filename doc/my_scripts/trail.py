import os
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def group_into_fire_events(df):
    """
    This function takes in a DataFrame and returns a copy of the DataFrame with three additional columns:
    'date_cluster', 'regional_cluster', and 'event_id'.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        df (pd.DataFrame): The output DataFrame with three additional columns: 'date_cluster', 'regional_cluster', and 'event_id'.
    """

    df = df.copy()
    df.rename(columns={'latitude_left': 'latitude', 'longitude_left': 'longitude'}, inplace=True)

    # Convert 'acq_date' to datetime
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by='date')

    # Create 'date_cluster' column based on consecutive dates
    df['date_cluster'] = (df['date'].diff().dt.days > 1).cumsum()

    # Function to apply DBSCAN on each consecutive date cluster
    def apply_dbscan(group):
        coords = group[['latitude', 'longitude']].values
        # Apply DBSCAN, hyperparameters are the same as Climada
        db = DBSCAN(eps=15 / 111.12, min_samples=1).fit(coords)
        group['regional_cluster'] = db.labels_
        return group

    # Apply DBSCAN on each date cluster
    df = df.groupby('date_cluster').apply(apply_dbscan)

    df.reset_index(drop=True, inplace=True)

    # Create a unique 'event_id' for each unique combination of 'date_cluster' and 'regional_cluster'
    df['event_id'] = df.groupby(['date_cluster', 'regional_cluster']).ngroup()

    df = df.sort_values(by='event_id').reset_index(drop=True)

    return df


def split_event_ids(event_id_pairs, test_size, random_seed):
    """
    Splits event_id pairs into training and test sets based on the specified test size.

    Args:
        event_id_pairs (list of tuples): List of event_id and their corresponding row counts.
        test_size (float): The proportion of the test set.
        random_seed (int): The seed for the random number generator.

    Returns:
        train_event_ids (list): List of event_ids for the training set.
        test_event_ids (list): List of event_ids for the test set.
    """
    np.random.seed(random_seed)
    np.random.shuffle(event_id_pairs)

    test_event_ids = []
    current_test_rows = 0
    total_rows = sum(count for _, count in event_id_pairs)
    test_rows_target = int(total_rows * test_size)

    for event_id, count in event_id_pairs:
        current_test_rows += count
        test_event_ids.append(event_id)
        if current_test_rows >= test_rows_target:
            break

    train_event_ids = [event_id for event_id, _ in event_id_pairs if event_id not in test_event_ids]

    return train_event_ids, test_event_ids


def shuffle_and_split(df, test_size=0.1, random_seed=42):
    """
    This function shuffles the DataFrame with respect to 'event_id' and then splits the data
    into training and test sets with approximately 10% test size with respect to 'event_id'.

    Args:
        df (pd.DataFrame): The input DataFrame
        test_size (float): The proportion of the test set
        random_seed (int): The seed for the random number generator

    Returns:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels
    """
    # Shuffle the dataframe based on event_id
    event_counts = df['event_id'].value_counts().sort_index()
    event_id_pairs = list(zip(event_counts.index, event_counts.values))

    train_event_ids, test_event_ids = split_event_ids(event_id_pairs, test_size, random_seed)

    X_train = df[df['event_id'].isin(train_event_ids)].drop(columns=['ignited'])
    y_train = df[df['event_id'].isin(train_event_ids)]['ignited']
    X_test = df[df['event_id'].isin(test_event_ids)].drop(columns=['ignited'])
    y_test = df[df['event_id'].isin(test_event_ids)]['ignited']

    return X_train, X_test, y_train, y_test


# Example usage:
# folder = '../../climada_petals/data/wildfire/outputs/'
# year = 2013
# file_path = os.path.join(folder, str(year), f'ignited_eu_{year}_gdf')
#
# # Load the DataFrame from the file
# df = gpd.read_file(file_path)
# X_train, X_test, y_train, y_test = shuffle_and_split(df)


# def custom_cv_split(X, y, n_splits=5, split_by_col='event_id', random_state=42):
#     event_counts = X['event_id'].value_counts().sort_index()
#     event_id_pairs = list(zip(event_counts.index, event_counts.values))
#
#     np.random.seed(random_state)
#     np.random.shuffle(event_id_pairs)
#
#     val_event_ids = []
#     current_test_rows = 0
#     val_size = 1 / n_splits
#     total_rows = sum(count for _, count in event_id_pairs)
#     val_rows_target = int(total_rows * val_size)
#
#     for event_id, count in event_id_pairs:
#         current_test_rows += count
#         val_event_ids.append(event_id)
#         if current_test_rows >= val_rows_target:
#             break
#
#     train_event_ids = [event_id for event_id, _ in event_id_pairs if event_id not in val_event_ids]
#
#     train_indices = X[X['event_id'].isin(train_event_ids)].index
#     val_indices = X[X['event_id'].isin(val_event_ids)].index
#
#     yield train_indices, val_indices
def custom_cv_split(X, y, n_splits=5, split_by_col='event_id', random_state=42):
    """
    This function creates cross-validation splits based on a specified column (e.g., 'event_id'),
    ensuring that all rows with the same value in the specified column are kept together in the same fold.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        n_splits (int): Number of folds
        split_by_col (str): Column name to split by (e.g., 'event_id')
        random_state (int): Random seed for reproducibility

    Returns:
        splits (list): List of tuples containing train and validation indices for each fold
    """
    # Ensure reproducibility
    np.random.seed(random_state)

    # Get unique event IDs and shuffle them
    unique_event_ids = X[split_by_col].unique()
    np.random.shuffle(unique_event_ids)

    # Calculate the number of rows in each fold
    fold_sizes = np.full(n_splits, X.shape[0] // n_splits, dtype=int)
    fold_sizes[:X.shape[0] % n_splits] += 1

    splits = []
    current = 0
    for fold_size in fold_sizes:
        val_event_ids = []
        current_fold_size = 0
        # Accumulate event IDs until the fold size limit is reached
        while current_fold_size < fold_size and current < len(unique_event_ids):
            event_id = unique_event_ids[current]
            event_size = len(X[X[split_by_col] == event_id])
            # Break if adding this event would exceed the fold size
            if current_fold_size + event_size > fold_size:
                break
            val_event_ids.append(event_id)
            current_fold_size += event_size
            current += 1

        # Get training event IDs by excluding validation event IDs
        train_event_ids = np.setdiff1d(unique_event_ids, val_event_ids)

        # Get indices for training and validation sets
        train_indices = X[X[split_by_col].isin(train_event_ids)].index
        val_indices = X[X[split_by_col].isin(val_event_ids)].index

        # Append the indices for the current fold to the list
        splits.append((train_indices, val_indices))

    return splits


def train_and_evaluate_models(df, random_state=42):
    # Preprocess the dataframe
    df = pd.get_dummies(df, columns=['land_cover'], prefix='land_cover')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.drop(columns=['month', 'date', 'distance_km', 'confidence', 'geometry'])
    df['fwi'].fillna(0, inplace=True)

    # Split the data
    X_train_val, X_test, y_train_val, y_test = shuffle_and_split(df, test_size=0.1, random_seed=random_state)
    X_train_val = X_train_val.drop(columns=['latitude', 'longitude', 'brightness', 'bright_t31'])
    X_test = X_test.drop(columns=['latitude', 'longitude', 'brightness', 'bright_t31'])

    # Initialize dictionaries to store the best models, hyperparameters, and scores
    classifier_names = ["LogisticRegression", "XGBClassifier"]
    best_models = {name: None for name in classifier_names}
    best_params = {name: None for name in classifier_names}
    best_scores = {name: 0 for name in classifier_names}

    # Define objective functions for Optuna
    def logisticregression_objective(trial):
        classifier_name = "LogisticRegression"
        logistic_c = trial.suggest_float('logistic_c', 1e-5, 1e5, log=True)
        classifier_obj = LogisticRegression(C=logistic_c, random_state=random_state)
        model_pipeline = make_pipeline(StandardScaler(), classifier_obj)
        scores = cross_val_score(model_pipeline, X_train_val, y_train_val, cv=list(custom_cv_split(X_train_val, y_train_val, n_splits=5, split_by_col='event_id', random_state=random_state)), n_jobs=-1, scoring='roc_auc')
        score = scores.mean()
        if score > best_scores[classifier_name]:
            best_scores[classifier_name] = score
            best_params[classifier_name] = trial.params
            best_models[classifier_name] = classifier_obj # classifier_obj remain untrained
        return score

    def xgboost_objective(trial):
        classifier_name = "XGBClassifier"
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': random_state
        }
        classifier_obj = XGBClassifier(**xgb_params)
        scores = cross_val_score(classifier_obj, X_train_val, y_train_val, cv=list(custom_cv_split(X_train_val, y_train_val, n_splits=5, split_by_col='event_id', random_state=random_state)), n_jobs=-1, scoring='roc_auc')
        score = scores.mean()
        if score > best_scores[classifier_name]:
            best_scores[classifier_name] = score
            best_params[classifier_name] = trial.params
            best_models[classifier_name] = classifier_obj # classifier_obj remain untrained
        return score

    # Optimize hyperparameters with Optuna
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    study.optimize(logisticregression_objective, n_trials=20)
    study.optimize(xgboost_objective, n_trials=100)

    results = {}
    for name in classifier_names:
        if name == "LogisticRegression":
            best_models[name].set_params(max_iter=200)
            model_pipeline = make_pipeline(StandardScaler(), best_models[name])
        else:
            model_pipeline = best_models[name]

        model_pipeline.fit(X_train_val, y_train_val)
        train_accuracy = model_pipeline.score(X_train_val, y_train_val)
        test_accuracy = model_pipeline.score(X_test, y_test)
        y_pred = model_pipeline.predict(X_test)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)

        results[name] = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": classification_rep,
            "confusion_matrix": confusion_mat
        }

        if name == "LogisticRegression":
            feature_importance = best_models[name].coef_[0]
        else:
            feature_importance = best_models[name].feature_importances_

        feature_importance_df = pd.DataFrame({'feature': X_train_val.columns, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        results[name]["feature_importance"] = feature_importance_df

    return results


import os

folder = '../../climada_petals/data/wildfire/outputs/'
years = np.arange(2013, 2014)

# Initialize an empty DataFrame to store the concatenated DataFrames
gdf_all_years = pd.DataFrame()

for year in years:
    '''Step1: Load gdf'''
    # construct the file path
    file_path = os.path.join(folder, str(year), f'ignited_eu_{year}_gdf')

    # Load the DataFrame from the file
    gdf = gpd.read_file(file_path)

    # Concatenate the loaded DataFrame with the initial DataFrame
    gdf_all_years = pd.concat([gdf_all_years, gdf])

    gdf_all_years = gpd.GeoDataFrame(gdf_all_years, geometry=gdf_all_years.geometry, crs=gdf_all_years.crs)

    '''Step2: Group gdf into fire events'''
    gdf_all_years = group_into_fire_events(gdf_all_years)

    '''Step3: Split data into training and testing sets and train and evaluate models'''
    results = train_and_evaluate_models(gdf_all_years)

    break