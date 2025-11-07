import pandas as pd
import numpy as np
import re
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input  
from tensorflow.keras.callbacks import EarlyStopping


# MLFLOW EXPERIMENT SETUP


mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "Predictive Maintenance (All Models)"
mlflow.set_experiment(experiment_name)

def get_experiment_id(name):
    """Finds or creates an MLflow experiment."""
    try:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            print(f"Creating new experiment: {name}")
            return mlflow.create_experiment(name)
        return experiment.experiment_id
    except Exception as e:
        print(f"Error getting or creating experiment: {e}")
        return mlflow.create_experiment(name)

# PIPELINE 1: AI4I (TABULAR) DATA PREPARATION

def load_and_prep_ai4i_data():
    """Loads and prepares the AI4I 2020 dataset."""
    print("--- Loading and preparing AI4I 2020 data ---")
    FILE_PATH = r'ai4i2020.csv'
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found.")
        return None
        
    df = pd.read_csv(FILE_PATH)
    df.columns = df.columns.str.strip() 

    y = df['Machine failure']
    X = df.drop(['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    
    clean_cols = []
    for col in X.columns:
        new_col = re.sub(r'\[.*?\]', '', col).strip().replace(' ', '_')
        clean_cols.append(new_col)
    X.columns = clean_cols
    
    X = pd.get_dummies(X, columns=['Type'], drop_first=True)
    model_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    joblib.dump(model_columns, 'ai4i_model_columns.joblib')
    return X_train, X_test, y_train, y_test, model_columns


# PIPELINE 2: NASA (TIME-SERIES) DATA PREPARATION
def create_sequences(df, sequence_length, feature_cols, target_col):
    """Helper function to create 3D sequences."""
    X_sequences, y_sequences = [], []
    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]
        features = unit_df[feature_cols].values
        target = unit_df[target_col].values
        
        for i in range(len(features) - sequence_length):
            X_sequences.append(features[i:i + sequence_length])
            y_sequences.append(target[i + sequence_length])
            
    return np.array(X_sequences), np.array(y_sequences)

def load_and_prep_nasa_data(sequence_length=50, warning_window=30):
    """Loads, preps, and windows the NASA Turbofan dataset."""
    print("\n--- Loading and preparing NASA Turbofan data ---")
    FILE_NAME = 'train_FD001.txt'
    if not os.path.exists(FILE_NAME):
        print(f"Error: {FILE_NAME} not found.")
        return None

    col_names = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    df_train = pd.read_csv(FILE_NAME, sep=' ', header=None, names=col_names, index_col=False)
    df_train = df_train.dropna(axis=1, how='all')

    max_cycles = df_train.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df_train = pd.merge(df_train, max_cycles, on='unit_number', how='left')
    df_train['RUL'] = df_train['max_cycles'] - df_train['time_cycles']
    df_train['will_fail_soon'] = (df_train['RUL'] <= warning_window).astype(int)
    
    constant_cols = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    setting_cols = ['setting_1', 'setting_2', 'setting_3']
    drop_cols = constant_cols + setting_cols + ['unit_number', 'time_cycles', 'RUL', 'will_fail_soon', 'max_cycles']
    feature_cols = [col for col in df_train.columns if col not in drop_cols]
    
    scaler = MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])

    print(f"--- Creating sequences of length {sequence_length} ---")
    X_lstm, y_lstm = create_sequences(
        df_train, sequence_length, feature_cols, 'will_fail_soon'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm
    )
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    total_train = len(y_train)
    weight_for_0 = (1 / counts_train[0]) * (total_train / 2.0)
    weight_for_1 = (1 / counts_train[1]) * (total_train / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print(f"LSTM/GRU data ready. X_train shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test, class_weight, X_train.shape[1], X_train.shape[2]


if __name__ == "__main__":
    
    experiment_id = get_experiment_id(experiment_name)
    
    tabular_data = load_and_prep_ai4i_data()
    
    if tabular_data:
        X_train, X_test, y_train, y_test, tab_cols = tabular_data
        
        # Random Forest
        print("\n--- Starting Run: Random Forest (AI4I Data) ---")
        with mlflow.start_run(run_name="RandomForest (Baseline)", experiment_id=experiment_id):
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("dataset", "AI4I_2020")
            model_rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
            model_rf.fit(X_train, y_train)
            y_pred_rf = model_rf.predict(X_test)
            report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
            rf_recall = report_rf['1']['recall']
            mlflow.log_metric("recall_class_1", rf_recall)
            mlflow.log_metric("precision_class_1", report_rf['1']['precision'])
            mlflow.sklearn.log_model(model_rf, "model")
            mlflow.log_artifact('ai4i_model_columns.joblib')
            print(f"Random Forest Recall: {rf_recall:.4f}")

        # XGBoost
        print("\n--- Starting Run: XGBoost (AI4I Data) ---")
        with mlflow.start_run(run_name="XGBoost (Challenger)", experiment_id=experiment_id):
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("dataset", "AI4I_2020")
            scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            model_xgb = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_estimators=100)
            model_xgb.fit(X_train, y_train)
            y_pred_xgb = model_xgb.predict(X_test)
            report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
            xgb_recall = report_xgb['1']['recall']
            mlflow.log_metric("recall_class_1", xgb_recall)
            mlflow.log_metric("precision_class_1", report_xgb['1']['precision'])
            mlflow.sklearn.log_model(model_xgb, "model")
            mlflow.log_artifact('ai4i_model_columns.joblib')
            print(f"XGBoost Recall: {xgb_recall:.4f}")

    # --- PIPELINE 2: Run Time-Series Models ---
    ts_data = load_and_prep_nasa_data()
    
    if ts_data:
        X_train_ts, X_test_ts, y_train_ts, y_test_ts, class_weight, n_timesteps, n_features = ts_data
        
        #LSTM
        print("\n--- Starting Run: LSTM (NASA Data) ---")
        with mlflow.start_run(run_name="LSTM (Time-Series)", experiment_id=experiment_id):
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("dataset", "NASA_Turbofan_FD001")
            mlflow.log_param("sequence_length", n_timesteps)
            mlflow.log_param("warning_window", 30)
            
            model_lstm = Sequential()
            model_lstm.add(Input(shape=(n_timesteps, n_features)))
            model_lstm.add(LSTM(units=50, return_sequences=True))
            model_lstm.add(Dropout(0.2))
            model_lstm.add(LSTM(units=25))
            model_lstm.add(Dropout(0.2))
            model_lstm.add(Dense(units=1, activation='sigmoid'))
            model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall'])
            
            es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
            model_lstm.fit(
                X_train_ts, y_train_ts, epochs=10, batch_size=64,
                validation_data=(X_test_ts, y_test_ts),
                class_weight=class_weight, callbacks=[es], verbose=1
            )
            
            y_pred_proba = model_lstm.predict(X_test_ts)
            y_pred = (y_pred_proba > 0.5).astype(int)
            report_lstm = classification_report(y_test_ts, y_pred, output_dict=True)
            lstm_recall = report_lstm['1']['recall']

            mlflow.log_metric("recall_class_1", lstm_recall)
            mlflow.log_metric("precision_class_1", report_lstm['1']['precision'])
            mlflow.tensorflow.log_model(model_lstm, "model")
            print(f"LSTM Recall: {lstm_recall:.4f}")

        # GRU - Gated Recurrent Unit
        print("\n--- Starting Run: GRU (NASA Data) ---")
        with mlflow.start_run(run_name="GRU (Time-Series)", experiment_id=experiment_id):
            mlflow.log_param("model_type", "GRU")
            mlflow.log_param("dataset", "NASA_Turbofan_FD001")
            mlflow.log_param("sequence_length", n_timesteps)
            mlflow.log_param("warning_window", 30)
            
            # 1. Build Model (Note the GRU layers)
            model_gru = Sequential()
            model_gru.add(Input(shape=(n_timesteps, n_features)))
            model_gru.add(GRU(units=50, return_sequences=True)) # <-- GRU Layer
            model_gru.add(Dropout(0.2))
            model_gru.add(GRU(units=25)) # <-- GRU Layer
            model_gru.add(Dropout(0.2))
            model_gru.add(Dense(units=1, activation='sigmoid'))
            
            model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall'])
            
            # 2. Train Model (Identical data to LSTM)
            es_gru = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
            model_gru.fit(
                X_train_ts, y_train_ts, epochs=10, batch_size=64,
                validation_data=(X_test_ts, y_test_ts),
                class_weight=class_weight, callbacks=[es_gru], verbose=1
            )
            
            # 3. Evaluate
            y_pred_proba_gru = model_gru.predict(X_test_ts)
            y_pred_gru = (y_pred_proba_gru > 0.5).astype(int)
            report_gru = classification_report(y_test_ts, y_pred_gru, output_dict=True)
            gru_recall = report_gru['1']['recall']

            # 4. Log to MLflow
            mlflow.log_metric("recall_class_1", gru_recall)
            mlflow.log_metric("precision_class_1", report_gru['1']['precision'])
            mlflow.tensorflow.log_model(model_gru, "model")
            print(f"GRU Recall: {gru_recall:.4f}")

    print("\n--- ALL TRAINING RUNS COMPLETE ---")