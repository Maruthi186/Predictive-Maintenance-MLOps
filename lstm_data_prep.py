import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os # Good to check for the file

# Make sure to run 'pip install tensorflow' if you haven't
try:
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("TensorFlow not found. Please run 'pip install tensorflow'")
    exit()

print("--- Starting LSTM Data Prep Script ---")

# --- 1. Define the column names for the dataset ---
col_names = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
col_names += [f's_{i}' for i in range(1, 22)] # s_1 to s_21 for sensor readings

# --- 2. Load the data ---
FILE_NAME = 'train_FD001.txt'
if not os.path.exists(FILE_NAME):
    print(f"Error: '{FILE_NAME}' not found.")
    print("Please download it from Kaggle and place it in this folder.")
    exit()

df_train = pd.read_csv(
    FILE_NAME, 
    sep=' ', 
    header=None, 
    names=col_names,
    index_col=False 
)
print("--- Data Loaded Successfully ---")
df_train = df_train.dropna(axis=1, how='all')

print("\n--- First 5 Rows: ---")
print(df_train.head())

# --- 3. Calculate RUL (Remaining Useful Life) ---
max_cycles = df_train.groupby('unit_number')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycles']
df_train = pd.merge(df_train, max_cycles, on='unit_number', how='left')
df_train['RUL'] = df_train['max_cycles'] - df_train['time_cycles']
df_train = df_train.drop('max_cycles', axis=1)

print("\n--- Data with RUL calculated (see last column): ---")
print(df_train[df_train['unit_number'] == 1].tail())

# --- 4. Create Our Binary Classification Target ---
window_size = 30
df_train['will_fail_soon'] = (df_train['RUL'] <= window_size).astype(int)

print("\n--- Data with new target 'will_fail_soon': ---")
print(df_train[df_train['unit_number'] == 1].iloc[160:168])


# --- 5. PRE-PROCESSING: Normalize the Data ---
constant_cols = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
setting_cols = ['setting_1', 'setting_2', 'setting_3']
drop_cols = constant_cols + setting_cols + ['unit_number', 'time_cycles', 'RUL', 'will_fail_soon']
feature_cols = [col for col in df_train.columns if col not in drop_cols]

print(f"\n--- Using {len(feature_cols)} sensor features: ---")
print(feature_cols)

scaler = MinMaxScaler()
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])

# --- 6. THE WINDOWING FUNCTION ---
def create_sequences(df, sequence_length, feature_cols, target_col):
    X_sequences = []
    y_sequences = []

    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]
        features = unit_df[feature_cols].values
        target = unit_df[target_col].values
        
        for i in range(len(features) - sequence_length):
            X_sequences.append(features[i:i + sequence_length])
            y_sequences.append(target[i + sequence_length])
            
    return np.array(X_sequences), np.array(y_sequences)

# --- 7. RUN THE FUNCTION ---
sequence_length = 50 
print(f"\n--- Creating sequences of length {sequence_length} ---")
X_lstm, y_lstm = create_sequences(
    df_train,
    sequence_length,
    feature_cols,
    'will_fail_soon'
)

# --- 8. CHECK OUR FINAL DATA SHAPE ---
print("\n--- Final Data Shapes ---")
print(f"X (features) shape: {X_lstm.shape}")
print(f"y (target) shape: {y_lstm.shape}")

print("\n--- Target Distribution ---")
unique, counts = np.unique(y_lstm, return_counts=True)
print(dict(zip(unique, counts)))
print("\n--- LSTM Data Prep Script Finished ---")

# --- 9. SPLIT THE DATA ---
# (This code goes at the bottom of your lstm_data_prep.py)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# We need TensorFlow to build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# We split our "movies" (X_lstm) and "labels" (y_lstm)
# We stratify by y_lstm to ensure both train/test sets get a
# fair mix of "failure" and "normal" samples.
X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_lstm, 
    test_size=0.2, 
    random_state=42,
    stratify=y_lstm
)

print("\n--- Splitting Data ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# --- 10. HANDLE CLASS IMBALANCE ---
# We calculate class weights to tell the model to "pay more attention"
# to the rare "failure" class (1).
unique_train, counts_train = np.unique(y_train, return_counts=True)
total_train = len(y_train)

weight_for_0 = (1 / counts_train[0]) * (total_train / 2.0)
weight_for_1 = (1 / counts_train[1]) * (total_train / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print("\n--- Class Weights ---")
print(f"Weight for class 0 (Normal): {weight_for_0:.2f}")
print(f"Weight for class 1 (Failure): {weight_for_1:.2f}")


# ... (all the code from steps 1-10 is the same) ...

# --- 11. BUILD THE LSTM MODEL (Polished Version) ---
from tensorflow.keras.layers import Input # <-- NEW: Import Input

# ...
n_timesteps = X_train.shape[1]  # 50
n_features = X_train.shape[2]   # 14

model = Sequential()
# --- THIS IS THE "MODERN" WAY THAT SILENCES THE WARNING ---
model.add(Input(shape=(n_timesteps, n_features))) # <-- CHANGED
model.add(LSTM(units=50, return_sequences=True)) # <-- CHANGED
# ---
model.add(Dropout(0.2))
model.add(LSTM(units=25))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['Recall'] # <-- We'll keep the Capital 'R'
)
model.summary()

# --- 12. TRAIN THE MODEL ---
# ... (this part is identical) ...
history = model.fit(...) 
print("--- Model Training Complete ---")

# --- 13. EVALUATE THE LSTM MODEL ---
# ... (this part is identical) ...
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# --- THIS IS THE PLOTTING FIX ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.subplot(1, 2, 2)
# FIX: Use 'Recall' (capital R) to match what we compiled
plt.plot(history.history['Recall'], label='Train Recall')     # <-- FIXED
plt.plot(history.history['val_Recall'], label='Validation Recall') # <-- FIXED
plt.title('Recall Over Epochs')
plt.legend()
plt.tight_layout()
plt.show()

# --- 12. TRAIN THE MODEL ---
print("\n--- Starting Model Training ---")
# We'll run for 10 "epochs" (passes over the data)
# We use EarlyStopping to make it stop if the validation loss
# doesn't improve, saving us time.
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)

history = model.fit(
    X_train, 
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weight, # Here is where we use our weights
    callbacks=[es]
)

print("--- Model Training Complete ---")

# --- 13. EVALUATE THE LSTM MODEL ---
print("\n--- Final Model Evaluation on Test Data ---")
# Get the model's final predictions
# .predict() gives probabilities (e.g., 0.83),
# so we round them to 0 or 1.
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print the final report
print("\n--- Classification Report (LSTM) ---")
print(classification_report(y_test, y_pred))

# Plot the training history to see if it learned well
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Plot Recall
plt.subplot(1, 2, 2)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()