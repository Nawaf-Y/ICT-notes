
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv('BTCUSDT_Binance_1h.csv')
df['Datetime'] = pd.to_datetime(df['Open time'])
df.set_index('Datetime', inplace=True)

# Feature Engineering
def add_features(df):
    # Calculate reward-to-risk ratio
    df['potential_reward'] = df['High'] - df['Close']
    df['potential_risk'] = df['Close'] - df['Low']
    df['reward_to_risk'] = (df['potential_reward'] / df['potential_risk']).where(df['potential_risk'] > 0, 0)

    # Market Maker Traps
    df['consolidation_high'] = df['High'].rolling(window=20).max()
    df['consolidation_low'] = df['Low'].rolling(window=20).min()
    df['false_breakout_above'] = (df['Close'] > df['consolidation_high']) & (df['Close'].shift(1) <= df['consolidation_high'])
    df['false_breakout_below'] = (df['Close'] < df['consolidation_low']) & (df['Close'].shift(1) >= df['consolidation_low'])

    # Convert boolean to integer
    df['false_breakout_above'] = df['false_breakout_above'].astype(int)
    df['false_breakout_below'] = df['false_breakout_below'].astype(int)

    # Mean Threshold and Liquidity
    df['mean_threshold'] = (df['High'] + df['Low']) / 2
    df['liquidity_sweep'] = ((df['Close'] > df['High'].shift(1)) | (df['Close'] < df['Low'].shift(1))).astype(int)

    # Combine Signals into 'trade_signal'
    df['trade_signal'] = 0
    df.loc[df['false_breakout_above'] == 1, 'trade_signal'] = -1
    df.loc[df['false_breakout_below'] == 1, 'trade_signal'] = 1

    # Dynamic Risk Management
    df['consecutive_losses'] = (df['trade_signal'] < 0).astype(int).rolling(window=3).sum()
    df['adjusted_risk'] = 1.0 / (df['consecutive_losses'] + 1)

    return df

# Apply Feature Engineering and Drop NaNs
df = add_features(df)
df = df.dropna()

# Debugging Output
print("Features after processing:")
print(df[['trade_signal', 'reward_to_risk', 'false_breakout_above', 'false_breakout_below']].head())
print("Data types of features:")
print(df.dtypes)

# Prepare Features and Labels
features = ['reward_to_risk', 'false_breakout_above', 'false_breakout_below', 
            'mean_threshold', 'liquidity_sweep', 'adjusted_risk']
X = df[features]
y = np.where(df['reward_to_risk'] > 3, 1, -1)

# Ensure Features are Numeric and Prepare LSTM Sequences
sequence_length = 60
X_lstm, y_lstm = [], []

for i in range(sequence_length, len(X)):
    X_lstm.append(X.iloc[i-sequence_length:i].values)
    y_lstm.append(y[i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Debugging Output for LSTM Inputs
print("Shape of X_lstm:", X_lstm.shape)
print("Data type of X_lstm:", X_lstm.dtype)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42, shuffle=False)

# Build and Train LSTM
sequence_input = Input(shape=(sequence_length, len(features)))
lstm_output = LSTM(50, return_sequences=True)(sequence_input)
attention_output = Attention()([lstm_output, lstm_output])
pooled_output = GlobalAveragePooling1D()(attention_output)
dense_output = Dense(50, activation='relu')(pooled_output)
dropout_output = Dropout(0.2)(dense_output)
final_output = Dense(1, activation='sigmoid')(dropout_output)

model = Model(inputs=sequence_input, outputs=final_output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], verbose=1)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Classification Report
print(classification_report(y_test, y_pred_binary))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Hold', 'Buy'], yticklabels=['Sell', 'Hold', 'Buy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
