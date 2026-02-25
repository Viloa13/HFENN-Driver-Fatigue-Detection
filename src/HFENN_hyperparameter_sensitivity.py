"""
HFENN Hyperparameter Sensitivity Analysis
==========================================
Analyze the sensitivity of model performance to key hyperparameters.

Parameters analyzed:
1. Dropout Rate: [0.1, 0.2, 0.3, 0.4, 0.5]
2. Window Size: [1305, 2610, 5220] (15s, 30s, 60s at 87Hz)

Output: Heatmaps showing parameter impact on performance

Author: Auto-generated
Date: 2026-01-23
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import skew, kurtosis
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
from datetime import datetime
import random
import json

# Deep learning
try:
    from keras import layers, models, Input, regularizers
    from keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    print("Using standalone keras")
except ImportError:
    from tensorflow.keras import layers, models, Input, regularizers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    print("Using tensorflow.keras")

# Set random seeds
np.random.seed(42)
random.seed(42)

# Output directory
OUTPUT_DIR = 'figures'
DATA_DIR = 'plot_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# Feature Extractor
# =============================================================================

class QuadChannelFeatureExtractor:
    def __init__(self, sampling_rate=87):
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal):
        features = []
        features.extend([
            np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
            np.median(signal), skew(signal), kurtosis(signal), np.var(signal),
            np.ptp(signal), np.percentile(signal, 25), np.percentile(signal, 75),
            len(find_peaks(signal)[0]), len(find_peaks(-signal)[0]),
            np.sum(signal**2), np.mean(signal**2), np.sqrt(np.mean(signal**2)),
        ])
        return features
    
    def extract_frequency_domain_features(self, signal):
        features = []
        freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
        features.extend([
            np.mean(psd), np.std(psd), np.max(psd),
            freqs[np.argmax(psd)] if len(psd) > 0 else 0, np.sum(psd),
        ])
        freq_bands = [(0, 1), (1, 5), (5, 15), (15, 25)]
        total_power = np.sum(psd)
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            features.append(band_power / total_power if total_power > 0 else 0)
        return features
    
    def extract_wavelet_features(self, signal):
        features = []
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        for coeff in coeffs:
            if len(coeff) > 0:
                features.extend([np.mean(coeff), np.std(coeff), np.sum(coeff**2), np.max(np.abs(coeff))])
        return features
    
    def extract_nonlinear_features(self, signal):
        features = []
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.angle(analytic_signal)
        features.extend([
            np.mean(amplitude_envelope), np.std(amplitude_envelope),
            np.mean(np.diff(instantaneous_phase)),
        ])
        zero_crossings = np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0) / len(signal)
        features.append(zero_crossings)
        return features
    
    def extract_diff_features(self, diff_signal):
        features = []
        features.extend([
            np.mean(diff_signal), np.std(diff_signal), np.max(diff_signal), np.min(diff_signal),
            np.sum(diff_signal > 0) / len(diff_signal), np.sum(diff_signal < 0) / len(diff_signal),
            np.sum(np.abs(diff_signal)), np.mean(np.abs(diff_signal)),
            skew(diff_signal), kurtosis(diff_signal),
        ])
        return features
    
    def extract_single_channel_features(self, signal):
        all_features = []
        all_features.extend(self.extract_time_domain_features(signal))
        all_features.extend(self.extract_frequency_domain_features(signal))
        all_features.extend(self.extract_wavelet_features(signal))
        all_features.extend(self.extract_nonlinear_features(signal))
        return np.array(all_features)
    
    def extract_quad_channel_features(self, pulse, fatigue, pulse_diff, fatigue_diff):
        pulse_features = self.extract_single_channel_features(pulse)
        fatigue_features = self.extract_single_channel_features(fatigue)
        pulse_diff_features = self.extract_diff_features(pulse_diff)
        fatigue_diff_features = self.extract_diff_features(fatigue_diff)
        return np.concatenate([pulse_features, fatigue_features, pulse_diff_features, fatigue_diff_features])


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    data_dir = 'data/processed'
    all_files = glob.glob(os.path.join(data_dir, '*_continuous.csv'))
    all_files = [f for f in all_files if 'all_participants' not in f]
    
    df_list = []
    for file in all_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)
    
    df = pd.concat(df_list, ignore_index=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    
    # Per-participant standardization
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        for col in ['Pulse', 'Fatigue']:
            data = df.loc[mask, col].values
            mean_val, std_val = np.mean(data), np.std(data)
            if std_val > 0:
                df.loc[mask, col] = (data - mean_val) / std_val
            else:
                df.loc[mask, col] = 0
    
    # Differential signals
    df['Pulse_Diff'] = 0.0
    df['Fatigue_Diff'] = 0.0
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        df.loc[mask, 'Pulse_Diff'] = df.loc[mask, 'Pulse'].diff().fillna(0).values
        df.loc[mask, 'Fatigue_Diff'] = df.loc[mask, 'Fatigue'].diff().fillna(0).values
    
    return df


def extract_segments(df, window_size=2610, overlap_ratio=0.5):
    """Extract segments with specified window size"""
    feature_extractor = QuadChannelFeatureExtractor()
    step = int(window_size * (1 - overlap_ratio))
    
    pulse_segments, fatigue_segments, enhanced_features = [], [], []
    target_values, participant_ids = [], []
    
    for pid in df['Participant_ID'].unique():
        df_p = df[df['Participant_ID'] == pid].reset_index(drop=True)
        pulse_data = df_p['Pulse'].values
        fatigue_data = df_p['Fatigue'].values
        pulse_diff_data = df_p['Pulse_Diff'].values
        fatigue_diff_data = df_p['Fatigue_Diff'].values
        target_data = df_p['Fatigue_Score_Smoothed'].values
        
        for start in range(0, len(pulse_data) - window_size + 1, step):
            end = start + window_size
            pulse_segments.append(pulse_data[start:end])
            fatigue_segments.append(fatigue_data[start:end])
            features = feature_extractor.extract_quad_channel_features(
                pulse_data[start:end], fatigue_data[start:end],
                pulse_diff_data[start:end], fatigue_diff_data[start:end]
            )
            enhanced_features.append(features)
            target_values.append(target_data[end-1])
            participant_ids.append(pid)
    
    pulse_ts = np.array(pulse_segments)
    fatigue_ts = np.array(fatigue_segments)
    features = np.nan_to_num(np.array(enhanced_features), nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.array(target_values)
    pids = np.array(participant_ids)
    
    return pulse_ts, fatigue_ts, features, targets, pids


def wavelet_transform_batch(data, wavelet='db4', level=4):
    n_samples = data.shape[0]
    sample_coeffs = pywt.wavedec(data[0, :, 0], wavelet, level=level)
    coeff_lengths = [len(c) for c in sample_coeffs]
    all_coeffs = [np.zeros((n_samples, length, 1)) for length in coeff_lengths]
    
    for i in range(n_samples):
        signal = data[i, :, 0]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        for j, coeff in enumerate(coeffs):
            all_coeffs[j][i, :, 0] = coeff
    
    return all_coeffs


# =============================================================================
# Model Builder with Configurable Dropout
# =============================================================================

def attention_block(x, filters):
    attention = layers.Dense(filters, activation='tanh')(x)
    attention = layers.Dense(filters, activation='softmax')(attention)
    attended = layers.Multiply()([x, attention])
    return attended


def residual_block(x, filters, kernel_size=3, dropout_rate=0.3):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    x = attention_block(x, filters)
    return x


def build_hfenn_with_dropout(wavelet_shapes, feature_dim, dropout_rate=0.3, filters=32):
    """Build HFENN with configurable dropout rate"""
    all_inputs = []
    wavelet_outputs = []
    
    for i, shape in enumerate(wavelet_shapes):
        inp = Input(shape=shape, name=f'wavelet_{i}_input')
        all_inputs.append(inp)
        x = residual_block(inp, filters, dropout_rate=dropout_rate)
        x = layers.GlobalMaxPooling1D()(x)
        wavelet_outputs.append(x)
    
    wavelet_merged = layers.Concatenate()(wavelet_outputs)
    
    input_features = Input(shape=(feature_dim,), name='features_input')
    all_inputs.append(input_features)
    feat = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_features)
    feat = layers.BatchNormalization()(feat)
    feat = layers.Dropout(min(dropout_rate + 0.2, 0.6))(feat)  # Slightly higher for features
    feat = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(feat)
    
    fused = layers.Concatenate()([wavelet_merged, feat])
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(min(dropout_rate + 0.2, 0.6))(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=all_inputs, outputs=output, name=f'HFENN_dropout_{dropout_rate}')
    return model


# =============================================================================
# Training Function
# =============================================================================

def train_and_evaluate(model, train_inputs, y_train, test_inputs, y_test, epochs=30, batch_size=32, verbose=0):
    """Train model and return metrics"""
    try:
        from keras import ops
        def rmse(y_true, y_pred):
            return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))
    except:
        import tensorflow as tf
        def rmse(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae', rmse]
    )
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
    ]
    
    model.fit(
        train_inputs, y_train,
        validation_data=(test_inputs, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    y_pred = model.predict(test_inputs, verbose=0).flatten()
    
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
    }


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def run_dropout_sensitivity(df, dropout_rates=[0.1, 0.2, 0.3, 0.4, 0.5], window_size=2610):
    """Run sensitivity analysis for dropout rate"""
    print("\n" + "="*60)
    print("Dropout Rate Sensitivity Analysis")
    print("="*60)
    
    # Extract data once
    pulse_ts, fatigue_ts, features, targets, pids = extract_segments(df, window_size=window_size)
    
    # Feature scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    selector = SelectKBest(score_func=f_regression, k=50)
    features_selected = selector.fit_transform(features_scaled, targets)
    
    # Reshape
    X_pulse = pulse_ts.reshape(pulse_ts.shape[0], pulse_ts.shape[1], 1)
    X_fatigue = fatigue_ts.reshape(fatigue_ts.shape[0], fatigue_ts.shape[1], 1)
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_pulse, targets, groups=pids))
    
    X_pulse_train, X_pulse_test = X_pulse[train_idx], X_pulse[test_idx]
    X_fatigue_train, X_fatigue_test = X_fatigue[train_idx], X_fatigue[test_idx]
    X_feat_train, X_feat_test = features_selected[train_idx], features_selected[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]
    
    # Wavelet transform
    pulse_train_coeffs = wavelet_transform_batch(X_pulse_train)
    pulse_test_coeffs = wavelet_transform_batch(X_pulse_test)
    fatigue_train_coeffs = wavelet_transform_batch(X_fatigue_train)
    fatigue_test_coeffs = wavelet_transform_batch(X_fatigue_test)
    
    wavelet_shapes = [c.shape[1:] for c in pulse_train_coeffs] + [c.shape[1:] for c in fatigue_train_coeffs]
    
    train_inputs = pulse_train_coeffs + fatigue_train_coeffs + [X_feat_train]
    test_inputs = pulse_test_coeffs + fatigue_test_coeffs + [X_feat_test]
    
    results = []
    for dropout in dropout_rates:
        print(f"\nTraining with dropout={dropout}...")
        model = build_hfenn_with_dropout(wavelet_shapes, feature_dim=50, dropout_rate=dropout)
        metrics = train_and_evaluate(model, train_inputs, y_train, test_inputs, y_test, epochs=30, verbose=0)
        metrics['dropout'] = dropout
        results.append(metrics)
        print(f"  R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    
    return pd.DataFrame(results)


def run_window_sensitivity(df, window_sizes=[1305, 2610, 5220], dropout_rate=0.3):
    """Run sensitivity analysis for window size"""
    print("\n" + "="*60)
    print("Window Size Sensitivity Analysis")
    print("="*60)
    
    results = []
    for window_size in window_sizes:
        print(f"\nProcessing window_size={window_size} ({window_size/87:.1f}s)...")
        
        # Extract data with this window size
        pulse_ts, fatigue_ts, features, targets, pids = extract_segments(df, window_size=window_size)
        
        if len(targets) < 100:
            print(f"  Skipping: too few samples ({len(targets)})")
            continue
        
        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        selector = SelectKBest(score_func=f_regression, k=min(50, features.shape[1]))
        features_selected = selector.fit_transform(features_scaled, targets)
        
        # Reshape
        X_pulse = pulse_ts.reshape(pulse_ts.shape[0], pulse_ts.shape[1], 1)
        X_fatigue = fatigue_ts.reshape(fatigue_ts.shape[0], fatigue_ts.shape[1], 1)
        
        # Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X_pulse, targets, groups=pids))
        
        X_pulse_train, X_pulse_test = X_pulse[train_idx], X_pulse[test_idx]
        X_fatigue_train, X_fatigue_test = X_fatigue[train_idx], X_fatigue[test_idx]
        X_feat_train, X_feat_test = features_selected[train_idx], features_selected[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        # Wavelet transform
        pulse_train_coeffs = wavelet_transform_batch(X_pulse_train)
        pulse_test_coeffs = wavelet_transform_batch(X_pulse_test)
        fatigue_train_coeffs = wavelet_transform_batch(X_fatigue_train)
        fatigue_test_coeffs = wavelet_transform_batch(X_fatigue_test)
        
        wavelet_shapes = [c.shape[1:] for c in pulse_train_coeffs] + [c.shape[1:] for c in fatigue_train_coeffs]
        
        train_inputs = pulse_train_coeffs + fatigue_train_coeffs + [X_feat_train]
        test_inputs = pulse_test_coeffs + fatigue_test_coeffs + [X_feat_test]
        
        model = build_hfenn_with_dropout(wavelet_shapes, feature_dim=features_selected.shape[1], dropout_rate=dropout_rate)
        metrics = train_and_evaluate(model, train_inputs, y_train, test_inputs, y_test, epochs=30, verbose=0)
        metrics['window_size'] = window_size
        metrics['window_sec'] = window_size / 87.0
        metrics['n_samples'] = len(targets)
        results.append(metrics)
        print(f"  R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, n={len(targets)}")
    
    return pd.DataFrame(results)


def run_full_grid_search(df, dropout_rates=[0.1, 0.3, 0.5], window_sizes=[1305, 2610]):
    """Run full grid search (simplified for speed)"""
    print("\n" + "="*60)
    print("Full Grid Search: Dropout × Window Size")
    print("="*60)
    
    results = []
    
    for window_size in window_sizes:
        print(f"\n--- Window Size: {window_size} ({window_size/87:.1f}s) ---")
        
        # Extract data
        pulse_ts, fatigue_ts, features, targets, pids = extract_segments(df, window_size=window_size)
        
        if len(targets) < 100:
            continue
        
        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        selector = SelectKBest(score_func=f_regression, k=min(50, features.shape[1]))
        features_selected = selector.fit_transform(features_scaled, targets)
        
        # Reshape
        X_pulse = pulse_ts.reshape(pulse_ts.shape[0], pulse_ts.shape[1], 1)
        X_fatigue = fatigue_ts.reshape(fatigue_ts.shape[0], fatigue_ts.shape[1], 1)
        
        # Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X_pulse, targets, groups=pids))
        
        X_pulse_train, X_pulse_test = X_pulse[train_idx], X_pulse[test_idx]
        X_fatigue_train, X_fatigue_test = X_fatigue[train_idx], X_fatigue[test_idx]
        X_feat_train, X_feat_test = features_selected[train_idx], features_selected[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        # Wavelet transform
        pulse_train_coeffs = wavelet_transform_batch(X_pulse_train)
        pulse_test_coeffs = wavelet_transform_batch(X_pulse_test)
        fatigue_train_coeffs = wavelet_transform_batch(X_fatigue_train)
        fatigue_test_coeffs = wavelet_transform_batch(X_fatigue_test)
        
        wavelet_shapes = [c.shape[1:] for c in pulse_train_coeffs] + [c.shape[1:] for c in fatigue_train_coeffs]
        
        train_inputs = pulse_train_coeffs + fatigue_train_coeffs + [X_feat_train]
        test_inputs = pulse_test_coeffs + fatigue_test_coeffs + [X_feat_test]
        
        for dropout in dropout_rates:
            print(f"  Training: dropout={dropout}...", end=" ")
            model = build_hfenn_with_dropout(wavelet_shapes, feature_dim=features_selected.shape[1], dropout_rate=dropout)
            metrics = train_and_evaluate(model, train_inputs, y_train, test_inputs, y_test, epochs=25, verbose=0)
            metrics['dropout'] = dropout
            metrics['window_size'] = window_size
            results.append(metrics)
            print(f"R²={metrics['r2']:.4f}")
    
    return pd.DataFrame(results)


# =============================================================================
# Plotting
# =============================================================================

def plot_sensitivity_results(df_dropout, df_window, df_grid=None):
    """Plot sensitivity analysis results"""
    print("\n" + "="*60)
    print("Generating Figure 9: Hyperparameter Sensitivity")
    print("="*60)
    
    if df_grid is not None and len(df_grid) > 0:
        # Create heatmap from grid search
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- Subplot A: R² Heatmap ---
        ax1 = axes[0]
        
        pivot_r2 = df_grid.pivot(index='dropout', columns='window_size', values='r2')
        sns.heatmap(pivot_r2, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax1,
                    cbar_kws={'label': 'R² Score'})
        ax1.set_xlabel('Window Size (samples)')
        ax1.set_ylabel('Dropout Rate')
        ax1.set_title('(A) R² Score Sensitivity')
        
        # --- Subplot B: RMSE Heatmap ---
        ax2 = axes[1]
        
        pivot_rmse = df_grid.pivot(index='dropout', columns='window_size', values='rmse')
        sns.heatmap(pivot_rmse, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax2,
                    cbar_kws={'label': 'RMSE'})
        ax2.set_xlabel('Window Size (samples)')
        ax2.set_ylabel('Dropout Rate')
        ax2.set_title('(B) RMSE Sensitivity')
        
    else:
        # Fallback: line plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- Subplot A: Dropout Sensitivity ---
        ax1 = axes[0]
        
        if df_dropout is not None and len(df_dropout) > 0:
            ax1.plot(df_dropout['dropout'], df_dropout['r2'], 'bo-', linewidth=2, markersize=8, label='R²')
            ax1.set_xlabel('Dropout Rate')
            ax1.set_ylabel('R² Score', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(df_dropout['dropout'], df_dropout['rmse'], 'rs--', linewidth=2, markersize=8, label='RMSE')
            ax1_twin.set_ylabel('RMSE', color='r')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            
            ax1.set_title('(A) Dropout Rate Sensitivity')
            ax1.grid(True, alpha=0.3)
        
        # --- Subplot B: Window Size Sensitivity ---
        ax2 = axes[1]
        
        if df_window is not None and len(df_window) > 0:
            ax2.plot(df_window['window_sec'], df_window['r2'], 'go-', linewidth=2, markersize=8, label='R²')
            ax2.set_xlabel('Window Size (seconds)')
            ax2.set_ylabel('R² Score', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df_window['window_sec'], df_window['rmse'], 'ms--', linewidth=2, markersize=8, label='RMSE')
            ax2_twin.set_ylabel('RMSE', color='m')
            ax2_twin.tick_params(axis='y', labelcolor='m')
            
            ax2.set_title('(B) Window Size Sensitivity')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure9_Hyperparameter_Sensitivity.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure9_Hyperparameter_Sensitivity.[png/pdf/svg]")
    
    # Save data
    if df_dropout is not None:
        df_dropout.to_csv(os.path.join(DATA_DIR, 'sensitivity_dropout.csv'), index=False)
    if df_window is not None:
        df_window.to_csv(os.path.join(DATA_DIR, 'sensitivity_window.csv'), index=False)
    if df_grid is not None:
        df_grid.to_csv(os.path.join(DATA_DIR, 'sensitivity_grid.csv'), index=False)
    
    plt.close()
    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("="*70)
    print("HFENN Hyperparameter Sensitivity Analysis")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    print(f"Data loaded: {len(df)} rows")
    
    # Choose analysis mode
    print("\nSelect analysis mode:")
    print("  1. Quick (dropout only, ~10 min)")
    print("  2. Standard (dropout + window, ~30 min)")
    print("  3. Full grid search (~1 hour)")
    
    # Default to standard mode
    mode = 2
    
    df_dropout = None
    df_window = None
    df_grid = None
    
    if mode >= 1:
        # Dropout sensitivity
        df_dropout = run_dropout_sensitivity(df, dropout_rates=[0.1, 0.2, 0.3, 0.4, 0.5])
    
    if mode >= 2:
        # Window size sensitivity
        df_window = run_window_sensitivity(df, window_sizes=[1305, 2610, 5220])
    
    if mode >= 3:
        # Full grid search
        df_grid = run_full_grid_search(df, dropout_rates=[0.1, 0.3, 0.5], window_sizes=[1305, 2610, 5220])
    
    # Plot results
    plot_sensitivity_results(df_dropout, df_window, df_grid)
    
    # Summary
    print("\n" + "="*70)
    print("Sensitivity Analysis Summary")
    print("="*70)
    
    if df_dropout is not None:
        best_dropout = df_dropout.loc[df_dropout['r2'].idxmax()]
        print(f"\nBest Dropout Rate: {best_dropout['dropout']}")
        print(f"  R²={best_dropout['r2']:.4f}, RMSE={best_dropout['rmse']:.4f}")
    
    if df_window is not None:
        best_window = df_window.loc[df_window['r2'].idxmax()]
        print(f"\nBest Window Size: {best_window['window_size']} ({best_window['window_sec']:.1f}s)")
        print(f"  R²={best_window['r2']:.4f}, RMSE={best_window['rmse']:.4f}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
