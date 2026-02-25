"""
HFENN Ablation Study (v2)
=========================
基于 HFENN_regression_random_split.ipynb 的消融实验代码
使用 Random Split 数据划分方式

消融实验设计:
1. Full HFENN: 完整模型（双通道小波 + 注意力 + 手工特征）
2. No Wavelet: 移除小波分支，仅使用手工特征
3. No Attention: 移除注意力机制
4. No Manual Features: 移除手工特征分支
5. Single Channel (Pulse Only): 仅使用Pulse通道
6. Single Channel (Fatigue Only): 仅使用Fatigue通道

输出:
- Figure 6: 消融实验结果对比图
- 格式: PNG, PDF, SVG
- 数据: CSV

Author: Auto-generated based on HFENN_regression_random_split.ipynb
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
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
from datetime import datetime
import random

# Deep learning
try:
    from keras import layers, models, Input, regularizers
    from keras.optimizers import Adam
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    import keras.backend as K
    print("Using standalone keras")
except ImportError:
    from tensorflow.keras import layers, models, Input, regularizers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    import tensorflow.keras.backend as K
    print("Using tensorflow.keras")

# Set random seeds (same as notebook)
np.random.seed(42)
random.seed(42)

# Set style for publication
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Output directory
OUTPUT_DIR = 'figures'
DATA_DIR = 'plot_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =============================================================================
# Data Loading and Preprocessing (EXACTLY same as HFENN_regression_random_split.ipynb)
# =============================================================================

class QuadChannelFeatureExtractor:
    """四通道特征提取器 - 与notebook完全一致"""
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
        features.extend([np.mean(psd), np.std(psd), np.max(psd),
                        freqs[np.argmax(psd)] if len(psd) > 0 else 0, np.sum(psd)])
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
        features.extend([np.mean(amplitude_envelope), np.std(amplitude_envelope),
                        np.mean(np.diff(instantaneous_phase))])
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


def load_and_preprocess_data():
    """加载数据 + 按参与者标准化 + 差分信号 (与notebook完全一致)"""
    print("正在加载连续疲劳指数数据...")
    
    data_dir = 'data/processed'
    all_files = glob.glob(os.path.join(data_dir, '*_continuous.csv'))
    
    df_list = []
    for file in all_files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)
        print(f"  加载: {os.path.basename(file)} - {len(df_temp)} 行")
    
    df = pd.concat(df_list, ignore_index=True)
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    
    print(f"\n数据加载完成: {df.shape}")
    print(f"  参与者数量: {df['Participant_ID'].nunique()}")
    
    # === 按参与者Z-Score标准化 ===
    print("\n执行按参与者Z-Score标准化...")
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        for col in ['Pulse', 'Fatigue']:
            data = df.loc[mask, col].values
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0:
                df.loc[mask, col] = (data - mean_val) / std_val
            else:
                df.loc[mask, col] = 0
    
    # === 增加差分信号 ===
    print("生成差分信号...")
    df['Pulse_Diff'] = 0.0
    df['Fatigue_Diff'] = 0.0
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        df.loc[mask, 'Pulse_Diff'] = df.loc[mask, 'Pulse'].diff().fillna(0).values
        df.loc[mask, 'Fatigue_Diff'] = df.loc[mask, 'Fatigue'].diff().fillna(0).values
    
    return df


def extract_regression_segments(df, window_size=2610, overlap_ratio=0.5):
    """数据分段和特征提取 (与notebook完全一致)"""
    print("开始数据分段和特征提取...")
    
    feature_extractor = QuadChannelFeatureExtractor()
    step = int(window_size * (1 - overlap_ratio))
    
    pulse_segments, fatigue_segments, enhanced_features = [], [], []
    target_values, participant_ids, segment_indices = [], [], []
    
    for pid in df['Participant_ID'].unique():
        df_p = df[df['Participant_ID'] == pid].reset_index(drop=True)
        pulse_data = df_p['Pulse'].values
        fatigue_data = df_p['Fatigue'].values
        pulse_diff_data = df_p['Pulse_Diff'].values
        fatigue_diff_data = df_p['Fatigue_Diff'].values
        target_data = df_p['Fatigue_Score_Smoothed'].values
        
        seg_idx = 0
        for start in range(0, len(pulse_data) - window_size + 1, step):
            end = start + window_size
            pulse_seg = pulse_data[start:end]
            fatigue_seg = fatigue_data[start:end]
            pulse_diff_seg = pulse_diff_data[start:end]
            fatigue_diff_seg = fatigue_diff_data[start:end]
            target_seg = target_data[start:end]
            
            pulse_segments.append(pulse_seg)
            fatigue_segments.append(fatigue_seg)
            features = feature_extractor.extract_quad_channel_features(
                pulse_seg, fatigue_seg, pulse_diff_seg, fatigue_diff_seg)
            enhanced_features.append(features)
            target_values.append(target_seg[-1])
            participant_ids.append(pid)
            segment_indices.append(seg_idx)
            seg_idx += 1
    
    pulse_ts = np.array(pulse_segments)
    fatigue_ts = np.array(fatigue_segments)
    features = np.array(enhanced_features)
    targets = np.array(target_values)
    pids = np.array(participant_ids)
    seg_indices = np.array(segment_indices)
    
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"数据分段完成: {len(targets)} 样本, {features.shape[1]} 特征")
    
    # 特征平滑
    def apply_feature_smoothing(features, pids, seg_indices, window_size=3):
        features_smoothed = features.copy()
        for pid in np.unique(pids):
            pid_mask = pids == pid
            pid_indices = np.where(pid_mask)[0]
            sorted_order = np.argsort(seg_indices[pid_mask])
            sorted_indices = pid_indices[sorted_order]
            pid_features = features[sorted_indices]
            n_samples = len(pid_features)
            if n_samples >= window_size:
                df_features = pd.DataFrame(pid_features)
                smoothed = df_features.rolling(window=window_size, min_periods=1, center=True).mean().values
                for i, orig_idx in enumerate(sorted_indices):
                    features_smoothed[orig_idx] = smoothed[i]
        return features_smoothed
    
    features = apply_feature_smoothing(features, pids, seg_indices, window_size=3)
    print("特征平滑完成")
    
    return pulse_ts, fatigue_ts, features, targets, pids, seg_indices


def wavelet_transform_batch(data, wavelet='db4', level=4):
    """批量小波变换"""
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


def prepare_data_random_split(pulse_ts, fatigue_ts, features, targets, pids):
    """数据准备 - 使用Random Split"""
    print("\n使用 Random Split 数据划分...")
    
    # 特征标准化和选择
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    n_features = min(50, features.shape[1])
    selector = SelectKBest(score_func=f_regression, k=n_features)
    features_selected = selector.fit_transform(features_scaled, targets)
    
    # 准备数据
    X_pulse = pulse_ts.reshape(pulse_ts.shape[0], pulse_ts.shape[1], 1)
    X_fatigue = fatigue_ts.reshape(fatigue_ts.shape[0], fatigue_ts.shape[1], 1)
    X_features = features_selected
    y = targets
    
    # Random Split
    X_pulse_train, X_pulse_test, X_fatigue_train, X_fatigue_test, \
    X_feat_train, X_feat_test, y_train, y_test, pids_train, pids_test = train_test_split(
        X_pulse, X_fatigue, X_features, y, pids,
        test_size=0.2, random_state=42
    )
    
    print(f"  训练集: {len(y_train)} 样本")
    print(f"  测试集: {len(y_test)} 样本")
    
    # 小波变换
    print("执行小波变换...")
    pulse_train_coeffs = wavelet_transform_batch(X_pulse_train)
    pulse_test_coeffs = wavelet_transform_batch(X_pulse_test)
    fatigue_train_coeffs = wavelet_transform_batch(X_fatigue_train)
    fatigue_test_coeffs = wavelet_transform_batch(X_fatigue_test)
    
    return {
        'X_pulse_train': X_pulse_train, 'X_pulse_test': X_pulse_test,
        'X_fatigue_train': X_fatigue_train, 'X_fatigue_test': X_fatigue_test,
        'X_feat_train': X_feat_train, 'X_feat_test': X_feat_test,
        'y_train': y_train, 'y_test': y_test,
        'pids_train': pids_train, 'pids_test': pids_test,
        'pulse_train_coeffs': pulse_train_coeffs,
        'pulse_test_coeffs': pulse_test_coeffs,
        'fatigue_train_coeffs': fatigue_train_coeffs,
        'fatigue_test_coeffs': fatigue_test_coeffs,
    }


# =============================================================================
# Model Building Functions for Ablation Study
# =============================================================================

def attention_block(x, filters):
    """注意力模块"""
    attention = layers.Dense(filters, activation='tanh')(x)
    attention = layers.Dense(filters, activation='softmax')(attention)
    attended = layers.Multiply()([x, attention])
    return attended


def residual_block(x, filters, kernel_size=3, dropout_rate=0.3, use_attention=True):
    """残差块（可选注意力）"""
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    if use_attention:
        x = attention_block(x, filters)
    return x


def build_wavelet_branch(input_shapes, prefix, filters=32, dropout_rate=0.3, use_attention=True):
    """构建小波分支"""
    inputs, outputs = [], []
    for i, (name, shape) in enumerate(input_shapes):
        inp = Input(shape=shape, name=f'{prefix}_{name}_input')
        inputs.append(inp)
        x = residual_block(inp, filters, dropout_rate=dropout_rate, use_attention=use_attention)
        x = layers.GlobalMaxPooling1D()(x)
        outputs.append(x)
    merged = layers.Concatenate(name=f'{prefix}_wavelet_merge')(outputs)
    return inputs, merged


def build_full_hfenn(data, use_attention=True, use_manual_features=True, 
                     use_pulse=True, use_fatigue=True):
    """
    构建HFENN模型（支持消融配置）
    
    Args:
        data: 数据字典
        use_attention: 是否使用注意力机制
        use_manual_features: 是否使用手工特征
        use_pulse: 是否使用Pulse通道
        use_fatigue: 是否使用Fatigue通道
    """
    all_inputs = []
    feature_branches = []
    
    # Pulse通道小波分支
    if use_pulse:
        pulse_shapes = [
            ('cA4', data['pulse_train_coeffs'][0].shape[1:]),
            ('cD4', data['pulse_train_coeffs'][1].shape[1:]),
            ('cD3', data['pulse_train_coeffs'][2].shape[1:]),
            ('cD2', data['pulse_train_coeffs'][3].shape[1:]),
            ('cD1', data['pulse_train_coeffs'][4].shape[1:]),
        ]
        pulse_inputs, pulse_features = build_wavelet_branch(
            pulse_shapes, 'pulse', use_attention=use_attention)
        all_inputs.extend(pulse_inputs)
        feature_branches.append(pulse_features)
    
    # Fatigue通道小波分支
    if use_fatigue:
        fatigue_shapes = [
            ('cA4', data['fatigue_train_coeffs'][0].shape[1:]),
            ('cD4', data['fatigue_train_coeffs'][1].shape[1:]),
            ('cD3', data['fatigue_train_coeffs'][2].shape[1:]),
            ('cD2', data['fatigue_train_coeffs'][3].shape[1:]),
            ('cD1', data['fatigue_train_coeffs'][4].shape[1:]),
        ]
        fatigue_inputs, fatigue_features = build_wavelet_branch(
            fatigue_shapes, 'fatigue', use_attention=use_attention)
        all_inputs.extend(fatigue_inputs)
        feature_branches.append(fatigue_features)
    
    # 手工特征分支
    if use_manual_features:
        input_enhanced = Input(shape=data['X_feat_train'].shape[1:], name='enhanced_features_input')
        enhanced_branch = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_enhanced)
        enhanced_branch = layers.BatchNormalization()(enhanced_branch)
        enhanced_branch = layers.Dropout(0.5)(enhanced_branch)
        enhanced_branch = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(enhanced_branch)
        all_inputs.append(input_enhanced)
        feature_branches.append(enhanced_branch)
    
    # 融合
    if len(feature_branches) > 1:
        fused = layers.Concatenate()(feature_branches)
    else:
        fused = feature_branches[0]
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='regression_output')(x)
    
    model = models.Model(inputs=all_inputs, outputs=output)
    return model


def build_no_wavelet_model(data):
    """仅使用手工特征的模型（无小波）"""
    input_enhanced = Input(shape=data['X_feat_train'].shape[1:], name='enhanced_features_input')
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_enhanced)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', name='regression_output')(x)
    
    model = models.Model(inputs=[input_enhanced], outputs=output)
    return model


# =============================================================================
# Training and Evaluation
# =============================================================================

def get_custom_metrics():
    """获取自定义指标"""
    try:
        from keras import ops
        def rmse(y_true, y_pred):
            return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))
        def r2_metric(y_true, y_pred):
            ss_res = ops.sum(ops.square(y_true - y_pred))
            ss_tot = ops.sum(ops.square(y_true - ops.mean(y_true)))
            return 1 - ss_res / (ss_tot + 1e-7)
    except ImportError:
        import tensorflow as tf
        def rmse(y_true, y_pred):
            return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
        def r2_metric(y_true, y_pred):
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
            ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return 1 - ss_res / (ss_tot + 1e-7)
    return rmse, r2_metric


def train_and_evaluate(model, train_inputs, test_inputs, y_train, y_test, 
                       model_name, epochs=50, batch_size=32):
    """训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"  Parameters: {model.count_params():,}")
    
    rmse_metric, r2_metric = get_custom_metrics()
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae', rmse_metric, r2_metric]
    )
    
    callbacks_list = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
    ]
    
    history = model.fit(
        train_inputs, y_train,
        validation_data=(test_inputs, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=0
    )
    
    # Evaluate
    y_pred = model.predict(test_inputs, verbose=0).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    rmse_val = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Results:")
    print(f"    R²:   {r2:.4f}")
    print(f"    RMSE: {rmse_val:.4f}")
    print(f"    MAE:  {mae:.4f}")
    
    return {
        'model_name': model_name,
        'R2': r2,
        'RMSE': rmse_val,
        'MAE': mae,
        'MSE': mse,
        'params': model.count_params(),
        'epochs_trained': len(history.history['loss']),
    }


def get_train_test_inputs(data, use_pulse=True, use_fatigue=True, use_manual_features=True):
    """根据配置获取训练和测试输入"""
    train_inputs = []
    test_inputs = []
    
    if use_pulse:
        train_inputs.extend(data['pulse_train_coeffs'])
        test_inputs.extend(data['pulse_test_coeffs'])
    
    if use_fatigue:
        train_inputs.extend(data['fatigue_train_coeffs'])
        test_inputs.extend(data['fatigue_test_coeffs'])
    
    if use_manual_features:
        train_inputs.append(data['X_feat_train'])
        test_inputs.append(data['X_feat_test'])
    
    return train_inputs, test_inputs


# =============================================================================
# Ablation Study Execution
# =============================================================================

def run_ablation_study(data, epochs=50):
    """运行消融实验"""
    print("\n" + "="*70)
    print("HFENN Ablation Study")
    print("="*70)
    
    results = []
    
    # 1. Full HFENN (完整模型)
    print("\n[1/6] Full HFENN (Complete Model)")
    model_full = build_full_hfenn(data, use_attention=True, use_manual_features=True,
                                   use_pulse=True, use_fatigue=True)
    train_inputs, test_inputs = get_train_test_inputs(data, True, True, True)
    result = train_and_evaluate(model_full, train_inputs, test_inputs,
                                data['y_train'], data['y_test'],
                                'Full HFENN', epochs=epochs)
    results.append(result)
    K.clear_session()
    
    # 2. No Attention (无注意力机制)
    print("\n[2/6] No Attention")
    model_no_attn = build_full_hfenn(data, use_attention=False, use_manual_features=True,
                                      use_pulse=True, use_fatigue=True)
    train_inputs, test_inputs = get_train_test_inputs(data, True, True, True)
    result = train_and_evaluate(model_no_attn, train_inputs, test_inputs,
                                data['y_train'], data['y_test'],
                                'No Attention', epochs=epochs)
    results.append(result)
    K.clear_session()
    
    # 3. No Manual Features (无手工特征)
    print("\n[3/6] No Manual Features")
    model_no_feat = build_full_hfenn(data, use_attention=True, use_manual_features=False,
                                      use_pulse=True, use_fatigue=True)
    train_inputs, test_inputs = get_train_test_inputs(data, True, True, False)
    result = train_and_evaluate(model_no_feat, train_inputs, test_inputs,
                                data['y_train'], data['y_test'],
                                'No Manual Features', epochs=epochs)
    results.append(result)
    K.clear_session()
    
    # 4. No Wavelet (无小波，仅手工特征)
    print("\n[4/6] No Wavelet (Manual Features Only)")
    model_no_wavelet = build_no_wavelet_model(data)
    train_inputs = [data['X_feat_train']]
    test_inputs = [data['X_feat_test']]
    result = train_and_evaluate(model_no_wavelet, train_inputs, test_inputs,
                                data['y_train'], data['y_test'],
                                'No Wavelet', epochs=epochs)
    results.append(result)
    K.clear_session()
    
    # 5. Single Channel - Pulse Only
    print("\n[5/6] Single Channel (Pulse Only)")
    model_pulse_only = build_full_hfenn(data, use_attention=True, use_manual_features=True,
                                         use_pulse=True, use_fatigue=False)
    train_inputs, test_inputs = get_train_test_inputs(data, True, False, True)
    result = train_and_evaluate(model_pulse_only, train_inputs, test_inputs,
                                data['y_train'], data['y_test'],
                                'Pulse Only', epochs=epochs)
    results.append(result)
    K.clear_session()
    
    # 6. Single Channel - Fatigue Only
    print("\n[6/6] Single Channel (Fatigue Only)")
    model_fatigue_only = build_full_hfenn(data, use_attention=True, use_manual_features=True,
                                           use_pulse=False, use_fatigue=True)
    train_inputs, test_inputs = get_train_test_inputs(data, False, True, True)
    result = train_and_evaluate(model_fatigue_only, train_inputs, test_inputs,
                                data['y_train'], data['y_test'],
                                'Fatigue Only', epochs=epochs)
    results.append(result)
    K.clear_session()
    
    return pd.DataFrame(results)


# =============================================================================
# Figure 6: Ablation Study Results
# =============================================================================

def plot_figure6(df_results):
    """Figure 6: Ablation Study Results"""
    print("\n" + "="*60)
    print("Generating Figure 6: Ablation Study Results")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Sort by R2 for better visualization
    df_sorted = df_results.sort_values('R2', ascending=True)
    model_names = df_sorted['model_name'].values
    
    # Colors
    colors = ['#4CAF50' if name == 'Full HFENN' else '#1976D2' for name in model_names]
    
    # --- Subplot A: R² Comparison ---
    ax1 = axes[0]
    bars1 = ax1.barh(model_names, df_sorted['R2'], color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('R² Score')
    ax1.set_title('(A) R² Score Comparison')
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars1, df_sorted['R2']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # Highlight full model
    full_idx = np.where(model_names == 'Full HFENN')[0]
    if len(full_idx) > 0:
        ax1.axvline(x=df_sorted[df_sorted['model_name'] == 'Full HFENN']['R2'].values[0],
                   color='#4CAF50', linestyle='--', linewidth=2, alpha=0.7)
    
    # --- Subplot B: RMSE Comparison ---
    ax2 = axes[1]
    df_sorted_rmse = df_results.sort_values('RMSE', ascending=False)
    model_names_rmse = df_sorted_rmse['model_name'].values
    colors_rmse = ['#4CAF50' if name == 'Full HFENN' else '#E53935' for name in model_names_rmse]
    
    bars2 = ax2.barh(model_names_rmse, df_sorted_rmse['RMSE'], color=colors_rmse, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('RMSE')
    ax2.set_title('(B) RMSE Comparison')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars2, df_sorted_rmse['RMSE']):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    # --- Subplot C: Performance Drop Analysis ---
    ax3 = axes[2]
    
    full_r2 = df_results[df_results['model_name'] == 'Full HFENN']['R2'].values[0]
    df_results['R2_drop'] = full_r2 - df_results['R2']
    df_results['R2_drop_pct'] = (df_results['R2_drop'] / full_r2) * 100
    
    df_drop = df_results[df_results['model_name'] != 'Full HFENN'].sort_values('R2_drop', ascending=True)
    
    colors_drop = plt.cm.Reds(np.linspace(0.3, 0.9, len(df_drop)))
    
    bars3 = ax3.barh(df_drop['model_name'], df_drop['R2_drop_pct'], color=colors_drop, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('R² Drop (%)')
    ax3.set_title('(C) Performance Drop vs Full Model')
    ax3.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars3, df_drop['R2_drop_pct']):
        ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure6_Ablation_Study.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure6_Ablation_Study.[png/pdf/svg]")
    
    # Save data
    df_results.to_csv(os.path.join(DATA_DIR, 'ablation_study_results.csv'), index=False)
    print(f"  Saved: ablation_study_results.csv")
    
    plt.close()
    return fig


def plot_figure6_detailed(df_results):
    """Figure 6 (Detailed): Ablation Study with Multiple Metrics"""
    print("\n" + "="*60)
    print("Generating Figure 6 (Detailed): Multi-Metric Comparison")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = df_results['model_name'].values
    x = np.arange(len(model_names))
    width = 0.25
    
    # Normalize metrics for comparison
    r2_vals = df_results['R2'].values
    rmse_vals = df_results['RMSE'].values
    mae_vals = df_results['MAE'].values
    
    bars1 = ax.bar(x - width, r2_vals, width, label='R²', color='#1976D2', alpha=0.8)
    bars2 = ax.bar(x, 1 - rmse_vals, width, label='1 - RMSE', color='#E53935', alpha=0.8)
    bars3 = ax.bar(x + width, 1 - mae_vals, width, label='1 - MAE', color='#4CAF50', alpha=0.8)
    
    ax.set_xlabel('Model Variant')
    ax.set_ylabel('Score (higher is better)')
    ax.set_title('Figure 6: Ablation Study - Multi-Metric Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    # Add value labels for R²
    for bar, val in zip(bars1, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure6_Ablation_Study_Detailed.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure6_Ablation_Study_Detailed.[png/pdf/svg]")
    
    plt.close()
    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("HFENN Ablation Study (v2)")
    print("Based on HFENN_regression_random_split.ipynb")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Extract segments and features
    pulse_ts, fatigue_ts, features, targets, pids, seg_indices = extract_regression_segments(df)
    
    # Prepare model inputs (Random Split)
    data = prepare_data_random_split(pulse_ts, fatigue_ts, features, targets, pids)
    
    # Run ablation study
    df_results = run_ablation_study(data, epochs=50)
    
    # Print summary
    print("\n" + "="*70)
    print("Ablation Study Summary")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # Generate figures
    plot_figure6(df_results)
    plot_figure6_detailed(df_results)
    
    print("\n" + "="*70)
    print("Ablation Study Complete!")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print(f"Data files saved to: {DATA_DIR}/")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
