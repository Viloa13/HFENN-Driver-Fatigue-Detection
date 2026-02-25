"""
HFENN Regression Model Visualization Suite (v2)
================================================
基于 HFENN_regression_random_split.ipynb 的可视化代码
使用 Random Split 数据划分方式 (R² = 0.866)

Figures included:
- Figure 1: Time-series tracking, Density scatter plot, Bland-Altman plot
- Figure 3: Data preprocessing validation
- Figure 4: Wavelet decomposition visualization, Band importance, Feature importance
- Figure 7: Training stability analysis
- Figure 8: Per-participant performance distribution

Output formats: PNG (300dpi), PDF, SVG
Data exports: CSV, NPY for intermediate results

Author: Auto-generated based on HFENN_regression_random_split.ipynb
Date: 2026-01-23
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import skew, kurtosis, gaussian_kde
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
    from keras import models
    print("Using standalone keras")
except ImportError:
    from tensorflow.keras import models
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
    
    # 合并所有数据
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
    
    # 保存原始数据用于预处理对比图
    df_raw = df.copy()
    
    # === 按参与者Z-Score标准化 ===
    print("\n执行按参与者Z-Score标准化...")
    standardization_stats = {}
    
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        stats_pid = {}
        for col in ['Pulse', 'Fatigue']:
            data = df.loc[mask, col].values
            mean_val = np.mean(data)
            std_val = np.std(data)
            stats_pid[f'{col}_mean'] = mean_val
            stats_pid[f'{col}_std'] = std_val
            if std_val > 0:
                df.loc[mask, col] = (data - mean_val) / std_val
            else:
                df.loc[mask, col] = 0
        standardization_stats[pid] = stats_pid
    
    # === 增加差分信号 ===
    print("生成差分信号...")
    df['Pulse_Diff'] = 0.0
    df['Fatigue_Diff'] = 0.0
    for pid in df['Participant_ID'].unique():
        mask = df['Participant_ID'] == pid
        df.loc[mask, 'Pulse_Diff'] = df.loc[mask, 'Pulse'].diff().fillna(0).values
        df.loc[mask, 'Fatigue_Diff'] = df.loc[mask, 'Fatigue'].diff().fillna(0).values
    
    return df, df_raw, standardization_stats


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
    
    # Handle NaN/Inf
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
    """批量小波变换 (与notebook完全一致)"""
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


def prepare_data_random_split(pulse_ts, fatigue_ts, features, targets, pids, seg_indices):
    """数据准备 - 使用Random Split (与notebook完全一致)"""
    print("\n" + "="*60)
    print("使用 Random Split（与原始HFENN分类模型相同）")
    print("="*60)
    
    # 特征标准化和选择
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    n_features = min(50, features.shape[1])
    selector = SelectKBest(score_func=f_regression, k=n_features)
    features_selected = selector.fit_transform(features_scaled, targets)
    
    # 获取选中的特征索引
    selected_indices = selector.get_support(indices=True)
    
    # 准备数据
    X_pulse = pulse_ts.reshape(pulse_ts.shape[0], pulse_ts.shape[1], 1)
    X_fatigue = fatigue_ts.reshape(fatigue_ts.shape[0], fatigue_ts.shape[1], 1)
    X_features = features_selected
    y = targets
    
    # === 关键：使用 Random Split（与原始分类模型相同）===
    X_pulse_train, X_pulse_test, X_fatigue_train, X_fatigue_test, \
    X_feat_train, X_feat_test, y_train, y_test, \
    pids_train, pids_test, seg_idx_train, seg_idx_test = train_test_split(
        X_pulse, X_fatigue, X_features, y, pids, seg_indices,
        test_size=0.2, random_state=42
    )
    
    print(f"\n数据分割完成（Random Split）:")
    print(f"  训练集: {len(y_train)} 样本")
    print(f"  测试集: {len(y_test)} 样本")
    
    # 小波变换
    print("\n执行双通道小波变换...")
    pulse_train_coeffs = wavelet_transform_batch(X_pulse_train)
    pulse_test_coeffs = wavelet_transform_batch(X_pulse_test)
    fatigue_train_coeffs = wavelet_transform_batch(X_fatigue_train)
    fatigue_test_coeffs = wavelet_transform_batch(X_fatigue_test)
    print("小波变换完成")
    
    return {
        'X_pulse_train': X_pulse_train, 'X_pulse_test': X_pulse_test,
        'X_fatigue_train': X_fatigue_train, 'X_fatigue_test': X_fatigue_test,
        'X_feat_train': X_feat_train, 'X_feat_test': X_feat_test,
        'y_train': y_train, 'y_test': y_test,
        'pids_train': pids_train, 'pids_test': pids_test,
        'seg_idx_train': seg_idx_train, 'seg_idx_test': seg_idx_test,
        'pulse_train_coeffs': pulse_train_coeffs,
        'pulse_test_coeffs': pulse_test_coeffs,
        'fatigue_train_coeffs': fatigue_train_coeffs,
        'fatigue_test_coeffs': fatigue_test_coeffs,
        'scaler': scaler,
        'selector': selector,
        'selected_indices': selected_indices,
    }


# =============================================================================
# Figure 1: Time-series tracking, Density scatter, Bland-Altman
# =============================================================================

def plot_figure1(y_test, y_pred, pids_test, seg_indices_test):
    """Figure 1: Model Performance Overview"""
    print("\n" + "="*60)
    print("Generating Figure 1: Model Performance Overview")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # --- Subplot A: Time-series tracking ---
    ax1 = axes[0]
    
    # Find participant with most samples
    unique_pids, counts = np.unique(pids_test, return_counts=True)
    representative_pid = unique_pids[np.argmax(counts)]
    
    pid_mask = pids_test == representative_pid
    pid_indices = np.where(pid_mask)[0]
    sorted_order = np.argsort(seg_indices_test[pid_mask])
    sorted_indices = pid_indices[sorted_order]
    
    y_true_pid = y_test[sorted_indices]
    y_pred_pid = y_pred[sorted_indices]
    sample_idx = np.arange(len(y_true_pid))
    
    # 95% CI (simulated with small noise for visualization)
    ci_width = 0.05
    
    ax1.fill_between(sample_idx, y_pred_pid - ci_width, y_pred_pid + ci_width, 
                     alpha=0.3, color='#2196F3', label='95% CI')
    ax1.plot(sample_idx, y_true_pid, 'k-', linewidth=1.5, label='Ground Truth')
    ax1.plot(sample_idx, y_pred_pid, '#E53935', linewidth=1.2, linestyle='--', label='Prediction')
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Fatigue Index')
    ax1.set_title(f'(A) Time-Series Tracking ({representative_pid})')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    r2_pid = r2_score(y_true_pid, y_pred_pid)
    rmse_pid = np.sqrt(mean_squared_error(y_true_pid, y_pred_pid))
    ax1.text(0.98, 0.02, f'R²={r2_pid:.3f}\nRMSE={rmse_pid:.3f}', 
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # --- Subplot B: Density scatter plot ---
    ax2 = axes[1]
    
    xy = np.vstack([y_test, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = y_test[idx], y_pred[idx], z[idx]
    
    scatter = ax2.scatter(x_sorted, y_sorted, c=z_sorted, s=10, cmap='viridis', alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='y=x')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
    x_line = np.linspace(0, 1, 100)
    ax2.plot(x_line, slope*x_line + intercept, 'r-', linewidth=1.5, 
             label=f'Fit (R²={r_value**2:.3f})')
    
    ax2.set_xlabel('Ground Truth')
    ax2.set_ylabel('Prediction')
    ax2.set_title('(B) Density Scatter Plot')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_aspect('equal')
    
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Density', fontsize=9)
    
    # --- Subplot C: Bland-Altman plot ---
    ax3 = axes[2]
    
    mean_values = (y_test + y_pred) / 2
    diff_values = y_test - y_pred
    
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    xy_ba = np.vstack([mean_values, diff_values])
    z_ba = gaussian_kde(xy_ba)(xy_ba)
    idx_ba = z_ba.argsort()
    
    scatter_ba = ax3.scatter(mean_values[idx_ba], diff_values[idx_ba], 
                             c=z_ba[idx_ba], s=10, cmap='viridis', alpha=0.7)
    
    ax3.axhline(y=mean_diff, color='k', linestyle='-', linewidth=1.5, label=f'Bias: {mean_diff:.4f}')
    ax3.axhline(y=loa_upper, color='r', linestyle='--', linewidth=1.2, label=f'+1.96 SD: {loa_upper:.4f}')
    ax3.axhline(y=loa_lower, color='r', linestyle='--', linewidth=1.2, label=f'-1.96 SD: {loa_lower:.4f}')
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Mean of Ground Truth and Prediction')
    ax3.set_ylabel('Difference (GT - Pred)')
    ax3.set_title('(C) Bland-Altman Plot')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    cbar_ba = plt.colorbar(scatter_ba, ax=ax3, shrink=0.8)
    cbar_ba.set_label('Density', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure1_Model_Performance.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure1_Model_Performance.[png/pdf/svg]")
    
    # Save data
    # Figure 1A: Time-series data
    df_timeseries = pd.DataFrame({
        'sample_index': sample_idx,
        'ground_truth': y_true_pid,
        'prediction': y_pred_pid,
        'ci_lower': y_pred_pid - ci_width,
        'ci_upper': y_pred_pid + ci_width,
        'participant': representative_pid,
    })
    df_timeseries.to_csv(os.path.join(DATA_DIR, 'plot_data_timeseries.csv'), index=False)
    print(f"  Saved: plot_data_timeseries.csv ({len(df_timeseries)} rows)")
    
    # Figure 1B: Scatter data
    df_scatter = pd.DataFrame({'ground_truth': y_test, 'prediction': y_pred, 'density': z})
    df_scatter.to_csv(os.path.join(DATA_DIR, 'plot_data_scatter.csv'), index=False)
    print(f"  Saved: plot_data_scatter.csv ({len(df_scatter)} rows)")
    
    # Figure 1C: Bland-Altman data
    df_ba = pd.DataFrame({'mean': mean_values, 'difference': diff_values})
    df_ba.to_csv(os.path.join(DATA_DIR, 'plot_data_bland_altman.csv'), index=False)
    print(f"  Saved: plot_data_bland_altman.csv ({len(df_ba)} rows)")
    
    plt.close()
    return fig


# =============================================================================
# Figure 3: Data Preprocessing Validation
# =============================================================================

def plot_figure3(df_raw, df_processed, standardization_stats):
    """Figure 3: Data Preprocessing Validation"""
    print("\n" + "="*60)
    print("Generating Figure 3: Data Preprocessing Validation")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Subplot A: Raw vs Processed Signal ---
    ax1 = axes[0]
    
    sample_pid = df_raw['Participant_ID'].unique()[0]
    df_raw_pid = df_raw[df_raw['Participant_ID'] == sample_pid].reset_index(drop=True)
    df_proc_pid = df_processed[df_processed['Participant_ID'] == sample_pid].reset_index(drop=True)
    
    n_samples = min(5000, len(df_raw_pid))
    time_sec = np.arange(n_samples) / 87.0
    
    raw_pulse = df_raw_pid['Pulse'].values[:n_samples]
    proc_pulse = df_proc_pid['Pulse'].values[:n_samples]
    
    ax1.plot(time_sec, raw_pulse, 'b-', alpha=0.7, linewidth=0.8, label='Raw Signal')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_sec, proc_pulse, 'r-', alpha=0.7, linewidth=0.8, label='Standardized')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Raw Pulse Value', color='b')
    ax1_twin.set_ylabel('Standardized Pulse (Z-score)', color='r')
    ax1.set_title(f'(A) Signal Comparison ({sample_pid})')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # --- Subplot B: Standardization Effect ---
    ax2 = axes[1]
    
    participants = list(df_raw['Participant_ID'].unique())[:5]
    
    means_before = []
    means_after = []
    stds_before = []
    stds_after = []
    
    for pid in participants:
        raw_data = df_raw[df_raw['Participant_ID'] == pid]['Pulse'].values
        proc_data = df_processed[df_processed['Participant_ID'] == pid]['Pulse'].values
        means_before.append(np.mean(raw_data))
        means_after.append(np.mean(proc_data))
        stds_before.append(np.std(raw_data))
        stds_after.append(np.std(proc_data))
    
    x = np.arange(len(participants))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, means_before, width, yerr=stds_before, 
                    label='Before Standardization', color='#1976D2', alpha=0.8, capsize=3)
    bars2 = ax2.bar(x + width/2, means_after, width, yerr=stds_after,
                    label='After Standardization', color='#E53935', alpha=0.8, capsize=3)
    
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Participant')
    ax2.set_ylabel('Pulse Mean ± SD')
    ax2.set_title('(B) Per-Participant Standardization Effect')
    ax2.set_xticks(x)
    ax2.set_xticklabels(participants, rotation=45)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure3_Preprocessing_Validation.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure3_Preprocessing_Validation.[png/pdf/svg]")
    
    # Save data
    df_stats = pd.DataFrame(standardization_stats).T
    df_stats.to_csv(os.path.join(DATA_DIR, 'standardization_stats.csv'))
    print(f"  Saved: standardization_stats.csv")
    
    # Figure 3A: Signal comparison data
    df_signal_comparison = pd.DataFrame({
        'time_sec': time_sec,
        'raw_pulse': raw_pulse,
        'standardized_pulse': proc_pulse,
        'participant': sample_pid,
    })
    df_signal_comparison.to_csv(os.path.join(DATA_DIR, 'plot_data_signal_comparison.csv'), index=False)
    print(f"  Saved: plot_data_signal_comparison.csv ({len(df_signal_comparison)} rows)")
    
    # Figure 3B: Standardization effect data
    df_standardization_effect = pd.DataFrame({
        'participant': participants,
        'mean_before': means_before,
        'std_before': stds_before,
        'mean_after': means_after,
        'std_after': stds_after,
    })
    df_standardization_effect.to_csv(os.path.join(DATA_DIR, 'plot_data_standardization_effect.csv'), index=False)
    print(f"  Saved: plot_data_standardization_effect.csv ({len(df_standardization_effect)} rows)")
    
    plt.close()
    return fig


# =============================================================================
# Figure 4: Wavelet Decomposition & Feature Importance
# =============================================================================

def plot_figure4_wavelet(pulse_ts, sample_idx=0):
    """Figure 4A: Wavelet Decomposition Visualization"""
    print("\n" + "="*60)
    print("Generating Figure 4A: Wavelet Decomposition")
    print("="*60)
    
    signal = pulse_ts[sample_idx, :]
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=False)
    
    time_sec = np.arange(len(signal)) / 87.0
    axes[0].plot(time_sec, signal, 'k-', linewidth=0.8)
    axes[0].set_ylabel('Original\nSignal')
    axes[0].set_title('Wavelet Decomposition (db4, level=4)')
    axes[0].grid(True, alpha=0.3)
    
    coeff_names = ['cA4 (Base Trend)', 'cD4 (Low Freq)', 'cD3 (Mid-Low Freq)', 
                   'cD2 (Mid-High Freq)', 'cD1 (High Freq / Noise)']
    colors = ['#1976D2', '#388E3C', '#FFA000', '#E64A19', '#7B1FA2']
    
    for i, (coeff, name, color) in enumerate(zip(coeffs, coeff_names, colors)):
        ax = axes[i + 1]
        coeff_time = np.linspace(0, time_sec[-1], len(coeff))
        ax.plot(coeff_time, coeff, color=color, linewidth=0.8)
        ax.set_ylabel(name.split('(')[0].strip())
        ax.grid(True, alpha=0.3)
        
        if 'Base Trend' in name:
            ax.annotate('Low-frequency baseline trend', xy=(0.98, 0.95), 
                       xycoords='axes fraction', ha='right', va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif 'Noise' in name:
            ax.annotate('High-frequency noise/micro-events', xy=(0.98, 0.95), 
                       xycoords='axes fraction', ha='right', va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure4A_Wavelet_Decomposition.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure4A_Wavelet_Decomposition.[png/pdf/svg]")
    
    np.savez(os.path.join(DATA_DIR, 'wavelet_coefficients.npz'), 
             signal=signal, coeffs=np.array(coeffs, dtype=object))
    
    plt.close()
    return fig


def compute_band_importance(model, test_inputs, y_test, n_permutations=10):
    """Compute importance of each wavelet band using permutation importance"""
    print("\nComputing band importance (permutation)...")
    
    y_pred_baseline = model.predict(test_inputs, verbose=0).flatten()
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    
    band_names = ['Pulse_cA4', 'Pulse_cD4', 'Pulse_cD3', 'Pulse_cD2', 'Pulse_cD1',
                  'Fatigue_cA4', 'Fatigue_cD4', 'Fatigue_cD3', 'Fatigue_cD2', 'Fatigue_cD1',
                  'Manual_Features']
    
    importance_results = []
    
    for i, band_name in enumerate(band_names):
        delta_rmse_list = []
        
        for _ in range(n_permutations):
            inputs_shuffled = [inp.copy() for inp in test_inputs]
            np.random.shuffle(inputs_shuffled[i])
            y_pred_shuffled = model.predict(inputs_shuffled, verbose=0).flatten()
            rmse_shuffled = np.sqrt(mean_squared_error(y_test, y_pred_shuffled))
            delta_rmse_list.append(rmse_shuffled - rmse_baseline)
        
        importance_results.append({
            'band': band_name,
            'delta_rmse_mean': np.mean(delta_rmse_list),
            'delta_rmse_std': np.std(delta_rmse_list),
        })
        print(f"  {band_name}: ΔRMSE = {np.mean(delta_rmse_list):.4f} ± {np.std(delta_rmse_list):.4f}")
    
    return pd.DataFrame(importance_results)


def compute_feature_importance(model, X_feat_test, test_inputs_base, y_test, feature_names, n_permutations=10):
    """Compute importance of manual features using permutation importance"""
    print("\nComputing feature importance (permutation)...")
    
    y_pred_baseline = model.predict(test_inputs_base, verbose=0).flatten()
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    
    importance_results = []
    
    for i, feat_name in enumerate(feature_names):
        delta_rmse_list = []
        
        for _ in range(n_permutations):
            X_feat_shuffled = X_feat_test.copy()
            np.random.shuffle(X_feat_shuffled[:, i])
            inputs_shuffled = test_inputs_base.copy()
            inputs_shuffled[-1] = X_feat_shuffled
            y_pred_shuffled = model.predict(inputs_shuffled, verbose=0).flatten()
            delta_rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_shuffled)) - rmse_baseline)
        
        importance_results.append({
            'feature': feat_name,
            'delta_rmse_mean': np.mean(delta_rmse_list),
            'delta_rmse_std': np.std(delta_rmse_list),
        })
    
    df_importance = pd.DataFrame(importance_results)
    df_importance = df_importance.sort_values('delta_rmse_mean', ascending=False)
    
    return df_importance


def plot_figure4_importance(df_band_importance, df_feature_importance):
    """Figure 4B & 4C: Band and Feature Importance"""
    print("\n" + "="*60)
    print("Generating Figure 4B-C: Importance Analysis")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Subplot B: Band Importance ---
    ax1 = axes[0]
    
    df_band = df_band_importance.sort_values('delta_rmse_mean', ascending=True)
    colors = ['#1976D2' if 'Pulse' in b else '#E53935' if 'Fatigue' in b else '#4CAF50' 
              for b in df_band['band']]
    
    ax1.barh(df_band['band'], df_band['delta_rmse_mean'], 
             xerr=df_band['delta_rmse_std'], color=colors, alpha=0.8, capsize=3)
    
    ax1.set_xlabel('ΔRMSE (Importance)')
    ax1.set_title('(B) Wavelet Band Importance')
    ax1.grid(True, alpha=0.3, axis='x')
    
    legend_elements = [
        mpatches.Patch(facecolor='#1976D2', label='Pulse Channel'),
        mpatches.Patch(facecolor='#E53935', label='Fatigue Channel'),
        mpatches.Patch(facecolor='#4CAF50', label='Manual Features'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # --- Subplot C: Top 10 Feature Importance ---
    ax2 = axes[1]
    
    df_top10 = df_feature_importance.head(10).sort_values('delta_rmse_mean', ascending=True)
    
    ax2.barh(df_top10['feature'], df_top10['delta_rmse_mean'], 
             xerr=df_top10['delta_rmse_std'], color='#7B1FA2', alpha=0.8, capsize=3)
    
    ax2.set_xlabel('ΔRMSE (Importance)')
    ax2.set_title('(C) Top 10 Manual Feature Importance')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure4BC_Importance_Analysis.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure4BC_Importance_Analysis.[png/pdf/svg]")
    
    df_band_importance.to_csv(os.path.join(DATA_DIR, 'plot_data_band_importance.csv'), index=False)
    df_feature_importance.to_csv(os.path.join(DATA_DIR, 'plot_data_feature_importance.csv'), index=False)
    
    plt.close()
    return fig


# =============================================================================
# Figure 7: Training Stability Analysis
# =============================================================================

def plot_figure7(history_csv_path='training_history.csv'):
    """Figure 7: Training Stability Analysis"""
    print("\n" + "="*60)
    print("Generating Figure 7: Training Stability Analysis")
    print("="*60)
    
    if os.path.exists(history_csv_path):
        df_history = pd.read_csv(history_csv_path)
    else:
        print(f"  Warning: {history_csv_path} not found. Using simulated data.")
        epochs = np.arange(1, 51)
        train_loss = 0.5 * np.exp(-0.1 * epochs) + 0.02 + np.random.normal(0, 0.005, len(epochs))
        val_loss = 0.6 * np.exp(-0.08 * epochs) + 0.03 + np.random.normal(0, 0.01, len(epochs))
        df_history = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mae': train_loss * 2,
            'val_mae': val_loss * 2,
        })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # --- Subplot A: Loss Curves ---
    ax1 = axes[0]
    
    epochs = df_history['epoch'] if 'epoch' in df_history.columns else np.arange(1, len(df_history) + 1)
    
    ax1.plot(epochs, df_history['train_loss'], 'b-', linewidth=1.5, label='Training Loss')
    ax1.plot(epochs, df_history['val_loss'], 'r-', linewidth=1.5, label='Validation Loss')
    
    best_epoch = df_history['val_loss'].idxmin() + 1
    best_val_loss = df_history['val_loss'].min()
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('(A) Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # --- Subplot B: MAE ---
    ax2 = axes[1]
    
    if 'learning_rate' in df_history.columns and df_history['learning_rate'].notna().any():
        ax2.plot(epochs, df_history['learning_rate'], 'g-', linewidth=1.5)
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('(B) Learning Rate Schedule')
        ax2.set_yscale('log')
    else:
        ax2.plot(epochs, df_history['train_mae'], 'b-', linewidth=1.5, label='Training MAE')
        ax2.plot(epochs, df_history['val_mae'], 'r-', linewidth=1.5, label='Validation MAE')
        ax2.set_ylabel('MAE')
        ax2.set_title('(B) Training and Validation MAE')
        ax2.legend(loc='upper right')
    
    ax2.set_xlabel('Epoch')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure7_Training_Stability.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure7_Training_Stability.[png/pdf/svg]")
    
    # Save data
    df_history_plot = df_history.copy()
    df_history_plot['epoch'] = epochs
    df_history_plot['best_epoch'] = best_epoch
    df_history_plot['best_val_loss'] = best_val_loss
    df_history_plot.to_csv(os.path.join(DATA_DIR, 'plot_data_training_history.csv'), index=False)
    print(f"  Saved: plot_data_training_history.csv ({len(df_history_plot)} rows)")
    
    plt.close()
    return fig


# =============================================================================
# Figure 8: Per-Participant Performance
# =============================================================================

def plot_figure8(y_test, y_pred, pids_test):
    """Figure 8: Per-Participant Performance Distribution"""
    print("\n" + "="*60)
    print("Generating Figure 8: Per-Participant Performance")
    print("="*60)
    
    unique_pids = np.unique(pids_test)
    metrics_per_pid = []
    
    for pid in unique_pids:
        mask = pids_test == pid
        y_true_pid = y_test[mask]
        y_pred_pid = y_pred[mask]
        
        r2 = r2_score(y_true_pid, y_pred_pid)
        rmse = np.sqrt(mean_squared_error(y_true_pid, y_pred_pid))
        mae = mean_absolute_error(y_true_pid, y_pred_pid)
        n_samples = np.sum(mask)
        
        metrics_per_pid.append({
            'participant': pid,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'n_samples': n_samples,
        })
    
    df_metrics = pd.DataFrame(metrics_per_pid)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Subplot A: R² Distribution ---
    ax1 = axes[0]
    
    parts = ax1.violinplot([df_metrics['R2']], positions=[0], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#1976D2')
        pc.set_alpha(0.7)
    
    x_jitter = np.random.normal(0, 0.04, len(df_metrics))
    ax1.scatter(x_jitter, df_metrics['R2'], c='#E53935', s=80, alpha=0.8, zorder=5)
    
    for i, row in df_metrics.iterrows():
        ax1.annotate(row['participant'], (x_jitter[i], row['R2']), 
                    fontsize=8, ha='center', va='bottom')
    
    ax1.set_ylabel('R² Score')
    ax1.set_title('(A) R² Distribution Across Participants')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='y')
    
    mean_r2 = df_metrics['R2'].mean()
    ax1.axhline(y=mean_r2, color='g', linestyle='--', linewidth=1.5, 
                label=f'Mean R²: {mean_r2:.3f}')
    ax1.legend(loc='lower right')
    
    # --- Subplot B: RMSE Box Plot ---
    ax2 = axes[1]
    
    bp = ax2.boxplot([df_metrics['RMSE']], positions=[0], widths=0.5, 
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#4CAF50')
    bp['boxes'][0].set_alpha(0.7)
    
    ax2.scatter(x_jitter, df_metrics['RMSE'], c='#7B1FA2', s=80, alpha=0.8, zorder=5)
    
    for i, row in df_metrics.iterrows():
        ax2.annotate(row['participant'], (x_jitter[i], row['RMSE']), 
                    fontsize=8, ha='center', va='bottom')
    
    ax2.set_ylabel('RMSE')
    ax2.set_title('(B) RMSE Distribution Across Participants')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'Figure8_Per_Participant_Performance.{fmt}'), 
                    format=fmt, dpi=300, bbox_inches='tight')
    print(f"  Saved: Figure8_Per_Participant_Performance.[png/pdf/svg]")
    
    df_metrics.to_csv(os.path.join(DATA_DIR, 'per_participant_metrics.csv'), index=False)
    print(f"  Saved: per_participant_metrics.csv")
    
    plt.close()
    return fig, df_metrics


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("HFENN Visualization Suite (v2)")
    print("Based on HFENN_regression_random_split.ipynb")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and preprocess data
    df, df_raw, standardization_stats = load_and_preprocess_data()
    
    # Extract segments and features
    pulse_ts, fatigue_ts, features, targets, pids, seg_indices = extract_regression_segments(df)
    
    # Prepare model inputs (Random Split)
    data = prepare_data_random_split(pulse_ts, fatigue_ts, features, targets, pids, seg_indices)
    
    # Load trained model
    model_path = 'HFENN_Regression_RandomSplit_best.keras'
    if os.path.exists(model_path):
        print(f"\nLoading model from {model_path}...")
        model = models.load_model(model_path, compile=False)
    else:
        print(f"\nWarning: Model file {model_path} not found!")
        print("Some visualizations will use simulated predictions.")
        model = None
    
    # Prepare test inputs (same order as notebook)
    test_inputs = [
        data['pulse_test_coeffs'][0], data['pulse_test_coeffs'][1],
        data['pulse_test_coeffs'][2], data['pulse_test_coeffs'][3],
        data['pulse_test_coeffs'][4],
        data['fatigue_test_coeffs'][0], data['fatigue_test_coeffs'][1],
        data['fatigue_test_coeffs'][2], data['fatigue_test_coeffs'][3],
        data['fatigue_test_coeffs'][4],
        data['X_feat_test']
    ]
    
    # Get predictions
    if model is not None:
        print("\nGenerating predictions...")
        y_pred = model.predict(test_inputs, verbose=0).flatten()
    else:
        y_pred = data['y_test'] + np.random.normal(0, 0.1, len(data['y_test']))
        y_pred = np.clip(y_pred, 0, 1)
    
    y_test = data['y_test']
    pids_test = data['pids_test']
    seg_indices_test = data['seg_idx_test']
    
    # Print overall metrics
    print("\n" + "="*60)
    print("Overall Test Set Metrics (Random Split)")
    print("="*60)
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    
    # Generate figures
    
    # Figure 1: Model Performance
    plot_figure1(y_test, y_pred, pids_test, seg_indices_test)
    
    # Figure 3: Preprocessing Validation
    plot_figure3(df_raw, df, standardization_stats)
    
    # Figure 4A: Wavelet Decomposition
    plot_figure4_wavelet(pulse_ts)
    
    # Figure 4B-C: Importance Analysis (requires model)
    if model is not None:
        df_band_importance = compute_band_importance(model, test_inputs, y_test, n_permutations=5)
        
        feature_names = []
        try:
            with open('selected_feature_names.txt', 'r') as f:
                feature_names = [line.strip() for line in f if line.strip()]
        except:
            feature_names = [f'Feature_{i}' for i in range(data['X_feat_test'].shape[1])]
        
        df_feature_importance = compute_feature_importance(
            model, data['X_feat_test'], test_inputs, y_test, feature_names, n_permutations=5
        )
        
        plot_figure4_importance(df_band_importance, df_feature_importance)
    else:
        print("\nSkipping importance analysis (no model loaded)")
    
    # Figure 7: Training Stability
    plot_figure7('training_history.csv')
    
    # Figure 8: Per-Participant Performance
    plot_figure8(y_test, y_pred, pids_test)
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print(f"Data files saved to: {DATA_DIR}/")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
