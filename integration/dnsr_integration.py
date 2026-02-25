"""
DNSR集成模块 - 连接感知模型与决策系统的桥梁
将HFENN（疲劳检测）和WCNN（情绪检测）模型集成到MARL系统中

作者: AI Assistant
版本: v1.1
日期: 2025-0904
"""

import numpy as np
import pandas as pd
import pywt
import warnings
from typing import Dict, List, Tuple, Optional, Union
import time
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

# 深度学习框架
try:
    from keras.models import load_model
    from keras.utils import to_categorical
    print("✅ 使用独立的 keras")
except ImportError:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical
    print("✅ 使用 tensorflow.keras")

# 信号处理和机器学习
from scipy.signal import find_peaks, hilbert, stft, welch
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

@dataclass
class DriverState:
    """驾驶员状态数据结构"""
    vehicle_id: str
    timestamp: float
    fatigue_level: int      # 0=正常, 1=轻度疲劳, 2=疲劳
    emotion_state: int      # 0=正常, 1=负面情绪
    risk_level: float       # 0.0-1.0综合风险等级
    confidence: float       # 预测置信度
    blink_features: np.ndarray  # 眨眼特征向量
    pulse_features: np.ndarray  # 心率特征向量

class EnhancedFeatureExtractor:
    """增强版特征提取器 - 从HFENN提取"""
    
    def __init__(self, sampling_rate: int = 87):
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """时域特征提取"""
        features = []
        
        # 基础统计特征
        features.extend([
            np.mean(signal),           # 均值
            np.std(signal),            # 标准差
            np.max(signal),            # 最大值
            np.min(signal),            # 最小值
            np.median(signal),         # 中位数
            skew(signal),              # 偏度
            kurtosis(signal),          # 峰度
            np.var(signal),            # 方差
            np.ptp(signal),            # 峰峰值
            np.percentile(signal, 25), # 25%分位数
            np.percentile(signal, 75), # 75%分位数
        ])
        
        # 峰值特征
        peaks, _ = find_peaks(signal)
        valleys, _ = find_peaks(-signal)
        features.extend([
            len(peaks),                # 峰值数量
            len(valleys),              # 谷值数量
        ])
        
        # 信号能量特征
        features.extend([
            np.sum(signal**2),         # 总能量
            np.mean(signal**2),        # 平均功率
            np.sqrt(np.mean(signal**2)), # RMS
        ])
        
        return np.array(features)
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """频域特征提取"""
        features = []
        
        # 功率谱密度
        freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
        
        # 频域统计特征
        features.extend([
            np.mean(psd),              # 平均功率
            np.std(psd),               # 功率标准差
            np.max(psd),               # 最大功率
            freqs[np.argmax(psd)],     # 主频率
            np.sum(psd),               # 总功率
        ])
        
        # 频带功率分布
        freq_bands = [(0, 1), (1, 5), (5, 15), (15, 25)]
        total_power = np.sum(psd)
        
        for low, high in freq_bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask])
            features.append(band_power / total_power if total_power > 0 else 0)
        
        return np.array(features)
    '''
    def extract_wavelet_features(self, signal: np.ndarray) -> np.ndarray:
        """小波特征提取 - 使用db4小波，4层分解"""
        features = []
        
        try:
            # 多层小波分解
            coeffs = pywt.wavedec(signal, 'db4', level=4)
            
            # 各层系数的统计特征
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    features.extend([
                        np.mean(coeff),        # 均值
                        np.std(coeff),         # 标准差
                        np.sum(coeff**2),      # 能量
                        np.max(np.abs(coeff)), # 最大绝对值
                    ])
        except Exception as e:
            logging.warning(f"小波变换失败: {e}, 使用零填充")
            # 如果小波变换失败，用零填充
            features.extend([0.0] * 20)  # 4层 * 4特征 = 16，加上一些额外特征
        
        return np.array(features)
    '''
    def extract_wavelet_features(self, signal: np.ndarray) -> np.ndarray:
        """小波特征提取 - 使用db4小波，4层分解，确保固定特征数量"""
        features = []
        
        try:
            # 多层小波分解
            coeffs = pywt.wavedec(signal, 'db4', level=4)
            
            # 固定特征数量：5层系数 × 4个统计特征 = 20个特征
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    features.extend([
                        np.mean(coeff),        # 均值
                        np.std(coeff),         # 标准差
                        np.sum(coeff**2),      # 能量
                        np.max(np.abs(coeff)), # 最大绝对值
                    ])
                else:
                    # 如果某层系数为空，用零填充
                    features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 确保正好20个特征
            while len(features) < 20:
                features.append(0.0)
            features = features[:20]  # 截断到20个
            
        except Exception as e:
            self.logger.warning(f"小波变换失败: {e}, 使用零填充")
            # 如果小波变换失败，用零填充20个特征
            features = [0.0] * 20
        
        return np.array(features)

    def extract_nonlinear_features(self, signal: np.ndarray) -> np.ndarray:
        """非线性动态特征"""
        features = []
        
        try:
            # 希尔伯特变换
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.angle(analytic_signal)
            
            features.extend([
                np.mean(amplitude_envelope),   # 包络均值
                np.std(amplitude_envelope),    # 包络标准差
                np.mean(np.diff(instantaneous_phase)), # 瞬时频率均值
            ])
            
            # 过零率
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
            features.append(zero_crossings)
            
        except Exception as e:
            logging.warning(f"非线性特征提取失败: {e}, 使用零填充")
            features.extend([0.0] * 4)
        
        return np.array(features)
    
    def extract_all_features(self, signal: np.ndarray) -> np.ndarray:
        """提取所有特征"""
        all_features = []
        
        # 时域特征
        time_features = self.extract_time_domain_features(signal)
        all_features.extend(time_features)
        
        # 频域特征
        freq_features = self.extract_frequency_domain_features(signal)
        all_features.extend(freq_features)
        
        # 小波特征
        wavelet_features = self.extract_wavelet_features(signal)
        all_features.extend(wavelet_features)
        
        # 非线性特征
        nonlinear_features = self.extract_nonlinear_features(signal)
        all_features.extend(nonlinear_features)
        
        return np.array(all_features)

class DNSRIntegration:
    """DNSR集成模块 - 连接感知模型与决策系统"""
    
    def __init__(self, 
                 hfenn_model_path: str = "Enhanced_HFENN_best.keras",
                 wcnn_model_path: str = "data/WCNN_best.keras",
                 feature_scaler_path: str = "data/wcnn_scaler.pkl",
                 selected_features_path: str = "selected_feature_names.pkl"):
        """
        初始化DNSR集成模块
        
        Args:
            hfenn_model_path: HFENN模型文件路径
            wcnn_model_path: WCNN模型文件路径
            feature_scaler_path: 特征标准化器文件路径
            selected_features_path: 特征选择器文件路径
        """
        self.logger = self._setup_logging()
        self.logger.info("🚀 初始化DNSR集成模块...")
        
        # 初始化特征提取器
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # 加载模型
        self.hfenn_model = self._load_hfenn_model(hfenn_model_path)
        self.wcnn_model = self._load_wcnn_model(wcnn_model_path)
        
        # 加载预处理器
        self.feature_scaler = self._load_feature_scaler(feature_scaler_path)
        self.selected_features = self._load_selected_features(selected_features_path)
        
        # 模型配置
        self.hfenn_config = {
            'window_size': 2610,      # HFENN窗口大小
            'overlap_ratio': 0.5,     # 重叠比例
            'sampling_rate': 87,      # 采样率
            'num_classes': 3          # 疲劳等级数
        }
        
        self.wcnn_config = {
            'window_size': 2610,      # WCNN窗口大小 (870*3)
            'sampling_rate': 87,      # 采样率
            'num_classes': 2          # 情绪类别数
        }
        
        # 状态缓存
        self.driver_states = {}
        self.last_update_time = {}
        
        self.logger.info("✅ DNSR集成模块初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_hfenn_model(self, model_path: str):
        """加载HFENN模型"""
        try:
            if Path(model_path).exists():
                model = load_model(model_path)
                self.logger.info(f"✅ 成功加载HFENN模型: {model_path}")
                return model
            else:
                self.logger.warning(f"⚠️ HFENN模型文件不存在: {model_path}")
                return None
        except Exception as e:
            self.logger.error(f"❌ 加载HFENN模型失败: {e}")
            return None
    
    def _load_wcnn_model(self, model_path: str):
        """加载WCNN模型"""
        try:
            if Path(model_path).exists():
                # 启用不安全的反序列化以支持Lambda层
                import keras
                keras.config.enable_unsafe_deserialization()
                
                model = load_model(model_path)
                self.logger.info(f"✅ 成功加载WCNN模型: {model_path}")
                return model
            else:
                self.logger.warning(f"⚠️ WCNN模型文件不存在: {model_path}")
                return None
        except Exception as e:
            self.logger.error(f"❌ 加载WCNN模型失败: {e}")
            return None
    '''
    def _load_feature_scaler(self, scaler_path: str):
        """加载特征标准化器"""
        try:
            if Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                self.logger.info(f"✅ 成功加载特征标准化器: {scaler_path}")
                return scaler
            else:
                self.logger.warning(f"⚠️ 特征标准化器文件不存在: {scaler_path}")
                return StandardScaler()
        except Exception as e:
            self.logger.error(f"❌ 加载特征标准化器失败: {e}")
            return StandardScaler()
    
    def _load_feature_scaler(self, scaler_path: str):
        """加载特征标准化器"""
        try:
            if Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # 验证scaler是否已fitted
                if hasattr(scaler, 'mean_'):
                    self.logger.info(f"✅ 成功加载特征标准化器: {scaler_path}")
                    self.logger.info(f"标准化器期望特征数量: {len(scaler.mean_)}")
                    return scaler
                else:
                    self.logger.warning(f"⚠️ 标准化器未fitted，将创建新的标准化器")
                    return self._create_fallback_scaler()
            else:
                self.logger.warning(f"⚠️ 特征标准化器文件不存在: {scaler_path}")
                return self._create_fallback_scaler()
        except Exception as e:
            self.logger.error(f"❌ 加载特征标准化器失败: {e}")
            return self._create_fallback_scaler()
    '''
    def _load_feature_scaler(self, scaler_path: str):
        """加载特征标准化器"""
        try:
            if Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # 验证scaler是否已fitted
                if hasattr(scaler, 'mean_'):
                    expected_features = len(scaler.mean_)
                    self.logger.info(f"✅ 成功加载特征标准化器: {scaler_path}")
                    self.logger.info(f"标准化器期望特征数量: {expected_features}")
                    
                    # 🔧 如果scaler不是50维，重新创建50维的
                    if expected_features != 50:
                        self.logger.warning(f"标准化器特征数({expected_features})不匹配模型要求(50)，重新创建")
                        return self._create_fallback_scaler(target_features=50)
                    
                    return scaler
                else:
                    self.logger.warning(f"⚠️ 标准化器未fitted，将创建新的标准化器")
                    return self._create_fallback_scaler(target_features=50)
            else:
                self.logger.warning(f"⚠️ 特征标准化器文件不存在: {scaler_path}")
                return self._create_fallback_scaler(target_features=50)
        except Exception as e:
            self.logger.error(f"❌ 加载特征标准化器失败: {e}")
            return self._create_fallback_scaler(target_features=50)

    '''
    def _create_fallback_scaler(self):
        """创建备用的标准化器"""
        # 创建一个dummy信号来拟合scaler
        dummy_signal = np.random.randn(2610)
        dummy_features = self.feature_extractor.extract_all_features(dummy_signal)
        
        scaler = StandardScaler()
        scaler.fit(dummy_features.reshape(1, -1))
        
        self.logger.warning(f"⚠️ 使用模拟数据创建标准化器，特征数量: {len(dummy_features)}")
        return scaler
    '''
    def _create_fallback_scaler(self, target_features: int = 50):
        """创建备用的标准化器"""
        # 创建一个dummy信号来拟合scaler
        dummy_signal = np.random.randn(2610)
        dummy_features = self.feature_extractor.extract_all_features(dummy_signal)
        
        # 🔧 确保特征数量匹配目标
        if len(dummy_features) < target_features:
            padding = np.zeros(target_features - len(dummy_features))
            dummy_features = np.concatenate([dummy_features, padding])
        elif len(dummy_features) > target_features:
            dummy_features = dummy_features[:target_features]
        
        scaler = StandardScaler()
        scaler.fit(dummy_features.reshape(1, -1))
        
        self.logger.warning(f"⚠️ 使用模拟数据创建{target_features}维标准化器")
        return scaler

    def _load_selected_features(self, features_path: str):
        """加载特征选择器"""
        try:
            if Path(features_path).exists():
                with open(features_path, 'rb') as f:
                    selected_features = pickle.load(f)
                self.logger.info(f"✅ 成功加载特征选择器: {features_path}")
                return selected_features
            else:
                self.logger.warning(f"⚠️ 特征选择器文件不存在: {features_path}")
                return None
        except Exception as e:
            self.logger.error(f"❌ 加载特征选择器失败: {e}")
            return None

    def _simulate_driver_data(self, vehicle_id: str, timestamp: float) -> Dict[str, np.ndarray]:
        """
        模拟驾驶员生理数据
        
        Args:
            vehicle_id: 车辆ID
            timestamp: 时间戳
            
        Returns:
            包含眨眼和心率数据的字典
        """
        # 基于车辆ID生成稳定的随机种子
        seed = hash(vehicle_id) % 10000
        np.random.seed(seed)
        
        # 生成时间序列
        window_size = self.hfenn_config['window_size']
        time_steps = np.arange(window_size)
        
        # 模拟眨眼数据（基于HFENN的归一化眨眼频率）
        # 基础眨眼频率 + 疲劳趋势 + 噪声
        base_blink = 0.5 + 0.2 * np.sin(timestamp * 0.1)  # 基础频率
        fatigue_trend = 0.1 * np.sin(timestamp * 0.05)     # 疲劳趋势
        noise = np.random.normal(0, 0.05, window_size)    # 随机噪声
        
        blink_data = np.clip(base_blink + fatigue_trend + noise, 0, 1)
        
        # 模拟心率数据（基于WCNN的归一化心率）
        # 基础心率 + 情绪波动 + 噪声
        base_pulse = 0.6 + 0.15 * np.sin(timestamp * 0.08)  # 基础心率
        emotion_fluctuation = 0.1 * np.sin(timestamp * 0.12) # 情绪波动
        pulse_noise = np.random.normal(0, 0.03, window_size) # 随机噪声
        
        pulse_data = np.clip(base_pulse + emotion_fluctuation + pulse_noise, 0, 1)
        
        return {
            'blink_data': blink_data,
            'pulse_data': pulse_data
        }
    '''
    def _preprocess_blink_data(self, blink_data: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        预处理眨眼数据（HFENN格式）
        
        Args:
            blink_data: 原始眨眼数据
            
        Returns:
            小波系数列表和增强特征
        """
        # 提取增强特征
        enhanced_features = self.feature_extractor.extract_all_features(blink_data)
        
        # 特征标准化
        if self.feature_scaler is not None:
            enhanced_features = enhanced_features.reshape(1, -1)
            enhanced_features = self.feature_scaler.transform(enhanced_features)
            enhanced_features = enhanced_features.flatten()
        
        # 特征选择
        if self.selected_features is not None:
            enhanced_features = enhanced_features[self.selected_features]
        
        # 小波变换（HFENN格式）
        try:
            coeffs = pywt.wavedec(blink_data, 'db4', level=4)
            
            # 重塑为模型输入格式
            cA4 = coeffs[0].reshape(1, -1, 1)
            cD4 = coeffs[1].reshape(1, -1, 1)
            cD3 = coeffs[2].reshape(1, -1, 1)
            cD2 = coeffs[3].reshape(1, -1, 1)
            cD1 = coeffs[4].reshape(1, -1, 1)
            
            wavelet_coeffs = [cA4, cD4, cD3, cD2, cD1]
            
        except Exception as e:
            self.logger.warning(f"小波变换失败: {e}, 使用零填充")
            # 使用零填充
            zero_coeff = np.zeros((1, 100, 1))
            wavelet_coeffs = [zero_coeff] * 5
        
        return wavelet_coeffs, enhanced_features
    '''
    '''
    def _preprocess_blink_data(self, blink_data: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        预处理眨眼数据（HFENN格式）
        
        Args:
            blink_data: 原始眨眼数据
            
        Returns:
            小波系数列表和增强特征
        """
        # 提取增强特征
        enhanced_features = self.feature_extractor.extract_all_features(blink_data)
        
        # 验证并修复特征维度
        enhanced_features = self._validate_and_fix_features(enhanced_features)
        
        # 特征标准化
        if self.feature_scaler is not None and hasattr(self.feature_scaler, 'mean_'):
            enhanced_features = enhanced_features.reshape(1, -1)
            enhanced_features = self.feature_scaler.transform(enhanced_features)
            enhanced_features = enhanced_features.flatten()
        else:
            self.logger.warning("特征标准化器未fitted，跳过标准化")
        
        # 特征选择
        if self.selected_features is not None:
            if isinstance(self.selected_features, np.ndarray):
                if len(self.selected_features) == len(enhanced_features):
                    enhanced_features = enhanced_features[self.selected_features]
                else:
                    self.logger.warning(f"特征选择器维度不匹配：特征{len(enhanced_features)}个，选择器{len(self.selected_features)}个")
        
        # 小波变换（HFENN格式）
        try:
            coeffs = pywt.wavedec(blink_data, 'db4', level=4)
            
            # 重塑为模型输入格式
            cA4 = coeffs[0].reshape(1, -1, 1)
            cD4 = coeffs[1].reshape(1, -1, 1)
            cD3 = coeffs[2].reshape(1, -1, 1)
            cD2 = coeffs[3].reshape(1, -1, 1)
            cD1 = coeffs[4].reshape(1, -1, 1)
            
            wavelet_coeffs = [cA4, cD4, cD3, cD2, cD1]
            
        except Exception as e:
            self.logger.warning(f"小波变换失败: {e}, 使用零填充")
            # 使用零填充
            zero_coeff = np.zeros((1, 100, 1))
            wavelet_coeffs = [zero_coeff] * 5
        
        return wavelet_coeffs, enhanced_features
    '''
    def _preprocess_blink_data(self, blink_data: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        预处理眨眼数据（HFENN格式）
        """
        # 提取增强特征
        enhanced_features = self.feature_extractor.extract_all_features(blink_data)
        
        # 🔧 强制确保特征数量为50个（HFENN模型要求）
        if len(enhanced_features) < 50:
            padding_size = 50 - len(enhanced_features)
            padding = np.zeros(padding_size)
            enhanced_features = np.concatenate([enhanced_features, padding])
            self.logger.info(f"特征填充：从{len(enhanced_features)-padding_size}个增加到50个")
        elif len(enhanced_features) > 50:
            enhanced_features = enhanced_features[:50]
            self.logger.info(f"特征截断：从{len(enhanced_features)}个截断到50个")
        
        # 特征标准化
        if self.feature_scaler is not None and hasattr(self.feature_scaler, 'mean_'):
            # 🔧 确保scaler也是50维的
            if len(self.feature_scaler.mean_) != 50:
                self.logger.warning(f"重新创建50维标准化器")
                self.feature_scaler = self._create_fallback_scaler(target_features=50)
            
            enhanced_features = enhanced_features.reshape(1, -1)
            enhanced_features = self.feature_scaler.transform(enhanced_features)
            enhanced_features = enhanced_features.flatten()
        else:
            self.logger.warning("特征标准化器未fitted，跳过标准化")
        
        # 特征选择
        if self.selected_features is not None:
            if isinstance(self.selected_features, np.ndarray):
                if len(self.selected_features) <= len(enhanced_features):
                    enhanced_features = enhanced_features[self.selected_features]
                else:
                    self.logger.warning(f"特征选择器维度不匹配，跳过特征选择")
        
        # 小波变换（HFENN格式）
        try:
            coeffs = pywt.wavedec(blink_data, 'db4', level=4)
            
            # 重塑为模型输入格式
            cA4 = coeffs[0].reshape(1, -1, 1)
            cD4 = coeffs[1].reshape(1, -1, 1)
            cD3 = coeffs[2].reshape(1, -1, 1)
            cD2 = coeffs[3].reshape(1, -1, 1)
            cD1 = coeffs[4].reshape(1, -1, 1)
            
            wavelet_coeffs = [cA4, cD4, cD3, cD2, cD1]
            
        except Exception as e:
            self.logger.warning(f"小波变换失败: {e}, 使用零填充")
            zero_coeff = np.zeros((1, 100, 1))
            wavelet_coeffs = [zero_coeff] * 5
        
        return wavelet_coeffs, enhanced_features

    def _preprocess_pulse_data(self, pulse_data: np.ndarray) -> List[np.ndarray]:
        """
        预处理心率数据（WCNN格式）
        
        Args:
            pulse_data: 原始心率数据
            
        Returns:
            小波系数列表
        """
        try:
            # 小波变换（WCNN格式）- 使用4级分解以匹配训练时的结构
            coeffs = pywt.wavedec(pulse_data, 'db4', level=4)
            
            # 重塑为模型输入格式 - 按照训练时的顺序：cD1, cD2, cD3, cD4
            cD1 = coeffs[4].reshape(1, -1, 1)  # 第4级细节系数
            cD2 = coeffs[3].reshape(1, -1, 1)  # 第3级细节系数
            cD3 = coeffs[2].reshape(1, -1, 1)  # 第2级细节系数
            cD4 = coeffs[1].reshape(1, -1, 1)  # 第1级细节系数
            
            # 🔧 修复：按照训练时的实际输入形状进行填充
            target_lengths = [1305, 653, 327, 327]  # 对应模型的4个输入
            
            # 填充或截断到目标长度
            def pad_or_truncate(coeff, target_len):
                if coeff.shape[1] > target_len:
                    return coeff[:, :target_len, :]
                elif coeff.shape[1] < target_len:
                    padding = np.zeros((1, target_len - coeff.shape[1], 1))
                    return np.concatenate([coeff, padding], axis=1)
                else:
                    return coeff
            
            cD1 = pad_or_truncate(cD1, target_lengths[0])  # 1305
            cD2 = pad_or_truncate(cD2, target_lengths[1])  # 653
            cD3 = pad_or_truncate(cD3, target_lengths[2])  # 327
            cD4 = pad_or_truncate(cD4, target_lengths[3])  # 327
            
            return [cD1, cD2, cD3, cD4]
            
        except Exception as e:
            self.logger.warning(f"WCNN小波变换失败: {e}, 使用零填充")
            # 使用零填充 - 按照训练时的实际输入形状
            zero_coeffs = [
                np.zeros((1, 1305, 1)),  # Input 0
                np.zeros((1, 653, 1)),   # Input 1
                np.zeros((1, 327, 1)),   # Input 2
                np.zeros((1, 327, 1))    # Input 3
            ]
            return zero_coeffs
    
    def get_risk_driver_state(self, vehicle_id: str, simulation_time: float) -> Optional[DriverState]:
        """
        获取风险驾驶员状态
        
        Args:
            vehicle_id: 车辆ID
            simulation_time: 仿真时间
            
        Returns:
            驾驶员状态对象，如果失败则返回None
        """
        try:
            start_time = time.time()
            
            # 模拟驾驶员数据
            simulated_data = self._simulate_driver_data(vehicle_id, simulation_time)
            blink_data = simulated_data['blink_data']
            pulse_data = simulated_data['pulse_data']
            
            # 预处理数据
            hfenn_inputs = self._preprocess_blink_data(blink_data)
            wcnn_inputs = self._preprocess_pulse_data(pulse_data)
            
            # 模型预测
            fatigue_prediction = self._predict_fatigue(hfenn_inputs)
            emotion_prediction = self._predict_emotion(wcnn_inputs)
            
            if fatigue_prediction is None or emotion_prediction is None:
                return None
            
            # 计算综合风险等级
            risk_level = self._calculate_risk_level(fatigue_prediction, emotion_prediction)
            
            # 创建驾驶员状态对象
            driver_state = DriverState(
                vehicle_id=vehicle_id,
                timestamp=simulation_time,
                fatigue_level=fatigue_prediction['predicted_class'],
                emotion_state=emotion_prediction['predicted_class'],
                risk_level=risk_level,
                confidence=min(fatigue_prediction['confidence'], emotion_prediction['confidence']),
                blink_features=blink_data,
                pulse_features=pulse_data
            )
            
            # 缓存状态
            self.driver_states[vehicle_id] = driver_state
            self.last_update_time[vehicle_id] = simulation_time
            
            # 记录性能
            inference_time = time.time() - start_time
            self.logger.debug(f"车辆 {vehicle_id} 状态预测完成，耗时: {inference_time:.3f}s")
            
            return driver_state
            
        except Exception as e:
            self.logger.error(f"获取车辆 {vehicle_id} 状态失败: {e}")
            return None
    
    def _predict_fatigue(self, hfenn_inputs: Tuple[List[np.ndarray], np.ndarray]) -> Optional[Dict]:
        """疲劳预测（HFENN）"""
        if self.hfenn_model is None:
            self.logger.warning("HFENN模型未加载，无法进行疲劳预测")
            return None
        
        try:
            wavelet_coeffs, enhanced_features = hfenn_inputs
            
            # 准备模型输入
            model_inputs = wavelet_coeffs + [enhanced_features.reshape(1, -1)]
            
            # 预测
            predictions = self.hfenn_model.predict(model_inputs, verbose=0)
            
            # 解析结果
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': predictions[0]
            }
            
        except Exception as e:
            self.logger.error(f"疲劳预测失败: {e}")
            return None
    
    def _predict_emotion(self, wcnn_inputs: List[np.ndarray]) -> Optional[Dict]:
        """情绪预测（WCNN）"""
        if self.wcnn_model is None:
            self.logger.warning("WCNN模型未加载，无法进行情绪预测")
            return None
        
        try:
            # 预测
            predictions = self.wcnn_model.predict(wcnn_inputs, verbose=0)
            
            # 解析结果
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': predictions[0]
            }
            
        except Exception as e:
            self.logger.error(f"情绪预测失败: {e}")
            return None
    
    def _calculate_risk_level(self, fatigue_pred: Dict, emotion_pred: Dict) -> float:
        """计算综合风险等级"""
        # 疲劳等级权重
        fatigue_weights = {0: 0.0, 1: 0.3, 2: 0.7}
        fatigue_score = fatigue_weights.get(fatigue_pred['predicted_class'], 0.0)
        
        # 情绪状态权重
        emotion_score = emotion_pred['predicted_class'] * 0.3
        
        # 置信度调整
        confidence_adjustment = (fatigue_pred['confidence'] + emotion_pred['confidence']) / 2
        
        # 综合风险等级 (0.0-1.0)
        risk_level = (fatigue_score + emotion_score) * confidence_adjustment
        
        return np.clip(risk_level, 0.0, 1.0)
    
    def get_batch_driver_states(self, vehicle_ids: List[str], simulation_time: float) -> Dict[str, DriverState]:
        """批量获取多个车辆的驾驶员状态"""
        results = {}
        
        for vehicle_id in vehicle_ids:
            state = self.get_risk_driver_state(vehicle_id, simulation_time)
            if state:
                results[vehicle_id] = state
        
        return results
    
    def get_cached_driver_state(self, vehicle_id: str) -> Optional[DriverState]:
        """获取缓存的驾驶员状态"""
        return self.driver_states.get(vehicle_id)
    
    def clear_cache(self):
        """清除状态缓存"""
        self.driver_states.clear()
        self.last_update_time.clear()
        self.logger.info("状态缓存已清除")
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        return {
            'hfenn_loaded': self.hfenn_model is not None,
            'wcnn_loaded': self.wcnn_model is not None,
            'feature_scaler_loaded': self.feature_scaler is not None,
            'selected_features_loaded': self.selected_features is not None,
            'cached_vehicles': len(self.driver_states),
            'last_update_times': self.last_update_time.copy()
        }

    def _validate_and_fix_features(self, features: np.ndarray) -> np.ndarray:
        """
        验证并修复特征维度不匹配问题
        
        Args:
            features: 输入特征数组
            
        Returns:
            修正后的特征数组
        """
        if self.feature_scaler is None:
            return features
        
        # 获取期望的特征数量
        if hasattr(self.feature_scaler, 'n_features_in_'):
            expected_features = self.feature_scaler.n_features_in_
        elif hasattr(self.feature_scaler, 'mean_'):
            expected_features = len(self.feature_scaler.mean_)
        else:
            # 如果scaler未fitted，返回原始特征
            self.logger.warning("StandardScaler未fitted，无法验证特征维度")
            return features
        
        current_features = len(features)
        
        if current_features == expected_features:
            return features
        elif current_features < expected_features:
            # 特征不足，用零填充
            padding_size = expected_features - current_features
            padding = np.zeros(padding_size)
            fixed_features = np.concatenate([features, padding])
            self.logger.warning(f"特征不足：当前{current_features}个，期望{expected_features}个，用{padding_size}个零填充")
            return fixed_features
        else:
            # 特征过多，截断
            fixed_features = features[:expected_features]
            self.logger.warning(f"特征过多：当前{current_features}个，期望{expected_features}个，截断至{expected_features}个")
            return fixed_features
    
    def debug_feature_extraction(self, signal: np.ndarray) -> Dict[str, int]:
        """调试特征提取，返回各类特征的数量"""
        time_features = self.feature_extractor.extract_time_domain_features(signal)
        freq_features = self.feature_extractor.extract_frequency_domain_features(signal)
        wavelet_features = self.feature_extractor.extract_wavelet_features(signal)
        nonlinear_features = self.feature_extractor.extract_nonlinear_features(signal)
        all_features = self.feature_extractor.extract_all_features(signal)
        
        feature_counts = {
            'time_features': len(time_features),
            'freq_features': len(freq_features),
            'wavelet_features': len(wavelet_features),
            'nonlinear_features': len(nonlinear_features),
            'total_features': len(all_features),
            'expected_total': len(time_features) + len(freq_features) + len(wavelet_features) + len(nonlinear_features)
        }
        
        self.logger.info(f"特征数量调试: {feature_counts}")
        return feature_counts
    
    def test_feature_dimensions(self):
        """测试特征维度一致性"""
        test_signal = np.random.randn(2610)
        
        # 调试特征提取
        feature_info = self.debug_feature_extraction(test_signal)
        
        # 测试预处理
        try:
            wavelet_coeffs, enhanced_features = self._preprocess_blink_data(test_signal)
            self.logger.info(f"✅ 预处理成功，增强特征维度: {enhanced_features.shape}")
            return True
        except Exception as e:
            self.logger.error(f"❌ 预处理失败: {e}")
            return False


def main():
    """主函数 - 演示DNSR集成模块"""
    print("🚗🤖 DNSR集成模块演示")
    print("="*50)
    
    # 创建DNSR集成模块
    dnsr = DNSRIntegration()
    
    # 测试特征维度
    print("🔍 测试特征维度...")
    if dnsr.test_feature_dimensions():
        print("✅ 特征维度测试通过")
    else:
        print("❌ 特征维度测试失败")
    
    # 检查系统状态
    status = dnsr.get_system_status()
    print(f"系统状态: {status}")
    
    # 模拟多个车辆
    vehicle_ids = ["vehicle_001", "vehicle_002", "vehicle_003"]
    simulation_time = 100.0
    
    print(f"\n📊 获取车辆状态 (仿真时间: {simulation_time}s):")
    
    for vehicle_id in vehicle_ids:
        state = dnsr.get_risk_driver_state(vehicle_id, simulation_time)
        if state:
            print(f"车辆 {vehicle_id}:")
            print(f"  疲劳等级: {state.fatigue_level}")
            print(f"  情绪状态: {state.emotion_state}")
            print(f"  风险等级: {state.risk_level:.3f}")
            print(f"  置信度: {state.confidence:.3f}")
        else:
            print(f"车辆 {vehicle_id}: 状态获取失败")
    
    # 批量获取状态
    print(f"\n🔄 批量获取状态:")
    batch_states = dnsr.get_batch_driver_states(vehicle_ids, simulation_time + 10)
    print(f"成功获取 {len(batch_states)} 个车辆状态")
    
    print("\n✅ 演示完成!")

if __name__ == "__main__":
    main() 