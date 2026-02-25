"""
Enhanced HFENN 疲劳检测系统集成代码0905
修复版本 - 支持更好的测试和错误处理
"""

import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 信号处理
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import skew, kurtosis
import pywt

# 机器学习
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

# 深度学习
try:
    from keras.models import load_model
    print("使用独立的 keras")
except ImportError:
    from tensorflow.keras.models import load_model
    print("使用 tensorflow.keras")


class EnhancedFeatureExtractor:
    """增强版特征提取器 - 修复边界情况处理"""
    
    def __init__(self, sampling_rate=87):
        self.sampling_rate = sampling_rate
        
    def _safe_statistical_feature(self, signal, func, default=0.0):
        """安全计算统计特征"""
        try:
            result = func(signal)
            return result if np.isfinite(result) else default
        except:
            return default
        
    def extract_time_domain_features(self, signal):
        """时域特征 - 增强的异常处理"""
        features = []
        
        # 基础统计特征
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.median(signal),
            self._safe_statistical_feature(signal, skew),  # 安全的偏度计算
        ])
        
        # 扩展时域特征
        features.extend([
            self._safe_statistical_feature(signal, kurtosis),  # 安全的峰度计算
            np.var(signal),
            np.ptp(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            len(find_peaks(signal)[0]),
            len(find_peaks(-signal)[0]),
        ])
        
        # 信号能量特征
        features.extend([
            np.sum(signal**2),
            np.mean(signal**2),
            np.sqrt(np.mean(signal**2)),
        ])
        
        # 确保所有特征都是有限的
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
    
    def extract_frequency_domain_features(self, signal):
        """频域特征 - 增强的异常处理"""
        features = []
        
        try:
            # 功率谱密度
            freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
            
            # 避免除零错误
            if len(psd) == 0 or np.sum(psd) == 0:
                return [0.0] * 9  # 返回9个零特征
            
            # 频域统计特征
            features.extend([
                np.mean(psd),
                np.std(psd),
                np.max(psd),
                freqs[np.argmax(psd)],
                np.sum(psd),
            ])
            
            # 频带功率分布
            freq_bands = [(0, 1), (1, 5), (5, 15), (15, 25)]
            total_power = np.sum(psd)
            
            for low, high in freq_bands:
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.sum(psd[band_mask])
                features.append(band_power / total_power if total_power > 0 else 0)
                
        except Exception as e:
            # 如果频域分析失败，返回零特征
            features = [0.0] * 9
        
        # 确保所有特征都是有限的
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
    
    def extract_wavelet_features(self, signal):
        """小波特征 - 增强的异常处理"""
        features = []
        
        try:
            # 多层小波分解
            coeffs = pywt.wavedec(signal, 'db4', level=min(4, pywt.dwt_max_level(len(signal), 'db4')))
            
            # 各层系数的统计特征
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        np.sum(coeff**2),
                        np.max(np.abs(coeff)),
                    ])
                    
        except Exception as e:
            # 如果小波分析失败，返回一些默认特征
            features = [0.0] * 20
        
        # 确保所有特征都是有限的
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
    
    def extract_nonlinear_features(self, signal):
        """非线性动态特征 - 增强的异常处理"""
        features = []
        
        try:
            # 希尔伯特变换
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.angle(analytic_signal)
            
            features.extend([
                np.mean(amplitude_envelope),
                np.std(amplitude_envelope),
                np.mean(np.diff(instantaneous_phase)),
            ])
            
            # 过零率
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
            features.append(zero_crossings)
            
            # 简化的近似熵
            try:
                m, r = 2, 0.2 * np.std(signal)
                N = len(signal)
                
                if N > 20 and r > 0:
                    patterns = []
                    for i in range(N - m):
                        patterns.append(signal[i:i+m])
                    
                    phi_m = 0
                    for i in range(len(patterns)):
                        matches = sum(1 for j in range(len(patterns)) 
                                    if max(abs(patterns[i][k] - patterns[j][k]) for k in range(m)) <= r)
                        if matches > 0:
                            phi_m += np.log(matches / len(patterns))
                    
                    phi_m /= len(patterns)
                    features.append(phi_m)
                else:
                    features.append(0)
            except:
                features.append(0)
                
        except Exception as e:
            features = [0.0] * 5
        
        # 确保所有特征都是有限的
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
    
    def extract_all_features(self, signal):
        """提取所有特征 - 增强的异常处理"""
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
        
        # 最终安全检查
        all_features = [f if np.isfinite(f) else 0.0 for f in all_features]
        
        return np.array(all_features)


class EnhancedHFENNIntegrator:
    """Enhanced HFENN 疲劳检测集成器 - 修复版本"""
    
    def __init__(self, 
                 model_path: str = "Enhanced_HFENN_best.keras",
                 scaler_path: str = "feature_scaler.pkl",
                 selector_path: str = "selected_feature_indices.pkl",
                 window_size: int = 2610,
                 log_level: int = logging.INFO,
                 auto_load: bool = True):  # 🔧 新增：控制是否自动加载
        """
        初始化Enhanced HFENN集成器
        
        Args:
            model_path: 模型文件路径
            scaler_path: 特征标准化器路径
            selector_path: 特征选择器路径
            window_size: 窗口大小（与训练时一致）
            log_level: 日志级别
            auto_load: 是否自动加载所有组件（测试时设为False）
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.selector_path = Path(selector_path)
        self.window_size = window_size
        
        # 设置日志
        self.logger = self._setup_logger(log_level)
        
        # 初始化组件
        self.feature_extractor = EnhancedFeatureExtractor()
        self.model = None
        self.scaler = None
        self.feature_selector_indices = None
        
        # 🔧 条件加载 - 支持测试模式
        if auto_load:
            self._load_all_components()
        
        # 类别映射
        self.class_mapping = {
            0: "Normal",
            1: "Mild Fatigue", 
            2: "Fatigue"
        }
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("EnhancedHFENN")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_all_components(self):
        """加载所有必需组件"""
        try:
            # 加载模型
            self._load_model()
            
            # 加载预处理器
            self._load_preprocessors()
            
            self.logger.info("所有组件加载完成")
            
        except Exception as e:
            self.logger.error(f"组件加载失败: {e}")
            raise
    
    def _load_model(self):
        """加载Enhanced HFENN模型"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
            self.model = load_model(str(self.model_path))
            self.logger.info(f"模型加载成功: {self.model_path}")
            self.logger.info(f"模型输入数量: {len(self.model.inputs)}")
            
            for i, input_layer in enumerate(self.model.inputs):
                self.logger.info(f"输入 {i}: {input_layer.name}, shape: {input_layer.shape}")
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_preprocessors(self):
        """加载预处理器"""
        try:
            # 加载特征标准化器
            if self.scaler_path.exists():
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"特征标准化器加载成功: {self.scaler_path}")
            else:
                self.logger.warning(f"特征标准化器文件不存在: {self.scaler_path}")
                
            # 加载特征选择器
            if self.selector_path.exists():
                with open(self.selector_path, 'rb') as f:
                    self.feature_selector_indices = pickle.load(f)
                self.logger.info(f"特征选择器加载成功，选择 {len(self.feature_selector_indices)} 个特征")
            else:
                self.logger.warning(f"特征选择器文件不存在: {self.selector_path}")
                
        except Exception as e:
            self.logger.error(f"预处理器加载失败: {e}")
            # 继续执行，但会使用备用方案
    
    # 🔧 新增：手动加载方法，支持测试
    def load_model_from_object(self, model_object):
        """从对象加载模型（用于测试）"""
        self.model = model_object
        self.logger.info("从对象加载模型成功")
    
    def load_scaler_from_object(self, scaler_object):
        """从对象加载标准化器（用于测试）"""
        self.scaler = scaler_object
        self.logger.info("从对象加载标准化器成功")
    
    def load_selector_from_object(self, selector_indices):
        """从对象加载特征选择器（用于测试）"""
        self.feature_selector_indices = selector_indices
        self.logger.info("从对象加载特征选择器成功")
    
    # 其余方法保持不变...
    def _preprocess_signal(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """预处理信号数据"""
        try:
            # 确保信号长度正确
            if len(signal) != self.window_size:
                self.logger.warning(f"信号长度 {len(signal)} 不等于期望的 {self.window_size}")
                if len(signal) > self.window_size:
                    signal = signal[:self.window_size]
                else:
                    padding = np.zeros(self.window_size - len(signal))
                    signal = np.concatenate([signal, padding])
                    
            self.logger.debug(f"预处理信号长度: {len(signal)}")
            
            # 1. 小波变换
            wavelet_coeffs = self._perform_wavelet_transform(signal)
            
            # 2. 特征提取
            enhanced_features = self._extract_and_process_features(signal)
            
            return wavelet_coeffs, enhanced_features
            
        except Exception as e:
            self.logger.error(f"信号预处理失败: {e}")
            raise
    
    def _perform_wavelet_transform(self, signal: np.ndarray) -> List[np.ndarray]:
        """执行小波变换"""
        try:
            coeffs = pywt.wavedec(signal.reshape(1, 1, -1), 'db4', level=4)
            
            cA4 = coeffs[0].reshape((coeffs[0].shape[0], coeffs[0].shape[2], 1))
            cD4 = coeffs[1].reshape((coeffs[1].shape[0], coeffs[1].shape[2], 1))
            cD3 = coeffs[2].reshape((coeffs[2].shape[0], coeffs[2].shape[2], 1))
            cD2 = coeffs[3].reshape((coeffs[3].shape[0], coeffs[3].shape[2], 1))
            cD1 = coeffs[4].reshape((coeffs[4].shape[0], coeffs[4].shape[2], 1))
            
            self.logger.debug(f"小波系数形状: cA4={cA4.shape}, cD4={cD4.shape}, "
                            f"cD3={cD3.shape}, cD2={cD2.shape}, cD1={cD1.shape}")
            
            return [cA4, cD4, cD3, cD2, cD1]
            
        except Exception as e:
            self.logger.error(f"小波变换失败: {e}")
            raise
    
    def _extract_and_process_features(self, signal: np.ndarray) -> np.ndarray:
        """提取和处理特征"""
        try:
            # 1. 提取所有特征
            raw_features = self.feature_extractor.extract_all_features(signal)
            self.logger.debug(f"原始特征数量: {len(raw_features)}")
            
            # 2. 处理异常值
            raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 3. 标准化
            if self.scaler is not None:
                features_scaled = self.scaler.transform(raw_features.reshape(1, -1))
                features_scaled = features_scaled.flatten()
                self.logger.debug("特征标准化完成")
            else:
                features_scaled = raw_features
                self.logger.warning("未找到标准化器，跳过标准化")
            
            # 4. 特征选择
            if self.feature_selector_indices is not None:
                valid_indices = self.feature_selector_indices[
                    self.feature_selector_indices < len(features_scaled)
                ]
                selected_features = features_scaled[valid_indices]
                self.logger.debug(f"特征选择完成，选择了 {len(selected_features)} 个特征")
            else:
                selected_features = features_scaled[:50]
                self.logger.warning("未找到特征选择器，使用前50个特征")
            
            # 5. 确保特征数量为50
            target_feature_count = 50
            if len(selected_features) < target_feature_count:
                padding = np.zeros(target_feature_count - len(selected_features))
                selected_features = np.concatenate([selected_features, padding])
                self.logger.warning(f"特征数量不足，填充至 {target_feature_count} 个")
            elif len(selected_features) > target_feature_count:
                selected_features = selected_features[:target_feature_count]
                self.logger.warning(f"特征数量过多，截断至 {target_feature_count} 个")
            
            # 6. 重塑为模型期望的形状
            final_features = selected_features.reshape(1, -1)
            self.logger.debug(f"最终特征形状: {final_features.shape}")
            
            return final_features
            
        except Exception as e:
            self.logger.error(f"特征处理失败: {e}")
            raise
    
    def predict_fatigue(self, signal: np.ndarray) -> Dict[str, Any]:
        """执行疲劳预测"""
        try:
            if self.model is None:
                raise RuntimeError("模型未加载")
                
            # 预处理
            wavelet_coeffs, enhanced_features = self._preprocess_signal(signal)
            
            # 准备模型输入
            model_inputs = wavelet_coeffs + [enhanced_features]
            
            # 记录输入信息
            self.logger.debug("模型输入准备完成:")
            for i, inp in enumerate(model_inputs):
                self.logger.debug(f"  输入 {i}: {inp.shape}")
            
            # 预测
            predictions = self.model.predict(model_inputs, verbose=0)
            
            # 解析结果
            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0]))
            probabilities = predictions[0].tolist()
            
            result = {
                'predicted_class': predicted_class,
                'predicted_label': self.class_mapping[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    self.class_mapping[i]: prob 
                    for i, prob in enumerate(probabilities)
                },
                'raw_probabilities': probabilities,
                'success': True,
                'message': '预测成功'
            }
            
            self.logger.info(f"预测完成: {result['predicted_label']} (置信度: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            error_msg = f"疲劳预测失败: {e}"
            self.logger.error(error_msg)
            
            return {
                'predicted_class': None,
                'predicted_label': None,
                'confidence': 0.0,
                'probabilities': {},
                'raw_probabilities': [],
                'success': False,
                'message': error_msg
            }
    
    def validate_setup(self) -> Dict[str, bool]:
        """验证系统设置"""
        validation_result = {}
        
        # 检查模型
        validation_result['model_loaded'] = self.model is not None
        
        # 检查预处理器
        validation_result['scaler_loaded'] = self.scaler is not None
        validation_result['selector_loaded'] = self.feature_selector_indices is not None
        
        # 测试预测
        try:
            test_signal = np.random.randn(self.window_size)
            result = self.predict_fatigue(test_signal)
            validation_result['prediction_test'] = result['success']
        except Exception as e:
            validation_result['prediction_test'] = False
            self.logger.error(f"预测测试失败: {e}")
        
        # 汇总
        all_valid = all(validation_result.values())
        validation_result['overall'] = all_valid
        
        self.logger.info(f"系统验证结果: {validation_result}")
        
        return validation_result


def main():
    """示例使用"""
    print("Enhanced HFENN 疲劳检测系统")
    print("=" * 50)
    
    try:
        # 创建集成器
        integrator = EnhancedHFENNIntegrator(
            model_path="Enhanced_HFENN_best.keras",
            scaler_path="feature_scaler.pkl", 
            selector_path="selected_feature_indices.pkl"
        )
        
        # 验证设置
        validation = integrator.validate_setup()
        if not validation['overall']:
            print("系统验证失败，请检查以下问题:")
            for key, value in validation.items():
                if not value and key != 'overall':
                    print(f"  - {key}: {'通过' if value else '失败'}")
            return
        
        print("系统验证通过！")
        
        # 示例预测
        print("\n执行示例预测...")
        test_signal = np.random.randn(2610)
        
        result = integrator.predict_fatigue(test_signal)
        
        if result['success']:
            print(f"预测结果: {result['predicted_label']}")
            print(f"置信度: {result['confidence']:.3f}")
            print("各类别概率:")
            for label, prob in result['probabilities'].items():
                print(f"  {label}: {prob:.3f}")
        else:
            print(f"预测失败: {result['message']}")
            
    except Exception as e:
        print(f"系统初始化失败: {e}")


if __name__ == "__main__":
    main()
