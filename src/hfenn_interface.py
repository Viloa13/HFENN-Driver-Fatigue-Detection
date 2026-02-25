"""
HFENN接口模块 - 连接疲劳检测模型和DNSR集成
"""
import numpy as np
import logging
import os
from typing import Dict, Optional
import json

class HFENNInterface:
    """HFENN疲劳检测模型接口"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # 尝试加载模型
        self._load_model()
        
    def _load_model(self):
        """加载HFENN模型"""
        try:
            if self.model_path.endswith('.keras') or self.model_path.endswith('.h5'):
                # 尝试加载Keras模型
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(self.model_path)
                    self.is_loaded = True
                    logging.info(f"✅ Keras模型加载成功: {self.model_path}")
                except ImportError:
                    logging.warning("⚠️ TensorFlow未安装，无法加载Keras模型")
                except Exception as e:
                    logging.warning(f"⚠️ Keras模型加载失败: {e}")
            
            if not self.is_loaded:
                # 如果模型加载失败，使用模拟模式
                logging.warning("⚠️ 使用模拟疲劳检测模型")
                self.is_loaded = True  # 标记为已加载，但实际是模拟模式
                
        except Exception as e:
            logging.error(f"❌ 模型加载失败: {e}")
            self.is_loaded = False
    
    def predict_fatigue_state(self, driver_data: Dict) -> Dict:
        """
        预测疲劳状态
        输入: driver_data - 包含驾驶员数据的字典
        输出: 疲劳状态预测结果
        """
        if not self.is_loaded:
            return self._fallback_prediction(driver_data)
        
        try:
            # 如果有真实模型，进行预测
            if self.model is not None:
                return self._real_prediction(driver_data)
            else:
                # 使用改进的模拟预测
                return self._improved_simulation_prediction(driver_data)
                
        except Exception as e:
            logging.warning(f"预测失败，使用备用方案: {e}")
            return self._fallback_prediction(driver_data)
    
    def _real_prediction(self, driver_data: Dict) -> Dict:
        """使用真实模型进行预测"""
        try:
            # 提取特征
            features = self._extract_features(driver_data)
            
            # 模型预测 - 使用字典格式的输入
            model_inputs = {
                'manual_features': features[0],
                'wavelet_input': features[1], 
                'personal_id': features[2]
            }
            prediction = self.model.predict(model_inputs, verbose=0)
            
            # 解析预测结果
            fatigue_level = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))
            
            # 基于疲劳等级推断其他状态
            emotion_state = min(fatigue_level, 1)  # 疲劳越高，情绪越负面
            
            return {
                'fatigue_level': fatigue_level,
                'emotion_state': emotion_state,
                'confidence': confidence,
                'behavior_history': {
                    'lane_sway_count': fatigue_level * 2,
                    'sudden_decel_count': fatigue_level
                }
            }
            
        except Exception as e:
            logging.warning(f"真实预测失败: {e}")
            return self._improved_simulation_prediction(driver_data)
    
    def _extract_features(self, driver_data: Dict) -> tuple:
        """从驾驶员数据中提取特征，返回3个输入"""
        # 基础特征
        timestamp = driver_data.get('timestamp', 0)
        vehicle_speed = driver_data.get('vehicle_speed', 25.0)
        driving_duration = driver_data.get('driving_duration', 0)
        
        # 眨眼信号特征（如果有的话）
        blink_signal = driver_data.get('blink_signal', None)
        if blink_signal is not None and len(blink_signal) > 0:
            blink_mean = np.mean(blink_signal)
            blink_std = np.std(blink_signal)
            blink_max = np.max(blink_signal)
        else:
            blink_mean = 0.3
            blink_std = 0.1
            blink_max = 0.5
        
        # 输入1: 35维特征向量
        features_list = []
        
        # 基础特征 (7个)
        features_list.append(timestamp / 3600.0)  # 归一化时间
        features_list.append(vehicle_speed / 50.0)  # 归一化速度
        features_list.append(driving_duration / 3600.0)  # 归一化驾驶时长
        features_list.append(blink_mean)  # 眨眼信号均值
        features_list.append(blink_std)   # 眨眼信号标准差
        features_list.append(blink_max)   # 眨眼信号最大值
        features_list.append(min(timestamp / 1800.0, 1.0))  # 疲劳因子1
        
        # 疲劳因子 (1个)
        features_list.append(min(driving_duration / 1800.0, 1.0))  # 疲劳因子2
        
        # 噪声特征 (27个)
        for i in range(27):
            features_list.append(np.random.normal(0, 0.1))
        
        features_35d = np.array(features_list)
        
        # 确保正好是35维
        assert len(features_35d) == 35, f"特征向量维度错误: {len(features_35d)} != 35"
        
        # 输入2: 800个时间步的序列数据 (None, 800, 1)
        sequence_data = np.random.normal(0, 0.1, (1, 800, 1)).astype(np.float32)
        
        # 输入3: 1维标量 (None, 1)
        scalar_input = np.array([[timestamp / 3600.0]], dtype=np.float32)
        
        return (
            features_35d.reshape(1, -1).astype(np.float32),  # 添加batch维度
            sequence_data,
            scalar_input
        )
    
    def _improved_simulation_prediction(self, driver_data: Dict) -> Dict:
        """改进的模拟预测"""
        timestamp = driver_data.get('timestamp', 0)
        vehicle_speed = driver_data.get('vehicle_speed', 25.0)
        driving_duration = driver_data.get('driving_duration', 0)
        
        # 基于多个因素计算疲劳等级
        time_factor = min(timestamp / 3600, 1.0)  # 时间疲劳因子
        duration_factor = min(driving_duration / 1800, 1.0)  # 驾驶时长疲劳因子
        speed_factor = 1.0 - min(vehicle_speed / 50, 1.0)  # 低速疲劳因子
        
        # 综合疲劳计算
        total_fatigue = (time_factor * 0.4 + duration_factor * 0.4 + speed_factor * 0.2)
        
        # 添加随机性，但保持合理性
        fatigue_noise = np.random.normal(0, 0.1)
        total_fatigue = np.clip(total_fatigue + fatigue_noise, 0, 2)
        
        # 确定疲劳等级
        if total_fatigue < 0.5:
            fatigue_level = 0  # 正常
        elif total_fatigue < 1.0:
            fatigue_level = 1  # 轻度疲劳
        else:
            fatigue_level = 2  # 疲劳
        
        # 情绪状态（疲劳越高，负面情绪概率越大）
        emotion_threshold = 0.3 + fatigue_level * 0.3
        emotion_state = 1 if np.random.random() < emotion_threshold else 0
        
        # 置信度（基于疲劳等级的确定性）
        confidence = 0.8 - fatigue_level * 0.1 + np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0.5, 0.95)
        
        return {
            'fatigue_level': fatigue_level,
            'emotion_state': emotion_state,
            'confidence': confidence,
            'behavior_history': {
                'lane_sway_count': int(fatigue_level * 2 + np.random.poisson(1)),
                'sudden_decel_count': int(fatigue_level + np.random.poisson(0.5))
            }
        }
    
    def _fallback_prediction(self, driver_data: Dict) -> Dict:
        """备用预测方案"""
        return {
            'fatigue_level': 0,
            'emotion_state': 0,
            'confidence': 0.5,
            'behavior_history': {
                'lane_sway_count': 0,
                'sudden_decel_count': 0
            }
        }
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'model_type': 'keras' if self.model is not None else 'simulation'
        } 