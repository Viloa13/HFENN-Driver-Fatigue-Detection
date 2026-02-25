#!/usr/bin/env python3
"""
DNSR集成模块单元测试
测试DNSRIntegration类的所有主要功能

作者: AI Assistant
版本: v1.1
日期: 2025-0904
"""

import unittest
import numpy as np
import tempfile
import os
import pickle
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 导入被测试的模块
from dnsr_integration import DNSRIntegration, EnhancedFeatureExtractor, DriverState

class TestEnhancedFeatureExtractor(unittest.TestCase):
    """测试增强特征提取器"""
    
    def setUp(self):
        """测试前准备"""
        self.extractor = EnhancedFeatureExtractor(sampling_rate=87)
        self.test_signal = np.random.randn(1000)
    
    def test_extract_time_domain_features(self):
        """测试时域特征提取"""
        features = self.extractor.extract_time_domain_features(self.test_signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 16)  # 16个时域特征
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_extract_frequency_domain_features(self):
        """测试频域特征提取"""
        features = self.extractor.extract_frequency_domain_features(self.test_signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 9)  # 9个频域特征
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_extract_wavelet_features(self):
        """测试小波特征提取"""
        features = self.extractor.extract_wavelet_features(self.test_signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_extract_nonlinear_features(self):
        """测试非线性特征提取"""
        features = self.extractor.extract_nonlinear_features(self.test_signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), 4)  # 4个非线性特征
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_extract_all_features(self):
        """测试所有特征提取"""
        features = self.extractor.extract_all_features(self.test_signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_empty_signal(self):
        """测试空信号处理"""
        empty_signal = np.array([])
        
        with self.assertRaises(Exception):
            self.extractor.extract_time_domain_features(empty_signal)

class TestDNSRIntegration(unittest.TestCase):
    """测试DNSR集成模块"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟模型文件
        self.hfenn_path = os.path.join(self.temp_dir, "test_hfenn.keras")
        self.wcnn_path = os.path.join(self.temp_dir, "test_wcnn.h5")
        self.scaler_path = os.path.join(self.temp_dir, "test_scaler.pkl")
        self.features_path = os.path.join(self.temp_dir, "test_features.pkl")
        
        # 创建模拟文件
        self._create_mock_files()
        
        # 创建DNSR集成模块实例
        self.dnsr = DNSRIntegration(
            hfenn_model_path=self.hfenn_path,
            wcnn_model_path=self.wcnn_path,
            feature_scaler_path=self.scaler_path,
            selected_features_path=self.features_path
        )
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_files(self):
        """创建模拟文件"""
        # 创建真实的特征标准化器并拟合
        scaler = StandardScaler()
        # 创建一些模拟数据进行拟合
        mock_data = np.random.randn(100, 50)  # 100个样本，50个特征
        scaler.fit(mock_data)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 创建模拟特征选择器
        selected_features = np.array([True] * 50)  # 50个特征
        with open(self.features_path, 'wb') as f:
            pickle.dump(selected_features, f)
    
    @patch('dnsr_integration.load_model')
    def test_initialization_with_models(self, mock_load_model):
        """测试带模型初始化"""
        # 模拟模型加载
        mock_hfenn = Mock()
        mock_wcnn = Mock()
        mock_load_model.side_effect = [mock_hfenn, mock_wcnn]
        
        dnsr = DNSRIntegration(
            hfenn_model_path="test.keras",
            wcnn_model_path="test.h5"
        )
        
        self.assertIsNotNone(dnsr.hfenn_model)
        self.assertIsNotNone(dnsr.wcnn_model)
    
    def test_initialization_without_models(self):
        """测试无模型初始化"""
        dnsr = DNSRIntegration()
        
        # 应该能够初始化，但模型为None
        self.assertIsNone(dnsr.hfenn_model)
        self.assertIsNone(dnsr.wcnn_model)
    
    def test_simulate_driver_data(self):
        """测试驾驶员数据模拟"""
        vehicle_id = "test_vehicle"
        timestamp = 100.0
        
        data = self.dnsr._simulate_driver_data(vehicle_id, timestamp)
        
        self.assertIn('blink_data', data)
        self.assertIn('pulse_data', data)
        self.assertEqual(len(data['blink_data']), self.dnsr.hfenn_config['window_size'])
        self.assertEqual(len(data['pulse_data']), self.dnsr.wcnn_config['window_size'])
        
        # 检查数据范围
        self.assertTrue(np.all((data['blink_data'] >= 0) & (data['blink_data'] <= 1)))
        self.assertTrue(np.all((data['pulse_data'] >= 0) & (data['pulse_data'] <= 1)))
    
    def test_preprocess_blink_data(self):
        """测试眨眼数据预处理"""
        blink_data = np.random.rand(2610)
        
        wavelet_coeffs, enhanced_features = self.dnsr._preprocess_blink_data(blink_data)
        
        # 检查小波系数
        self.assertEqual(len(wavelet_coeffs), 5)  # 5层小波分解
        for coeff in wavelet_coeffs:
            self.assertEqual(coeff.shape[0], 1)  # batch size
            self.assertEqual(coeff.shape[2], 1)  # channels
        
        # 检查增强特征
        self.assertIsInstance(enhanced_features, np.ndarray)
        self.assertTrue(len(enhanced_features) > 0)
    
    def test_preprocess_pulse_data(self):
        """测试心率数据预处理"""
        pulse_data = np.random.rand(2610)
        
        wavelet_coeffs = self.dnsr._preprocess_pulse_data(pulse_data)
        
        # 检查小波系数
        self.assertEqual(len(wavelet_coeffs), 4)  # 4层小波分解
        for coeff in wavelet_coeffs:
            self.assertEqual(coeff.shape[0], 1)  # batch size
            self.assertEqual(coeff.shape[2], 1)  # channels
    
    @patch.object(DNSRIntegration, '_predict_fatigue')
    @patch.object(DNSRIntegration, '_predict_emotion')
    def test_get_risk_driver_state_success(self, mock_emotion, mock_fatigue):
        """测试成功获取驾驶员状态"""
        # 模拟预测结果
        mock_fatigue.return_value = {
            'predicted_class': 1,
            'confidence': 0.85
        }
        mock_emotion.return_value = {
            'predicted_class': 0,
            'confidence': 0.90
        }
        
        vehicle_id = "test_vehicle"
        simulation_time = 100.0
        
        state = self.dnsr.get_risk_driver_state(vehicle_id, simulation_time)
        
        self.assertIsNotNone(state)
        self.assertEqual(state.vehicle_id, vehicle_id)
        self.assertEqual(state.fatigue_level, 1)
        self.assertEqual(state.emotion_state, 0)
        self.assertTrue(0.0 <= state.risk_level <= 1.0)
        self.assertTrue(0.0 <= state.confidence <= 1.0)
    
    @patch.object(DNSRIntegration, '_predict_fatigue')
    def test_get_risk_driver_state_fatigue_failure(self, mock_fatigue):
        """测试疲劳预测失败的情况"""
        mock_fatigue.return_value = None
        
        vehicle_id = "test_vehicle"
        simulation_time = 100.0
        
        state = self.dnsr.get_risk_driver_state(vehicle_id, simulation_time)
        
        self.assertIsNone(state)
    
    def test_calculate_risk_level(self):
        """测试风险等级计算"""
        fatigue_pred = {'predicted_class': 2, 'confidence': 0.8}
        emotion_pred = {'predicted_class': 1, 'confidence': 0.9}
        
        risk_level = self.dnsr._calculate_risk_level(fatigue_pred, emotion_pred)
        
        self.assertTrue(0.0 <= risk_level <= 1.0)
        self.assertIsInstance(risk_level, float)
    
    def test_get_batch_driver_states(self):
        """测试批量获取驾驶员状态"""
        vehicle_ids = ["vehicle_1", "vehicle_2", "vehicle_3"]
        simulation_time = 100.0
        
        # 模拟成功获取状态
        with patch.object(self.dnsr, 'get_risk_driver_state') as mock_get:
            mock_state = Mock(spec=DriverState)
            mock_get.return_value = mock_state
            
            results = self.dnsr.get_batch_driver_states(vehicle_ids, simulation_time)
            
            self.assertEqual(len(results), 3)
            self.assertEqual(mock_get.call_count, 3)
    
    def test_get_cached_driver_state(self):
        """测试获取缓存的驾驶员状态"""
        vehicle_id = "test_vehicle"
        
        # 初始状态为空
        cached_state = self.dnsr.get_cached_driver_state(vehicle_id)
        self.assertIsNone(cached_state)
        
        # 添加状态到缓存
        mock_state = Mock(spec=DriverState)
        self.dnsr.driver_states[vehicle_id] = mock_state
        
        cached_state = self.dnsr.get_cached_driver_state(vehicle_id)
        self.assertIsNotNone(cached_state)
    
    def test_clear_cache(self):
        """测试清除缓存"""
        # 添加一些测试数据
        self.dnsr.driver_states["test1"] = Mock()
        self.dnsr.driver_states["test2"] = Mock()
        self.dnsr.last_update_time["test1"] = 100.0
        
        self.dnsr.clear_cache()
        
        self.assertEqual(len(self.dnsr.driver_states), 0)
        self.assertEqual(len(self.dnsr.last_update_time), 0)
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.dnsr.get_system_status()
        
        self.assertIn('hfenn_loaded', status)
        self.assertIn('wcnn_loaded', status)
        self.assertIn('feature_scaler_loaded', status)
        self.assertIn('selected_features_loaded', status)
        self.assertIn('cached_vehicles', status)
        self.assertIn('last_update_times', status)
        
        self.assertIsInstance(status['cached_vehicles'], int)
        self.assertIsInstance(status['last_update_times'], dict)

class TestDriverState(unittest.TestCase):
    """测试驾驶员状态数据结构"""
    
    def test_driver_state_creation(self):
        """测试驾驶员状态对象创建"""
        state = DriverState(
            vehicle_id="test_vehicle",
            timestamp=100.0,
            fatigue_level=1,
            emotion_state=0,
            risk_level=0.3,
            confidence=0.85,
            blink_features=np.random.rand(100),
            pulse_features=np.random.rand(100)
        )
        
        self.assertEqual(state.vehicle_id, "test_vehicle")
        self.assertEqual(state.timestamp, 100.0)
        self.assertEqual(state.fatigue_level, 1)
        self.assertEqual(state.emotion_state, 0)
        self.assertEqual(state.risk_level, 0.3)
        self.assertEqual(state.confidence, 0.85)
        self.assertEqual(len(state.blink_features), 100)
        self.assertEqual(len(state.pulse_features), 100)

class TestIntegrationScenarios(unittest.TestCase):
    """测试集成场景"""
    
    def setUp(self):
        """测试前准备"""
        self.dnsr = DNSRIntegration()
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        vehicle_ids = ["vehicle_001", "vehicle_002"]
        simulation_times = [100.0, 110.0, 120.0]
        
        for time_step in simulation_times:
            # 获取所有车辆状态
            states = self.dnsr.get_batch_driver_states(vehicle_ids, time_step)
            
            # 验证状态
            for vehicle_id, state in states.items():
                self.assertIsInstance(state, DriverState)
                self.assertEqual(state.timestamp, time_step)
                self.assertTrue(0 <= state.fatigue_level <= 2)
                self.assertTrue(0 <= state.emotion_state <= 1)
                self.assertTrue(0.0 <= state.risk_level <= 1.0)
                self.assertTrue(0.0 <= state.confidence <= 1.0)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效车辆ID
        state = self.dnsr.get_risk_driver_state("", 100.0)
        # 应该能够处理，返回None或有效状态
        
        # 测试无效时间
        state = self.dnsr.get_risk_driver_state("test", -100.0)
        # 应该能够处理，返回None或有效状态

def run_performance_test():
    """运行性能测试"""
    print("🚀 运行性能测试...")
    
    dnsr = DNSRIntegration()
    vehicle_ids = [f"vehicle_{i:03d}" for i in range(10)]
    simulation_time = 100.0
    
    import time
    
    # 测试单个车辆性能
    start_time = time.time()
    for _ in range(100):
        dnsr.get_risk_driver_state("test_vehicle", simulation_time)
    single_time = time.time() - start_time
    
    # 测试批量性能
    start_time = time.time()
    for _ in range(10):
        dnsr.get_batch_driver_states(vehicle_ids, simulation_time)
    batch_time = time.time() - start_time
    
    print(f"单个车辆100次预测耗时: {single_time:.3f}s")
    print(f"10个车辆10次批量预测耗时: {batch_time:.3f}s")
    print(f"平均单次预测耗时: {single_time/100*1000:.2f}ms")

if __name__ == "__main__":
    # 运行单元测试
    print("🧪 运行DNSR集成模块单元测试...")
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    print("\n" + "="*50)
    run_performance_test()
    
    print("\n✅ 所有测试完成!") 