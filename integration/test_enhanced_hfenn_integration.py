#!/usr/bin/env python3
"""
Enhanced HFENN 集成模块单元测试
测试 EnhancedHFENNIntegrator 类的所有主要功能
"""

import unittest
import numpy as np
import tempfile
import os
import pickle
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 导入被测试的模块
from enhanced_hfenn_integration import EnhancedHFENNIntegrator, EnhancedFeatureExtractor


class TestEnhancedFeatureExtractor(unittest.TestCase):
    """测试增强特征提取器"""
    
    def setUp(self):
        """测试前准备"""
        self.extractor = EnhancedFeatureExtractor(sampling_rate=87)
        self.test_signal = np.random.randn(2610)  # 使用正确的窗口大小
    
    def test_extract_time_domain_features(self):
        """测试时域特征提取"""
        features = self.extractor.extract_time_domain_features(self.test_signal)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 16)  # 16个时域特征
        self.assertTrue(all(np.isfinite(f) for f in features))
    
    def test_extract_frequency_domain_features(self):
        """测试频域特征提取"""
        features = self.extractor.extract_frequency_domain_features(self.test_signal)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 9)  # 9个频域特征
        self.assertTrue(all(np.isfinite(f) for f in features))
    
    def test_extract_wavelet_features(self):
        """测试小波特征提取"""
        features = self.extractor.extract_wavelet_features(self.test_signal)
        
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)
        self.assertTrue(all(np.isfinite(f) for f in features))
    
    def test_extract_nonlinear_features(self):
        """测试非线性特征提取"""
        features = self.extractor.extract_nonlinear_features(self.test_signal)
        
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 5)  # 5个非线性特征
        self.assertTrue(all(np.isfinite(f) for f in features))
    
    def test_extract_all_features(self):
        """测试所有特征提取"""
        features = self.extractor.extract_all_features(self.test_signal)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 30)  # 应该有30+个特征
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_short_signal(self):
        """测试短信号处理"""
        short_signal = np.random.randn(100)
        
        # 应该能够处理短信号，不抛出异常
        features = self.extractor.extract_all_features(short_signal)
        self.assertIsInstance(features, np.ndarray)
    
    def test_constant_signal(self):
        """测试常数信号"""
        constant_signal = np.ones(2610)
        
        features = self.extractor.extract_all_features(constant_signal)
        self.assertIsInstance(features, np.ndarray)
        # 某些特征可能为0（如标准差），但不应该是NaN
        self.assertTrue(np.all(np.isfinite(features)))


class TestEnhancedHFENNIntegrator(unittest.TestCase):
    """测试Enhanced HFENN集成器"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟文件路径
        self.model_path = os.path.join(self.temp_dir, "test_model.keras")
        self.scaler_path = os.path.join(self.temp_dir, "test_scaler.pkl")
        self.selector_path = os.path.join(self.temp_dir, "test_selector.pkl")
        
        # 创建模拟文件
        self._create_mock_files()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_files(self):
        """创建模拟文件"""
        # 创建模拟标准化器
        scaler = StandardScaler()
        mock_data = np.random.randn(100, 50)  # 100个样本，50个特征
        scaler.fit(mock_data)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # 创建模拟特征选择器索引
        selected_indices = np.arange(50)  # 选择前50个特征
        with open(self.selector_path, 'wb') as f:
            pickle.dump(selected_indices, f)
    
    def _create_mock_model(self):
        """创建模拟模型"""
        mock_model = Mock()
        mock_model.inputs = [Mock() for _ in range(6)]  # 6个输入
        for i, input_layer in enumerate(mock_model.inputs):
            input_layer.name = f"input_{i}"
            input_layer.shape = (None, 100, 1) if i < 5 else (None, 50)
        
        # 模拟预测结果
        mock_predictions = np.array([[0.1, 0.2, 0.7]])  # 3个类别的概率
        mock_model.predict.return_value = mock_predictions
        
        return mock_model
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_initialization_success(self, mock_load_model):
        """测试成功初始化"""
        mock_model = self._create_mock_model()
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            selector_path=self.selector_path,
            log_level=logging.WARNING,  # 减少测试日志
            auto_load=False
        )
        
        self.assertIsNotNone(integrator.model)
        self.assertIsNotNone(integrator.scaler)
        self.assertIsNotNone(integrator.feature_selector_indices)
    
    def test_initialization_missing_files(self):
        """测试缺失文件的初始化"""
        with self.assertRaises(FileNotFoundError):
            EnhancedHFENNIntegrator(
                model_path="nonexistent.keras",
                log_level=logging.CRITICAL  # 抑制错误日志
            )
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_preprocess_signal_correct_length(self, mock_load_model):
        """测试预处理正确长度的信号"""
        mock_model = self._create_mock_model()
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            selector_path=self.selector_path,
            log_level=logging.CRITICAL
        )
        
        # 测试正确长度的信号
        signal = np.random.randn(2610)
        wavelet_coeffs, enhanced_features = integrator._preprocess_signal(signal)
        
        # 检查小波系数
        self.assertEqual(len(wavelet_coeffs), 5)
        for coeff in wavelet_coeffs:
            self.assertEqual(len(coeff.shape), 3)  # (batch, length, channels)
            self.assertEqual(coeff.shape[0], 1)    # batch size = 1
            self.assertEqual(coeff.shape[2], 1)    # channels = 1
        
        # 检查增强特征
        self.assertEqual(enhanced_features.shape, (1, 50))
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_preprocess_signal_wrong_length(self, mock_load_model):
        """测试预处理错误长度的信号"""
        mock_model = self._create_mock_model()
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            selector_path=self.selector_path,
            log_level=logging.CRITICAL
        )
        
        # 测试短信号
        short_signal = np.random.randn(1000)
        wavelet_coeffs, enhanced_features = integrator._preprocess_signal(short_signal)
        
        self.assertEqual(len(wavelet_coeffs), 5)
        self.assertEqual(enhanced_features.shape, (1, 50))
        
        # 测试长信号
        long_signal = np.random.randn(5000)
        wavelet_coeffs, enhanced_features = integrator._preprocess_signal(long_signal)
        
        self.assertEqual(len(wavelet_coeffs), 5)
        self.assertEqual(enhanced_features.shape, (1, 50))
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_predict_fatigue_success(self, mock_load_model):
        """测试成功的疲劳预测"""
        mock_model = self._create_mock_model()
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            selector_path=self.selector_path,
            log_level=logging.CRITICAL
        )
        
        signal = np.random.randn(2610)
        result = integrator.predict_fatigue(signal)
        
        # 检查返回结果
        self.assertTrue(result['success'])
        self.assertIn('predicted_class', result)
        self.assertIn('predicted_label', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)
        
        # 检查数值范围
        self.assertIn(result['predicted_class'], [0, 1, 2])
        self.assertIn(result['predicted_label'], ["Normal", "Mild Fatigue", "Fatigue"])
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)
        
        # 检查概率总和
        probs = list(result['probabilities'].values())
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_predict_fatigue_model_error(self, mock_load_model):
        """测试模型预测错误"""
        mock_model = self._create_mock_model()
        mock_model.predict.side_effect = Exception("模型预测错误")
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            selector_path=self.selector_path,
            log_level=logging.CRITICAL
        )
        
        signal = np.random.randn(2610)
        result = integrator.predict_fatigue(signal)
        
        # 应该返回失败结果
        self.assertFalse(result['success'])
        self.assertIsNone(result['predicted_class'])
        self.assertEqual(result['confidence'], 0.0)
    
    def test_predict_fatigue_no_model(self):
        """测试没有模型的预测"""
        integrator = EnhancedHFENNIntegrator(
            model_path="nonexistent.keras",
            log_level=logging.CRITICAL
        )
        integrator.model = None  # 确保模型为空
        
        signal = np.random.randn(2610)
        result = integrator.predict_fatigue(signal)
        
        self.assertFalse(result['success'])
        self.assertIsNone(result['predicted_class'])
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_validate_setup(self, mock_load_model):
        """测试系统验证"""
        mock_model = self._create_mock_model()
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            selector_path=self.selector_path,
            log_level=logging.CRITICAL
        )
        
        validation = integrator.validate_setup()
        
        self.assertIn('model_loaded', validation)
        self.assertIn('scaler_loaded', validation)
        self.assertIn('selector_loaded', validation)
        self.assertIn('prediction_test', validation)
        self.assertIn('overall', validation)
        
        self.assertTrue(validation['model_loaded'])
        self.assertTrue(validation['scaler_loaded'])
        self.assertTrue(validation['selector_loaded'])
        self.assertTrue(validation['overall'])
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_extract_and_process_features_without_preprocessors(self, mock_load_model):
        """测试没有预处理器的特征处理"""
        mock_model = self._create_mock_model()
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path=self.model_path,
            log_level=logging.CRITICAL
        )
        
        # 确保预处理器为空
        integrator.scaler = None
        integrator.feature_selector_indices = None
        
        signal = np.random.randn(2610)
        features = integrator._extract_and_process_features(signal)
        
        # 应该能够处理并返回50个特征
        self.assertEqual(features.shape, (1, 50))
    
    def test_class_mapping(self):
        """测试类别映射"""
        integrator = EnhancedHFENNIntegrator(
            model_path="nonexistent.keras",
            log_level=logging.CRITICAL
        )
        
        expected_mapping = {
            0: "Normal",
            1: "Mild Fatigue", 
            2: "Fatigue"
        }
        
        self.assertEqual(integrator.class_mapping, expected_mapping)


class TestIntegrationScenarios(unittest.TestCase):
    """测试集成场景"""
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_end_to_end_workflow(self, mock_load_model):
        """测试端到端工作流程"""
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "test.keras")
        scaler_path = os.path.join(temp_dir, "scaler.pkl")
        selector_path = os.path.join(temp_dir, "selector.pkl")
        
        try:
            # 创建模拟文件
            scaler = StandardScaler()
            scaler.fit(np.random.randn(100, 50))
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            with open(selector_path, 'wb') as f:
                pickle.dump(np.arange(50), f)
            
            # 创建模拟模型
            mock_model = Mock()
            mock_model.inputs = [Mock() for _ in range(6)]
            for i, input_layer in enumerate(mock_model.inputs):
                input_layer.name = f"input_{i}"
                input_layer.shape = (None, 100, 1) if i < 5 else (None, 50)
            mock_model.predict.return_value = np.array([[0.2, 0.3, 0.5]])
            mock_load_model.return_value = mock_model
            
            # 创建集成器
            integrator = EnhancedHFENNIntegrator(
                model_path=model_path,
                scaler_path=scaler_path,
                selector_path=selector_path,
                log_level=logging.CRITICAL
            )
            
            # 验证系统
            validation = integrator.validate_setup()
            self.assertTrue(validation['overall'])
            
            # 执行多次预测
            signals = [np.random.randn(2610) for _ in range(5)]
            results = []
            
            for signal in signals:
                result = integrator.predict_fatigue(signal)
                self.assertTrue(result['success'])
                results.append(result)
            
            # 验证所有结果
            for result in results:
                self.assertIn(result['predicted_class'], [0, 1, 2])
                self.assertTrue(0.0 <= result['confidence'] <= 1.0)
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    @patch('enhanced_hfenn_integration.load_model')
    def test_error_recovery(self, mock_load_model):
        """测试错误恢复"""
        mock_model = Mock()
        mock_model.inputs = [Mock() for _ in range(6)]
        for i, input_layer in enumerate(mock_model.inputs):
            input_layer.name = f"input_{i}"
            input_layer.shape = (None, 100, 1) if i < 5 else (None, 50)
        
        # 模拟间歇性错误
        error_count = 0
        def side_effect(*args, **kwargs):
            nonlocal error_count
            error_count += 1
            if error_count % 3 == 0:  # 每3次调用出现一次错误
                raise Exception("间歇性错误")
            return np.array([[0.1, 0.4, 0.5]])
        
        mock_model.predict.side_effect = side_effect
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path="test.keras",
            log_level=logging.CRITICAL
        )
        integrator.scaler = None
        integrator.feature_selector_indices = None
        
        # 多次调用，测试错误恢复
        success_count = 0
        for i in range(10):
            signal = np.random.randn(2610)
            result = integrator.predict_fatigue(signal)
            if result['success']:
                success_count += 1
        
        # 应该有成功的预测
        self.assertGreater(success_count, 0)


def run_performance_test():
    """运行性能测试"""
    print("\n" + "="*50)
    print("🚀 运行Enhanced HFENN性能测试...")
    
    with patch('enhanced_hfenn_integration.load_model') as mock_load_model:
        # 创建模拟模型
        mock_model = Mock()
        mock_model.inputs = [Mock() for _ in range(6)]
        for i, input_layer in enumerate(mock_model.inputs):
            input_layer.name = f"input_{i}"
            input_layer.shape = (None, 100, 1) if i < 5 else (None, 50)
        mock_model.predict.return_value = np.array([[0.1, 0.3, 0.6]])
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path="test.keras",
            log_level=logging.CRITICAL
        )
        integrator.scaler = None
        integrator.feature_selector_indices = None
        
        import time
        
        # 测试单次预测性能
        test_signal = np.random.randn(2610)
        
        start_time = time.time()
        for _ in range(100):
            result = integrator.predict_fatigue(test_signal)
        single_time = time.time() - start_time
        
        # 测试不同信号长度的处理时间
        signal_lengths = [1000, 2610, 5000, 10000]
        length_times = {}
        
        for length in signal_lengths:
            test_signal = np.random.randn(length)
            start_time = time.time()
            for _ in range(10):
                integrator._preprocess_signal(test_signal)
            length_times[length] = (time.time() - start_time) / 10
        
        print(f"单次预测性能 (100次平均): {single_time/100*1000:.2f}ms")
        print("不同信号长度处理时间:")
        for length, avg_time in length_times.items():
            print(f"  长度 {length}: {avg_time*1000:.2f}ms")


def run_memory_test():
    """运行内存测试"""
    print("\n内存使用测试...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    with patch('enhanced_hfenn_integration.load_model') as mock_load_model:
        mock_model = Mock()
        mock_model.inputs = [Mock() for _ in range(6)]
        mock_model.predict.return_value = np.array([[0.1, 0.3, 0.6]])
        mock_load_model.return_value = mock_model
        
        integrator = EnhancedHFENNIntegrator(
            model_path="test.keras",
            log_level=logging.CRITICAL
        )
        integrator.scaler = None
        integrator.feature_selector_indices = None
        
        # 大量预测测试内存泄漏
        for i in range(1000):
            signal = np.random.randn(2610)
            result = integrator.predict_fatigue(signal)
            
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"  第{i}次预测后内存: {current_memory:.1f}MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"初始内存: {initial_memory:.1f}MB")
        print(f"最终内存: {final_memory:.1f}MB")
        print(f"内存增长: {memory_increase:.1f}MB")
        
        if memory_increase > 100:  # 如果内存增长超过100MB
            print("⚠️  可能存在内存泄漏")
        else:
            print("✅ 内存使用正常")


if __name__ == "__main__":
    # 运行单元测试
    print("🧪 运行Enhanced HFENN集成模块单元测试...")
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    run_performance_test()
    
    # 运行内存测试
    run_memory_test()
    
    print("\n✅ 所有测试完成!")
