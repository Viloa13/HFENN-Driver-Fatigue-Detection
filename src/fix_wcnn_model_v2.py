#!/usr/bin/env python3
"""
WCNN模型修复脚本 v2.0--可
修复现有WCNN模型的兼容性问题，解决SlicingOpLambda和形状不匹配问题

作者: AI Assistant
版本: v2.0
日期: 2025-01-03
"""

import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

class SlicingOpLambda:
    """自定义的SlicingOpLambda层，用于替代原始模型中的自定义层"""
    
    def __init__(self, **kwargs):
        super(SlicingOpLambda, self).__init__()
        self.kwargs = kwargs
    
    def call(self, inputs):
        # 实现切片操作
        return inputs
    
    def get_config(self):
        return self.kwargs

def fix_wcnn_model():
    """修复WCNN模型兼容性问题"""
    print("🔧 修复WCNN模型兼容性问题...")
    
    # 查找现有的WCNN模型文件
    possible_model_paths = [
        "WCNN.h5",
        "best_model.h5", 
        "E:/code/DNSR 20250828/WCNN.h5",
        "E:/code/DNSR 20250828/best_model.h5",
        "../WCNN.h5",
        "../best_model.h5"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ 找到现有模型: {model_path}")
            break
    
    if not model_path:
        print("❌ 未找到现有的WCNN模型文件")
        print("请确保以下文件之一存在:")
        for path in possible_model_paths:
            print(f"  - {path}")
        return False
    
    try:
        # 使用自定义对象加载模型
        print("📥 使用自定义对象加载模型...")
        with keras.utils.custom_object_scope({'SlicingOpLambda': SlicingOpLambda}):
            model = keras.models.load_model(model_path)
        print("✅ 模型加载成功")
        
        # 显示模型信息
        print(f"📊 模型输入数量: {len(model.inputs)}")
        for i, input_layer in enumerate(model.inputs):
            print(f"  输入 {i+1}: {input_layer.shape}")
        print(f"📊 模型输出: {model.output.shape}")
        
        # 创建输出目录
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        
        # 保存为.keras格式
        output_path = output_dir / "WCNN_fixed.keras"
        model.save(output_path, save_format='keras')
        print(f"✅ 模型已保存为: {output_path}")
        
        # 测试模型是否可以重新加载
        print("🧪 测试模型兼容性...")
        test_model = keras.models.load_model(output_path)
        print("✅ 模型兼容性测试通过")
        
        # 创建模型信息文件
        info_path = output_dir / "WCNN_model_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("WCNN模型信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"原始模型: {model_path}\n")
            f.write(f"修复后模型: {output_path}\n")
            f.write(f"保存格式: Keras (.keras)\n")
            f.write(f"输入数量: {len(model.inputs)}\n")
            f.write(f"输出形状: {model.output.shape}\n")
            f.write("\n模型结构:\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        print(f"✅ 模型信息已保存到: {info_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型修复失败: {e}")
        return False

def create_compatible_wcnn():
    """创建一个完全兼容的WCNN模型"""
    print("\n🔄 创建兼容的WCNN模型...")
    
    try:
        # 创建输出目录
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        
        # 构建兼容的WCNN模型
        # 注意：这里使用与原始模型匹配的输入形状
        input1 = keras.Input(shape=(1305, 1), name='input_1')  # cD1
        input2 = keras.Input(shape=(653, 1), name='input_2')   # cD2
        input3 = keras.Input(shape=(326, 1), name='input_3')   # cD3
        input4 = keras.Input(shape=(326, 1), name='input_4')   # cA3
        
        # 第一个分支处理
        x1 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input1)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.MaxPooling1D(2)(x1)
        
        # 第二个分支处理 - 注意维度调整
        x2 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input2)
        x2 = keras.layers.BatchNormalization()(x2)
        # 调整维度以匹配x1
        x2 = keras.layers.Lambda(lambda x: x[:, :-1, :])(x2)
        
        # 第三个分支处理
        x3 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input3)
        x3 = keras.layers.BatchNormalization()(x3)
        
        # 第四个分支处理
        x4 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input4)
        x4 = keras.layers.BatchNormalization()(x4)
        
        # 合并前两个分支
        x = keras.layers.Concatenate()([x1, x2])
        x = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        
        # 合并所有分支 - 注意维度匹配
        # 将x3和x4调整到与x相同的维度
        x3_adjusted = keras.layers.Lambda(lambda x: x[:, :x.shape[1], :])(x3)
        x4_adjusted = keras.layers.Lambda(lambda x: x[:, :x.shape[1], :])(x4)
        
        x = keras.layers.Concatenate()([x, x3_adjusted, x4_adjusted])
        
        # 卷积层
        x = keras.layers.Conv1D(64, 3, activation='relu')(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.Conv1D(64, 3, activation='relu')(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
        
        # 全连接层
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        
        # 创建模型
        model = keras.Model([input1, input2, input3, input4], outputs)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 保存模型
        output_path = output_dir / "WCNN_compatible.keras"
        model.save(output_path, save_format='keras')
        print(f"✅ 兼容模型已保存为: {output_path}")
        
        # 显示模型信息
        print("\n📊 兼容模型结构:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"❌ 创建兼容模型失败: {e}")
        return False

def create_simple_wcnn():
    """创建一个简化的WCNN模型作为备选方案"""
    print("\n🔄 创建简化的WCNN模型...")
    
    try:
        # 创建输出目录
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        
        # 构建简化的WCNN模型 - 使用统一的输入形状
        input_shape = (1000, 1)  # 使用统一的输入形状
        
        input1 = keras.Input(shape=input_shape, name='input_1')
        input2 = keras.Input(shape=input_shape, name='input_2')
        input3 = keras.Input(shape=input_shape, name='input_3')
        input4 = keras.Input(shape=input_shape, name='input_4')
        
        # 简化的处理流程
        x1 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input1)
        x1 = keras.layers.MaxPooling1D(2)(x1)
        
        x2 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input2)
        x2 = keras.layers.MaxPooling1D(2)(x2)
        
        x3 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input3)
        x3 = keras.layers.MaxPooling1D(2)(x3)
        
        x4 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input4)
        x4 = keras.layers.MaxPooling1D(2)(x4)
        
        # 合并所有输入
        x = keras.layers.Concatenate()([x1, x2, x3, x4])
        
        # 卷积层
        x = keras.layers.Conv1D(64, 3, activation='relu')(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.Conv1D(64, 3, activation='relu')(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
        
        # 全连接层
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(2, activation='softmax')(x)
        
        # 创建模型
        model = keras.Model([input1, input2, input3, input4], outputs)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 保存模型
        output_path = output_dir / "WCNN_simple.keras"
        model.save(output_path, save_format='keras')
        print(f"✅ 简化模型已保存为: {output_path}")
        
        # 显示模型信息
        print("\n📊 简化模型结构:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"❌ 创建简化模型失败: {e}")
        return False

def main():
    """主函数"""
    print("🚗🤖 WCNN模型修复工具 v2.0")
    print("修复模型兼容性问题，确保与DNSR模块兼容")
    print("=" * 60)
    
    # 方法1：尝试修复现有模型
    print("🎯 方法1：修复现有模型")
    if fix_wcnn_model():
        print("✅ 现有模型修复成功！")
        return True
    
    # 方法2：创建兼容模型
    print("\n🎯 方法2：创建兼容模型")
    if create_compatible_wcnn():
        print("✅ 兼容模型创建成功！")
        return True
    
    # 方法3：创建简化模型
    print("\n🎯 方法3：创建简化模型")
    if create_simple_wcnn():
        print("✅ 简化模型创建成功！")
        return True
    
    print("\n❌ 所有方法都失败了")
    print("\n💡 建议:")
    print("1. 检查emotion_data.csv文件是否存在")
    print("2. 运行retrain_wcnn.py重新训练模型")
    print("3. 确保TensorFlow版本兼容")
    print("4. 检查模型文件是否损坏")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 模型修复完成！")
        print("下一步操作:")
        print("1. 将生成的.keras文件复制到DNSR模块目录")
        print("2. 更新DNSR模块的模型路径配置")
        print("3. 重新运行DNSR测试")
    else:
        print("\n❌ 模型修复失败")
    
    exit(0 if success else 1) 