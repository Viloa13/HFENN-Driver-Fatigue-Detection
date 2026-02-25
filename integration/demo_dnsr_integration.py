#!/usr/bin/env python3
"""
DNSR集成模块演示脚本
展示如何使用DNSR集成模块进行驾驶员状态检测和风险评估

作者: AI Assistant
版本: v1.0
日期: 2025-0904
"""

import numpy as np
import time
import logging
from pathlib import Path

# 导入DNSR集成模块
from dnsr_integration import DNSRIntegration, DriverState

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def demo_basic_usage():
    """演示基本使用方法"""
    print("\n" + "="*60)
    print("🚗 演示1: 基本使用方法")
    print("="*60)
    
    # 创建DNSR集成模块实例
    print("📦 创建DNSR集成模块实例...")
    dnsr = DNSRIntegration()
    
    # 检查系统状态
    status = dnsr.get_system_status()
    print(f"✅ 系统状态:")
    print(f"   HFENN模型: {'✅ 已加载' if status['hfenn_loaded'] else '❌ 未加载'}")
    print(f"   WCNN模型: {'✅ 已加载' if status['wcnn_loaded'] else '❌ 未加载'}")
    print(f"   特征标准化器: {'✅ 已加载' if status['feature_scaler_loaded'] else '❌ 未加载'}")
    print(f"   特征选择器: {'✅ 已加载' if status['selected_features_loaded'] else '❌ 未加载'}")
    
    # 获取单个车辆状态
    print(f"\n🔍 获取车辆状态...")
    vehicle_id = "demo_vehicle_001"
    simulation_time = 100.0
    
    driver_state = dnsr.get_risk_driver_state(vehicle_id, simulation_time)
    
    if driver_state:
        print(f"✅ 成功获取车辆 {vehicle_id} 状态:")
        print(f"   疲劳等级: {driver_state.fatigue_level} ({['正常', '轻度疲劳', '疲劳'][driver_state.fatigue_level]})")
        print(f"   情绪状态: {driver_state.emotion_state} ({['正常', '负面情绪'][driver_state.emotion_state]})")
        print(f"   风险等级: {driver_state.risk_level:.3f}")
        print(f"   置信度: {driver_state.confidence:.3f}")
        print(f"   时间戳: {driver_state.timestamp}")
    else:
        print(f"❌ 获取车辆 {vehicle_id} 状态失败")

def demo_batch_processing():
    """演示批量处理方法"""
    print("\n" + "="*60)
    print("🚗 演示2: 批量处理方法")
    print("="*60)
    
    dnsr = DNSRIntegration()
    
    # 创建多个车辆ID
    vehicle_ids = [f"demo_vehicle_{i:03d}" for i in range(1, 6)]
    simulation_time = 150.0
    
    print(f"📊 批量获取 {len(vehicle_ids)} 个车辆状态...")
    start_time = time.time()
    
    batch_states = dnsr.get_batch_driver_states(vehicle_ids, simulation_time)
    
    processing_time = time.time() - start_time
    
    print(f"✅ 批量处理完成，耗时: {processing_time:.3f}s")
    print(f"📈 成功获取 {len(batch_states)} 个车辆状态:")
    
    # 统计风险分布
    risk_levels = [state.risk_level for state in batch_states.values()]
    fatigue_levels = [state.fatigue_level for state in batch_states.values()]
    emotion_states = [state.emotion_state for state in batch_states.values()]
    
    print(f"   风险等级分布: 平均={np.mean(risk_levels):.3f}, 范围=[{np.min(risk_levels):.3f}, {np.max(risk_levels):.3f}]")
    print(f"   疲劳等级分布: {dict(zip(*np.unique(fatigue_levels, return_counts=True)))}")
    print(f"   情绪状态分布: {dict(zip(*np.unique(emotion_states, return_counts=True)))}")
    
    # 显示高风险车辆
    high_risk_vehicles = [(vid, state) for vid, state in batch_states.items() if state.risk_level > 0.5]
    if high_risk_vehicles:
        print(f"🚨 检测到 {len(high_risk_vehicles)} 个高风险车辆:")
        for vid, state in high_risk_vehicles:
            print(f"   {vid}: 风险等级={state.risk_level:.3f}, 疲劳={state.fatigue_level}, 情绪={state.emotion_state}")

def demo_time_series_analysis():
    """演示时间序列分析"""
    print("\n" + "="*60)
    print("🚗 演示3: 时间序列分析")
    print("="*60)
    
    dnsr = DNSRIntegration()
    
    vehicle_id = "demo_vehicle_002"
    time_steps = np.arange(100, 200, 10)  # 10秒间隔
    
    print(f"📈 分析车辆 {vehicle_id} 在 {len(time_steps)} 个时间点的状态变化...")
    
    states_over_time = []
    for t in time_steps:
        state = dnsr.get_risk_driver_state(vehicle_id, t)
        if state:
            states_over_time.append(state)
    
    if states_over_time:
        print(f"✅ 成功获取 {len(states_over_time)} 个时间点的状态")
        
        # 分析趋势
        risk_trend = [state.risk_level for state in states_over_time]
        fatigue_trend = [state.fatigue_level for state in states_over_time]
        
        print(f"📊 风险等级趋势:")
        print(f"   起始值: {risk_trend[0]:.3f}")
        print(f"   结束值: {risk_trend[-1]:.3f}")
        print(f"   变化幅度: {risk_trend[-1] - risk_trend[0]:+.3f}")
        
        # 检测状态变化
        risk_changes = np.diff(risk_trend)
        significant_changes = np.where(np.abs(risk_changes) > 0.1)[0]
        
        if len(significant_changes) > 0:
            print(f"🚨 检测到 {len(significant_changes)} 次显著风险变化:")
            for i, change_idx in enumerate(significant_changes[:3]):  # 只显示前3次
                time_idx = change_idx + 1
                change = risk_changes[change_idx]
                print(f"   时间点 {time_steps[time_idx]}s: {change:+.3f}")

def demo_error_handling():
    """演示错误处理"""
    print("\n" + "="*60)
    print("🚗 演示4: 错误处理")
    print("="*60)
    
    dnsr = DNSRIntegration()
    
    # 测试无效输入
    print("🧪 测试错误处理...")
    
    # 空车辆ID
    print("   测试空车辆ID...")
    state = dnsr.get_risk_driver_state("", 100.0)
    print(f"   结果: {'✅ 正常处理' if state is not None else '❌ 返回None'}")
    
    # 无效时间
    print("   测试无效时间...")
    state = dnsr.get_risk_driver_state("test", -100.0)
    print(f"   结果: {'✅ 正常处理' if state is not None else '❌ 返回None'}")
    
    # 缓存管理
    print("   测试缓存管理...")
    dnsr.clear_cache()
    cached_state = dnsr.get_cached_driver_state("test_vehicle")
    print(f"   清除缓存后获取状态: {'✅ 返回None' if cached_state is None else '❌ 意外返回状态'}")

def demo_performance_benchmark():
    """演示性能基准测试"""
    print("\n" + "="*60)
    print("🚗 演示5: 性能基准测试")
    print("="*60)
    
    dnsr = DNSRIntegration()
    
    # 单次预测性能
    print("📊 单次预测性能测试...")
    vehicle_id = "benchmark_vehicle"
    simulation_time = 200.0
    
    times = []
    for i in range(10):
        start_time = time.time()
        state = dnsr.get_risk_driver_state(vehicle_id, simulation_time + i)
        end_time = time.time()
        if state:
            times.append(end_time - start_time)
    
    if times:
        avg_time = np.mean(times) * 1000  # 转换为毫秒
        std_time = np.std(times) * 1000
        print(f"✅ 单次预测性能:")
        print(f"   平均耗时: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"   最快耗时: {min(times)*1000:.2f}ms")
        print(f"   最慢耗时: {max(times)*1000:.2f}ms")
    
    # 批量处理性能
    print(f"\n📊 批量处理性能测试...")
    vehicle_ids = [f"benchmark_vehicle_{i:03d}" for i in range(1, 21)]  # 20个车辆
    simulation_time = 300.0
    
    start_time = time.time()
    batch_states = dnsr.get_batch_driver_states(vehicle_ids, simulation_time)
    batch_time = time.time() - start_time
    
    print(f"✅ 批量处理性能:")
    print(f"   处理车辆数: {len(vehicle_ids)}")
    print(f"   成功获取状态: {len(batch_states)}")
    print(f"   总耗时: {batch_time:.3f}s")
    print(f"   平均每车耗时: {batch_time/len(vehicle_ids)*1000:.2f}ms")
    print(f"   处理速度: {len(batch_states)/batch_time:.1f} 车辆/秒")

def demo_integration_scenario():
    """演示集成场景"""
    print("\n" + "="*60)
    print("🚗 演示6: 集成场景模拟")
    print("="*60)
    
    dnsr = DNSRIntegration()
    
    # 模拟交通场景
    print("🚦 模拟交通场景: 5辆车在交叉路口...")
    
    # 创建不同风险等级的车辆
    scenario_vehicles = {
        "emergency_vehicle": {"base_risk": 0.8, "description": "紧急车辆"},
        "tired_driver": {"base_risk": 0.6, "description": "疲劳驾驶员"},
        "normal_driver_1": {"base_risk": 0.2, "description": "正常驾驶员1"},
        "normal_driver_2": {"base_risk": 0.2, "description": "正常驾驶员2"},
        "aggressive_driver": {"base_risk": 0.7, "description": "激进驾驶员"}
    }
    
    simulation_time = 400.0
    
    print(f"📊 获取所有车辆状态...")
    all_states = {}
    
    for vehicle_id in scenario_vehicles.keys():
        state = dnsr.get_risk_driver_state(vehicle_id, simulation_time)
        if state:
            all_states[vehicle_id] = state
    
    # 分析场景风险
    if all_states:
        print(f"✅ 场景分析完成，共 {len(all_states)} 辆车:")
        
        # 按风险等级排序
        sorted_vehicles = sorted(all_states.items(), key=lambda x: x[1].risk_level, reverse=True)
        
        for i, (vehicle_id, state) in enumerate(sorted_vehicles):
            risk_level = state.risk_level
            description = scenario_vehicles[vehicle_id]["description"]
            
            if risk_level > 0.6:
                status = "🚨 高风险"
            elif risk_level > 0.3:
                status = "⚠️ 中风险"
            else:
                status = "✅ 低风险"
            
            print(f"   {i+1}. {vehicle_id} ({description}): {status}")
            print(f"      风险等级: {risk_level:.3f}, 疲劳: {state.fatigue_level}, 情绪: {state.emotion_state}")
        
        # 计算场景总体风险
        overall_risk = np.mean([state.risk_level for state in all_states.values()])
        print(f"\n📈 场景总体风险评估:")
        print(f"   平均风险等级: {overall_risk:.3f}")
        
        if overall_risk > 0.5:
            print("   🚨 场景风险较高，需要特别注意!")
        elif overall_risk > 0.3:
            print("   ⚠️ 场景风险中等，建议监控")
        else:
            print("   ✅ 场景风险较低，相对安全")

def main():
    """主函数"""
    print("🚗🤖 DNSR集成模块演示")
    print("="*60)
    print("本演示将展示DNSR集成模块的主要功能和使用方法")
    print("包括基本使用、批量处理、时间序列分析、错误处理、性能测试和集成场景")
    
    logger = setup_logging()
    
    try:
        # 运行所有演示
        demo_basic_usage()
        demo_batch_processing()
        demo_time_series_analysis()
        demo_error_handling()
        demo_performance_benchmark()
        demo_integration_scenario()
        
        print("\n" + "="*60)
        print("🎉 所有演示完成!")
        print("="*60)
        
        print("\n💡 使用建议:")
        print("   1. 在实际应用中，建议使用真实的预训练模型文件")
        print("   2. 可以根据具体需求调整配置参数")
        print("   3. 建议在生产环境中启用日志记录和性能监控")
        print("   4. 可以扩展支持更多类型的传感器数据")
        
        print("\n📚 更多信息请参考:")
        print("   - README_DNSR_Integration.md: 详细使用说明")
        print("   - test_dnsr_integration.py: 单元测试")
        print("   - dnsr_integration.py: 核心模块源码")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n❌ 演示过程中发生错误: {e}")
        print("请检查依赖库是否正确安装，以及模型文件是否存在")

if __name__ == "__main__":
    main() 