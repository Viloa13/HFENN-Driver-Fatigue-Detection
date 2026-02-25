"""
V2X智能避让系统集成示例20250801
将个性化HFENN/WCNN驾驶员状态检测与多智能体强化学习避让系统集成
"""

import time
import numpy as np
import threading
from typing import Dict, List, Optional
import json
from driver_state_detector import DriverStateDetector, create_personalized_detector

# 模拟V2X通信模块
class V2XCommunicator:
    """V2X通信模块"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.message_queue = []
        self.subscribers = []
    
    def broadcast_message(self, message: Dict):
        """广播V2X消息"""
        v2x_message = {
            'sender_id': self.vehicle_id,
            'timestamp': time.time(),
            'message_type': message.get('message_type', 'DRIVER_STATE'),
            'payload': message
        }
        
        print(f"🚗 [{self.vehicle_id}] 广播V2X消息: {message['safety_status']} - "
              f"风险等级: {message['risk_level']}")
        
        # 通知所有订阅者
        for subscriber in self.subscribers:
            subscriber.receive_v2x_message(v2x_message)
    
    def subscribe(self, subscriber):
        """订阅V2X消息"""
        self.subscribers.append(subscriber)

# 模拟智能眼镜传感器
class SmartGlassesSensor:
    """智能眼镜传感器模拟器"""
    
    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.is_running = False
        self.callbacks = []
        
        # 模拟不同用户的基础生理参数
        self.base_pulse = np.random.uniform(0.3, 0.7)
        self.base_blink = np.random.uniform(0.2, 0.6)
        
        # 状态模拟参数
        self.fatigue_trend = 0.0
        self.emotion_state = 0.0
    
    def add_callback(self, callback):
        """添加数据回调函数"""
        self.callbacks.append(callback)
    
    def simulate_fatigue_progression(self):
        """模拟疲劳状态渐进变化"""
        # 模拟驾驶过程中疲劳度逐渐增加
        self.fatigue_trend += np.random.uniform(0.001, 0.003)
        
        # 随机情绪波动
        if np.random.random() < 0.05:  # 5%概率情绪波动
            self.emotion_state = np.random.uniform(-0.3, 0.3)
    
    def generate_sensor_data(self) -> Dict:
        """生成传感器数据"""
        self.simulate_fatigue_progression()
        
        # 生成带噪声的生理信号
        pulse_noise = np.random.normal(0, 0.05)
        blink_noise = np.random.normal(0, 0.03)
        
        # 疲劳影响：疲劳时脉搏变化，眨眼频率/持续时间变化
        fatigue_effect = min(self.fatigue_trend, 0.4)
        emotion_effect = self.emotion_state * 0.2
        
        pulse_value = self.base_pulse + emotion_effect + pulse_noise
        blink_value = self.base_blink + fatigue_effect + blink_noise
        
        # 确保数值在合理范围内
        pulse_value = np.clip(pulse_value, 0, 1)
        blink_value = np.clip(blink_value, 0, 1)
        
        return {
            'pulse': pulse_value,
            'blink': blink_value,
            'timestamp': time.time()
        }
    
    def start_streaming(self, interval: float = 0.01):
        """开始数据流"""
        self.is_running = True
        
        def stream_loop():
            while self.is_running:
                sensor_data = self.generate_sensor_data()
                
                # 调用所有回调函数
                for callback in self.callbacks:
                    callback(sensor_data)
                
                time.sleep(interval)
        
        self.stream_thread = threading.Thread(target=stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        print(f"📡 [{self.participant_id}] 智能眼镜传感器开始数据流")
    
    def stop_streaming(self):
        """停止数据流"""
        self.is_running = False
        print(f"📡 [{self.participant_id}] 智能眼镜传感器停止数据流")

# 模拟MARL避让控制器
class MARLAvoidanceController:
    """多智能体强化学习避让控制器"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.received_alerts = []
        self.current_risk_assessment = 0
        
    def receive_v2x_message(self, message: Dict):
        """接收V2X消息"""
        payload = message['payload']
        sender_id = message['sender_id']
        risk_level = payload['risk_level']
        safety_status = payload['safety_status']
        
        print(f"🤖 [{self.vehicle_id}] 收到来自 {sender_id} 的风险警告: "
              f"{safety_status} (风险等级: {risk_level})")
        
        # 更新风险评估
        self.current_risk_assessment = max(self.current_risk_assessment, risk_level)
        
        # 根据风险等级执行避让策略
        if risk_level >= 4:  # 高风险
            self.execute_emergency_avoidance(payload)
        elif risk_level >= 2:  # 中等风险
            self.execute_preventive_maneuver(payload)
    
    def execute_emergency_avoidance(self, alert_data: Dict):
        """执行紧急避让"""
        print(f"🚨 [{self.vehicle_id}] 执行紧急避让! "
              f"目标车辆: {alert_data['vehicle_id']}, "
              f"疲劳等级: {alert_data['fatigue_level']}, "
              f"情绪状态: {alert_data['emotion_state']}")
        
        # 这里将调用MARL算法选择最优避让动作
        # 例如：变道、减速、保持安全距离等
        action = self.select_avoidance_action(alert_data)
        print(f"   └─ 选择动作: {action}")
    
    def execute_preventive_maneuver(self, alert_data: Dict):
        """执行预防性机动"""
        print(f"⚠️  [{self.vehicle_id}] 执行预防性机动 "
              f"(风险车辆: {alert_data['vehicle_id']})")
        
        action = self.select_preventive_action(alert_data)
        print(f"   └─ 选择动作: {action}")
    
    def select_avoidance_action(self, alert_data: Dict) -> str:
        """选择避让动作（模拟MARL决策）"""
        # 这里应该是MARL算法的核心决策逻辑
        # 简化模拟不同的避让策略
        
        fatigue_level = alert_data['fatigue_level']
        emotion_state = alert_data['emotion_state']
        
        if fatigue_level == 2 and emotion_state == 1:
            # 疲劳+负面情绪：最高风险，执行紧急避让
            return "紧急制动 + 向右变道"
        elif fatigue_level == 2:
            # 仅疲劳：执行减速和保持距离
            return "中度减速 + 增加跟车距离"
        elif emotion_state == 1:
            # 仅负面情绪：谨慎操作
            return "轻微减速 + 准备变道"
        else:
            return "保持警觉 + 预备机动"
    
    def select_preventive_action(self, alert_data: Dict) -> str:
        """选择预防性动作"""
        return "增加跟车距离 + 降低车速"

# 综合系统控制器
class IntelligentV2XSystem:
    """智能V2X安全系统"""
    
    def __init__(self, vehicle_id: str, participant_id: str, 
                 personalized_model_path: Optional[str] = None):
        self.vehicle_id = vehicle_id
        self.participant_id = participant_id
        
        # 初始化各个组件
        self.sensor = SmartGlassesSensor(participant_id)
        self.v2x_comm = V2XCommunicator(vehicle_id)
        
        # 初始化驾驶员状态检测器
        if personalized_model_path:
            self.state_detector = create_personalized_detector(
                participant_id, personalized_model_path
            )
            print(f"🎯 使用个性化模型: {participant_id}")
        else:
            self.state_detector = DriverStateDetector()
            print(f"🌐 使用通用模型")
        
        # 设置传感器回调
        self.sensor.add_callback(self.on_sensor_data)
        
        # 状态跟踪
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # 5秒警告冷却时间
        
        print(f"✅ 智能V2X系统初始化完成 - 车辆: {vehicle_id}, 驾驶员: {participant_id}")
    
    def on_sensor_data(self, sensor_data: Dict):
        """处理传感器数据"""
        # 更新状态检测器的数据缓冲区
        self.state_detector.update_data_buffer(
            sensor_data['pulse'], 
            sensor_data['blink']
        )
        
        # 尝试获取实时预测
        prediction = self.state_detector.get_realtime_prediction()
        
        if prediction:
            self.handle_driver_state_prediction(prediction)
    
    def handle_driver_state_prediction(self, prediction: Dict):
        """处理驾驶员状态预测结果"""
        risk_level = prediction['risk_level']
        current_time = time.time()
        
        # 只在风险状态下发送警告，并考虑冷却时间
        if risk_level > 0 and (current_time - self.last_alert_time) > self.alert_cooldown:
            # 转换为V2X消息
            v2x_message = self.state_detector.to_v2x_message(prediction)
            
            # 广播警告
            self.v2x_comm.broadcast_message(v2x_message)
            self.last_alert_time = current_time
            
            # 记录详细信息
            print(f"📊 [{self.vehicle_id}] 状态检测结果:")
            print(f"   ├─ 情绪: {prediction['emotion_label']} ({prediction['emotion_confidence']:.2f})")
            print(f"   ├─ 疲劳: {prediction['fatigue_label']} ({prediction['fatigue_confidence']:.2f})")
            print(f"   └─ 风险等级: {risk_level}")
    
    def connect_to_traffic_network(self, other_vehicles: List['MARLAvoidanceController']):
        """连接到交通网络"""
        for vehicle in other_vehicles:
            self.v2x_comm.subscribe(vehicle)
        print(f"🌐 [{self.vehicle_id}] 已连接到交通网络 ({len(other_vehicles)} 辆车)")
    
    def start_monitoring(self):
        """开始监测"""
        self.sensor.start_streaming()
        print(f"🚀 [{self.vehicle_id}] 开始驾驶员状态监测")
    
    def stop_monitoring(self):
        """停止监测"""
        self.sensor.stop_streaming()
        print(f"🛑 [{self.vehicle_id}] 停止驾驶员状态监测")
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'vehicle_id': self.vehicle_id,
            'participant_id': self.participant_id,
            'sensor_status': self.sensor.is_running,
            'detector_info': self.state_detector.get_model_info(),
            'last_alert_time': self.last_alert_time
        }

def main():
    """主函数 - 演示完整的V2X智能避让系统"""
    print("🚗🤖 V2X智能避让系统演示")
    print("=" * 60)
    
    # 创建风险车辆（配备智能眼镜的车辆）
    risk_vehicle = IntelligentV2XSystem(
        vehicle_id="CAR_001", 
        participant_id="driver_zhang",
        personalized_model_path=None  # 可以指定个性化模型路径
    )
    
    # 创建周边车辆（MARL避让车辆）
    surrounding_vehicles = [
        MARLAvoidanceController("CAR_002"),
        MARLAvoidanceController("CAR_003"),
        MARLAvoidanceController("CAR_004"),
        MARLAvoidanceController("CAR_005")
    ]
    
    # 建立V2X网络连接
    risk_vehicle.connect_to_traffic_network(surrounding_vehicles)
    
    # 开始监测
    risk_vehicle.start_monitoring()
    
    # 运行仿真
    try:
        print("\n🔄 开始实时监测...")
        print("系统将监测驾驶员状态并在检测到风险时触发V2X警告")
        print("按 Ctrl+C 停止仿真\n")
        
        # 运行60秒仿真
        for i in range(60):
            time.sleep(1)
            
            # 每10秒显示系统状态
            if i % 10 == 0:
                status = risk_vehicle.get_system_status()
                buffer_status = status['detector_info']['buffer_status']
                print(f"\n📈 系统状态 (第{i}秒):")
                print(f"   缓冲区: 脉搏({buffer_status['pulse_buffer_size']}) "
                      f"眨眼({buffer_status['blink_buffer_size']})")
        
        print("\n✅ 仿真完成")
        
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断仿真")
    
    finally:
        # 清理资源
        risk_vehicle.stop_monitoring()
        print("🧹 系统资源已清理")

if __name__ == "__main__":
    main() 