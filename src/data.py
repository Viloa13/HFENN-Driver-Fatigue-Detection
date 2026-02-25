import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置学术风格的绘图样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def draw_latency_comparison():
    # ==========================================
    # 1. 准备数据
    # ==========================================
    # 这里我们模拟三组数据。
    # 您可以将 'Micro-motion' 的数据替换为您 measure_performance() 跑出来的真实数据列表
    
    np.random.seed(42) # 固定随机种子保证结果可复现
    
    # A. 您的微动作交互 (Micro-motion)
    # 假设均值 20ms，标准差 2ms (非常稳定)
    # 来源：您的实验数据
    data_micro = np.random.normal(loc=20, scale=2, size=50)
    
    # B. 视觉手势交互 (Visual Gesture)
    # 假设均值 200ms，标准差 15ms
    # 来源：典型文献 (e.g., Camera-based detection + recognition)
    data_gesture = np.random.normal(loc=200, scale=15, size=50)
    
    # C. 语音交互 (Voice Command)
    # 假设均值 800ms，标准差 50ms
    # 来源：典型文献 (e.g., Cloud-based ASR or large local models)
    data_voice = np.random.normal(loc=800, scale=50, size=50)

    # 汇总数据
    data = [data_micro, data_gesture, data_voice]
    labels = ['Ours\n(Micro-motion)', 'Visual Gesture', 'Voice Command']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] # 学术常用配色：红(突出重点)、绿、蓝
    
    # 计算均值和标准差用于柱状图
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]

    # ==========================================
    # 2. 绘制图表
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 绘制柱状图 (带有误差棒 yerr)
    bars = ax.bar(labels, means, yerr=stds, capsize=10, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)

    # ==========================================
    # 3. 添加细节和标注
    # ==========================================
    
    # 添加具体的数值标签在柱子上方
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 40, # 稍微高一点避免遮挡误差棒
                f'{mean:.1f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 绘制 "优势倍数" 的连接线 (例如：比手势快10倍)
    # 获取柱子坐标
    x_locs = [bar.get_x() + bar.get_width()/2. for bar in bars]
    
    # 对比 1: Ours vs Gesture
    y_max_1 = max(means[0], means[1]) + 150
    ax.plot([x_locs[0], x_locs[0], x_locs[1], x_locs[1]], 
            [means[0]+50, y_max_1, y_max_1, means[1]+50], color='gray', linestyle='--', linewidth=1)
    ax.text((x_locs[0] + x_locs[1])/2, y_max_1 + 10, 
            f'{means[1]/means[0]:.1f}× Faster', ha='center', va='bottom', fontsize=10, color='gray')

    # 对比 2: Ours vs Voice (如果跨度太大，这个可以考虑用断轴或对数坐标，这里先用普通坐标)
    # 注意：如果Voice太高，可能会导致Ours的柱子看不清。
    # 如果差距过大，建议使用 "Log Scale" (对数坐标)
    # 这里演示开启对数坐标的方法：
    # ax.set_yscale('log') 
    
    # 设置标题和标签
    ax.set_ylabel('Response Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of End-to-End Interaction Latency', fontsize=16, pad=20)
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # 添加网格线 (仅Y轴)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    
    # 保存图片
    plt.savefig('response_time_comparison.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    draw_latency_comparison()