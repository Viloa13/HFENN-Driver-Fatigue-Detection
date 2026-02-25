"""
生成连续疲劳指数脚本
将离散的疲劳标签(1,2,3)转换为连续的Fatigue_Score(0.0-1.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

# 配置参数
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data" / "processed"
SMOOTHING_WINDOW = 50  # 滑动平均窗口大小

# 疲劳分数映射区间
FATIGUE_RANGES = {
    1: (0.0, 0.3),   # Label 1: 清醒状态
    2: (0.3, 0.7),   # Label 2: 轻度疲劳
    3: (0.7, 1.0),   # Label 3: 重度疲劳
}

# 13组参与者
PARTICIPANTS = ['gs1', 'gs2', 'hww', 'jz', 'lx1', 'lx2', 'lx3', 'pm', 'ysp', 'zch1', 'zch2', 'zm', 'zxl']


def extract_participant_prefix(filename: str) -> str:
    """从文件名提取参与者前缀，如 'gs1-1.csv' -> 'gs1'"""
    match = re.match(r'^([a-zA-Z]+\d*)-\d+\.csv$', filename)
    if match:
        return match.group(1)
    return None


def clean_numeric_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """清洗数值列：强制转换为numeric，处理NaN"""
    df[column] = pd.to_numeric(df[column], errors='coerce')
    # 使用前后值插值填充NaN
    df[column] = df[column].interpolate(method='linear', limit_direction='both')
    # 如果仍有NaN（如首尾），用均值填充
    if df[column].isna().any():
        df[column] = df[column].fillna(df[column].mean())
    return df


def calculate_fatigue_score(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """
    计算连续疲劳分数
    根据行在文件中的相对位置进行线性插值
    """
    start_val, end_val = FATIGUE_RANGES[label]
    total_rows = len(df)
    
    # 计算每行的相对进度 (0 到 1)
    progress = np.arange(total_rows) / max(total_rows - 1, 1)
    
    # 线性映射到疲劳分数区间
    df['Fatigue_Score'] = start_val + progress * (end_val - start_val)
    
    return df


def process_participant(participant: str) -> pd.DataFrame:
    """处理单个参与者的所有数据文件"""
    dfs = []
    
    for label in [1, 2, 3]:
        filename = f"{participant}-{label}.csv"
        filepath = DATA_DIR / filename
        
        if not filepath.exists():
            print(f"  警告: 文件不存在 {filename}")
            continue
        
        # 读取CSV
        df = pd.read_csv(filepath)
        
        # 验证Label列
        if 'Label' in df.columns:
            unique_labels = df['Label'].unique()
            if len(unique_labels) != 1 or unique_labels[0] != label:
                print(f"  警告: {filename} 的Label值不符合预期，期望{label}，实际{unique_labels}")
        
        # 清洗Pulse和Fatigue列
        if 'Pulse' in df.columns:
            df = clean_numeric_column(df, 'Pulse')
        if 'Fatigue' in df.columns:
            df = clean_numeric_column(df, 'Fatigue')
        
        # 计算连续疲劳分数
        df = calculate_fatigue_score(df, label)
        
        # 添加参与者标识（使用文件名前缀）
        df['Participant_ID'] = participant
        
        dfs.append(df)
        print(f"  处理完成: {filename} ({len(df)} 行)")
    
    if not dfs:
        return None
    
    # 纵向合并三个文件
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 应用滑动平均平滑处理
    combined_df['Fatigue_Score_Smoothed'] = combined_df['Fatigue_Score'].rolling(
        window=SMOOTHING_WINDOW, 
        min_periods=1, 
        center=True
    ).mean()
    
    return combined_df


def plot_fatigue_curve(df: pd.DataFrame, participant: str, output_path: Path):
    """绘制疲劳分数曲线验证图"""
    plt.figure(figsize=(12, 6))
    
    # 原始分数
    plt.plot(df.index, df['Fatigue_Score'], alpha=0.3, label='Original', color='blue')
    
    # 平滑后分数
    plt.plot(df.index, df['Fatigue_Score_Smoothed'], label='Smoothed', color='red', linewidth=1.5)
    
    # 标记Label边界
    label_changes = df['Label'].diff().fillna(0) != 0
    for idx in df[label_changes].index:
        plt.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Global Index')
    plt.ylabel('Fatigue Score')
    plt.title(f'Continuous Fatigue Score - Participant: {participant}')
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """主函数"""
    print("=" * 60)
    print("生成连续疲劳指数")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_participants_data = []
    
    for participant in PARTICIPANTS:
        print(f"\n处理参与者: {participant}")
        print("-" * 40)
        
        df = process_participant(participant)
        
        if df is None:
            print(f"  跳过: 无有效数据")
            continue
        
        # 保存单个参与者的文件
        output_csv = OUTPUT_DIR / f"{participant}_continuous.csv"
        df.to_csv(output_csv, index=False)
        print(f"  保存: {output_csv.name}")
        
        # 绘制验证图
        output_plot = OUTPUT_DIR / f"{participant}_fatigue_curve.png"
        plot_fatigue_curve(df, participant, output_plot)
        print(f"  图表: {output_plot.name}")
        
        all_participants_data.append(df)
    
    # 合并所有参与者数据为一个大文件
    if all_participants_data:
        print("\n" + "=" * 60)
        print("合并所有参与者数据")
        print("=" * 60)
        
        all_data = pd.concat(all_participants_data, ignore_index=True)
        all_output = OUTPUT_DIR / "all_participants_continuous.csv"
        all_data.to_csv(all_output, index=False)
        print(f"合并文件: {all_output.name} ({len(all_data)} 行)")
        
        # 统计信息
        print("\n数据统计:")
        print(f"  总行数: {len(all_data)}")
        print(f"  参与者数: {all_data['Participant_ID'].nunique()}")
        print(f"  Fatigue_Score 范围: [{all_data['Fatigue_Score'].min():.4f}, {all_data['Fatigue_Score'].max():.4f}]")
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
