#!/usr/bin/env python3
"""
可视化评估结果脚本
用于生成k=2实验的对比图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文避免中文问题
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """加载评估数据"""
    try:
        with open('corrected_visualization_metrics.json', 'r', encoding='utf-8') as f:
            vis_data = json.load(f)
        print("✓ 修正后可视化数据加载成功")
        return vis_data
    except FileNotFoundError:
        print("❌ 未找到corrected_visualization_metrics.json")
        return None

def plot_main_comparison(vis_data):
    """绘制主要对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    algorithms = vis_data['algorithms']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 1. 平均胜率对比
    ax = axes[0, 0]
    win_rates = [wr * 100 for wr in vis_data['avg_win_rates']]
    bars = ax.bar(algorithms, win_rates, color=colors, alpha=0.8)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, label='Theoretical (50%)')
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Average Win Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 60])
    
    # 在柱子上添加数值
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.legend()
    
    # 2. ROI对比
    ax = axes[0, 1]
    rois = vis_data['avg_rois']
    bars = ax.bar(algorithms, rois, color=colors, alpha=0.8)
    ax.set_ylabel('ROI (%)', fontsize=12)
    ax.set_title('Average ROI', fontsize=14, fontweight='bold')
    
    for bar, roi in zip(bars, rois):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{roi:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. 不对称度对比（核心指标）
    ax = axes[0, 2]
    wr_diffs = [diff * 100 for diff in vis_data['win_rate_differences']]
    
    # 根据差异程度着色
    bar_colors = []
    for diff in wr_diffs:
        if diff < 5:
            bar_colors.append('#2ECC71')  # 绿色 - 优秀
        elif diff < 15:
            bar_colors.append('#F39C12')  # 橙色 - 可接受
        else:
            bar_colors.append('#E74C3C')  # 红色 - 问题严重
    
    bars = ax.bar(algorithms, wr_diffs, color=bar_colors, alpha=0.8)
    ax.axhline(y=5, color='green', linestyle='--', label='Good (<5%)')
    ax.axhline(y=15, color='orange', linestyle='--', label='Acceptable (<15%)')
    ax.set_ylabel('Win Rate Difference (%)', fontsize=12)
    ax.set_title('Asymmetry Analysis (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(wr_diffs) + 10])
    
    for bar, diff in zip(bars, wr_diffs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{diff:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.legend()
    
    # 4. 个体胜率对比
    ax = axes[1, 0]
    x = np.arange(len(algorithms))
    width = 0.35
    
    l0_win_rates = [vis_data['individual_performance'][algo]['Learning_0']['win_rate'] * 100 
                     for algo in algorithms]
    l1_win_rates = [vis_data['individual_performance'][algo]['Learning_1']['win_rate'] * 100 
                     for algo in algorithms]
    
    bars1 = ax.bar(x - width/2, l0_win_rates, width, label='Learning_0', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, l1_win_rates, width, label='Learning_1', color='#9B59B6', alpha=0.8)
    
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Individual Agent Win Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. 个体ROI对比
    ax = axes[1, 1]
    
    l0_rois = [vis_data['individual_performance'][algo]['Learning_0']['roi'] for algo in algorithms]
    l1_rois = [vis_data['individual_performance'][algo]['Learning_1']['roi'] for algo in algorithms]
    
    bars1 = ax.bar(x - width/2, l0_rois, width, label='Learning_0', color='#3498DB', alpha=0.8)
    bars2 = ax.bar(x + width/2, l1_rois, width, label='Learning_1', color='#9B59B6', alpha=0.8)
    
    ax.set_ylabel('ROI (%)', fontsize=12)
    ax.set_title('Individual Agent ROI', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    # 6. 性能雷达图
    ax = axes[1, 2]
    ax.axis('off')  # 暂时关闭这个子图
    
    # 添加总结文本
    summary = vis_data.get('summary_stats', {})
    if summary:
        summary_text = f"""
Experiment Summary (k=2 Multi-Agent RL)

Best Algorithm: {summary.get('best_algorithm', 'N/A')}
Highest Win Rate: {summary.get('best_win_rate', 0) * 100:.1f}%

Most Balanced: {summary.get('most_balanced_algorithm', 'N/A')}
Min Difference: {summary.get('min_win_rate_difference', 0) * 100:.1f}%

Avg Performance vs Theoretical: {summary.get('avg_performance_vs_theoretical', 0) * 100:.1f}%
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, 
                transform=ax.transAxes, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('k=2 Multi-Agent RL Algorithm Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('k2_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 主对比图已保存: k2_algorithm_comparison.png")
    plt.show()

def plot_performance_matrix(vis_data):
    """绘制性能矩阵热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = vis_data['algorithms']
    metrics = ['Win Rate', 'ROI', 'Win Rate Diff', 'L0 Win Rate', 'L1 Win Rate']
    
    # 构建数据矩阵
    data_matrix = []
    for algo in algorithms:
        row = [
            vis_data['avg_win_rates'][algorithms.index(algo)] * 100,
            vis_data['avg_rois'][algorithms.index(algo)],
            vis_data['win_rate_differences'][algorithms.index(algo)] * 100,
            vis_data['individual_performance'][algo]['Learning_0']['win_rate'] * 100,
            vis_data['individual_performance'][algo]['Learning_1']['win_rate'] * 100
        ]
        data_matrix.append(row)
    
    # 转置矩阵以便算法为列，指标为行
    data_matrix = np.array(data_matrix).T
    
    # 创建热力图
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
    
    # 设置标签
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(algorithms)
    ax.set_yticklabels(metrics)
    
    # 旋转顶部标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标注
    for i in range(len(metrics)):
        for j in range(len(algorithms)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                         ha="center", va="center", color="black", fontsize=11)
    
    ax.set_title("Algorithm Performance Matrix", fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='性能分数')
    
    plt.tight_layout()
    plt.savefig('k2_performance_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ 性能矩阵图已保存: k2_performance_matrix.png")
    plt.show()

def generate_latex_table(vis_data):
    """生成LaTeX格式的表格（用于论文）"""
    algorithms = vis_data['algorithms']
    
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{k=2 多智能体强化学习算法性能对比}
\\begin{tabular}{lccccc}
\\hline
算法 & 平均胜率(\\%) & 平均ROI(\\%) & L0胜率(\\%) & L1胜率(\\%) & 胜率差异(\\%) \\\\
\\hline
"""
    
    for algo in algorithms:
        avg_wr = vis_data['avg_win_rates'][algorithms.index(algo)] * 100
        avg_roi = vis_data['avg_rois'][algorithms.index(algo)]
        l0_wr = vis_data['individual_performance'][algo]['Learning_0']['win_rate'] * 100
        l1_wr = vis_data['individual_performance'][algo]['Learning_1']['win_rate'] * 100
        wr_diff = vis_data['win_rate_differences'][algorithms.index(algo)] * 100
        
        latex_table += f"{algo} & {avg_wr:.1f} & {avg_roi:.1f} & {l0_wr:.1f} & {l1_wr:.1f} & {wr_diff:.1f} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
\\label{tab:k2_comparison}
\\end{table}
"""
    
    with open('k2_results_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("✓ LaTeX表格已保存: k2_results_table.tex")
    print("\nLaTeX表格内容:")
    print(latex_table)

def main():
    """主函数"""
    print("="*60)
    print("k=2 多智能体强化学习结果可视化")
    print("="*60)
    
    # 加载数据
    vis_data = load_data()
    if vis_data is None:
        return
    
    # 打印基本信息
    print(f"\n检测到算法: {', '.join(vis_data['algorithms'])}")
    print(f"理论胜率基准: {vis_data['theoretical_benchmarks']['win_rate'] * 100:.1f}%")
    print(f"环境描述: {vis_data['theoretical_benchmarks']['description']}")
    
    # 生成图表
    print("\n开始生成可视化图表...")
    plot_main_comparison(vis_data)
    plot_performance_matrix(vis_data)
    
    # 生成LaTeX表格
    generate_latex_table(vis_data)
    
    print("\n所有可视化完成！")
    print("生成的文件：")
    print("  - k2_algorithm_comparison.png (主对比图)")
    print("  - k2_performance_matrix.png (性能矩阵)")
    print("  - k2_results_table.tex (LaTeX表格)")

if __name__ == "__main__":
    main()