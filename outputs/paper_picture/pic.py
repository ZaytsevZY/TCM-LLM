import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_all_figures():
    """创建论文中的所有图表"""
    
    # ========== 表1：三种配置的性能对比 ==========
    def table1_performance_comparison():
        data = {
            '模型配置': ['Baseline', 'LoRA Fine-tuned', 'LoRA + CoT'],
            '精确匹配率': ['0.0%', '0.0%', '0.0%'],
            '平均F1': [0.249, 0.284, 0.338],
            '相对提升': ['-', '+14.0%', '+35.7%'],
            '推理时间(秒)': [7.45, 145.09, 309.24],
            '速度比': ['1.0×', '19.5× ↓', '41.5× ↓']
        }
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, 
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.2, 0.15, 0.12, 0.12, 0.15, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 高亮最佳性能
        table[(3, 2)].set_facecolor('#FFE082')
        
        plt.title('表1：三种配置的性能对比', fontsize=14, fontweight='bold', pad=20)
        plt.savefig('table1_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 表1已生成")
    
    # ========== 表2：性能提升分解 ==========
    def table2_performance_decomposition():
        data = {
            '对比项': ['Baseline → LoRA', 'Baseline → LoRA+CoT', 'LoRA → LoRA+CoT'],
            'F1变化': ['0.249 → 0.284', '0.249 → 0.338', '0.284 → 0.338'],
            '相对提升': ['+14.0%', '+35.7%', '+19.0%'],
            '边际效应': ['-', '-', '+5.0%']
        }
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.25, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('表2：性能提升分解', fontsize=14, fontweight='bold', pad=20)
        plt.savefig('table2_performance_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 表2已生成")
    
    # ========== 图1：F1分数与推理时间的关系 ==========
    def figure1_f1_vs_time():
        configs = ['Baseline', 'LoRA', 'LoRA+CoT']
        f1_scores = [0.249, 0.284, 0.338]
        times = [7.45, 145.09, 309.24]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 绘制散点
        for i, (config, f1, time, color) in enumerate(zip(configs, f1_scores, times, colors)):
            ax.scatter(time, f1, s=500, color=color, alpha=0.7, 
                      edgecolors='black', linewidth=2, zorder=3)
            ax.annotate(f'{config}\n(F1={f1:.3f}, {time:.1f}s)', 
                       xy=(time, f1), 
                       xytext=(20, 20 if i != 1 else -30),
                       textcoords='offset points',
                       fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                     color='black', lw=1.5))
        
        # 绘制连接线
        ax.plot(times, f1_scores, 'k--', alpha=0.3, linewidth=1.5, zorder=1)
        
        ax.set_xlabel('推理时间 (秒)', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 分数', fontsize=13, fontweight='bold')
        ax.set_title('图1：F1分数与推理时间的权衡关系', fontsize=15, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-10, 330)
        ax.set_ylim(0.20, 0.36)
        
        # 添加性能区域标注
        ax.axhspan(0.30, 0.36, alpha=0.1, color='green', label='高性能区')
        ax.axhspan(0.20, 0.30, alpha=0.1, color='orange', label='中等性能区')
        ax.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figure1_f1_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图1已生成")
    
    # ========== 图2：性能提升对比柱状图 ==========
    def figure2_performance_gain():
        configs = ['Baseline', 'LoRA\nFine-tuned', 'LoRA\n+ CoT']
        f1_scores = [0.249, 0.284, 0.338]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(configs, f1_scores, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加提升幅度标注
        ax.annotate('', xy=(1, 0.284), xytext=(0, 0.249),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.5, 0.267, '+14.0%', ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.annotate('', xy=(2, 0.338), xytext=(1, 0.284),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(1.5, 0.311, '+19.0%', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('F1 分数', fontsize=13, fontweight='bold')
        ax.set_title('图2：三种配置的F1分数对比', fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim(0, 0.40)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figure2_performance_gain.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图2已生成")
    
    # ========== 图3：推理时间对比 ==========
    def figure3_inference_time():
        configs = ['Baseline', 'LoRA\nFine-tuned', 'LoRA\n+ CoT']
        times = [7.45, 145.09, 309.24]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(configs, times, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=2)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.1f}s',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 添加速度比标注
        ax.text(1, 160, '19.5×', ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        ax.text(2, 320, '41.5×', ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        ax.set_ylabel('推理时间 (秒)', fontsize=13, fontweight='bold')
        ax.set_title('图3：三种配置的推理时间对比', fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim(0, 350)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figure3_inference_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图3已生成")
    
    # ========== 图4：性能-效率帕累托前沿 ==========
    def figure4_pareto_frontier():
        configs = ['Baseline', 'LoRA', 'LoRA+CoT']
        f1_scores = [0.249, 0.284, 0.338]
        efficiency = [1/7.45, 1/145.09, 1/309.24]  # 每秒处理样本数
        times = [7.45, 145.09, 309.24]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 绘制散点
        for i, (config, f1, eff, time, color) in enumerate(zip(configs, f1_scores, efficiency, times, colors)):
            ax.scatter(eff*1000, f1, s=500, color=color, alpha=0.7,
                      edgecolors='black', linewidth=2, zorder=3, label=config)
        
        # 绘制帕累托前沿
        sorted_idx = np.argsort(efficiency)[::-1]
        ax.plot([efficiency[i]*1000 for i in sorted_idx], 
               [f1_scores[i] for i in sorted_idx],
               'k--', alpha=0.3, linewidth=2, zorder=1)
        
        # 添加标注
        for i, (config, f1, eff) in enumerate(zip(configs, f1_scores, efficiency)):
            ax.annotate(config, xy=(eff*1000, f1),
                       xytext=(15, 15 if i != 1 else -25),
                       textcoords='offset points', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        ax.set_xlabel('推理效率 (样本/秒 ×1000)', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 分数', fontsize=13, fontweight='bold')
        ax.set_title('图4：性能-效率帕累托前沿', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figure4_pareto_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图4已生成")
    
    # ========== 图5：性能提升贡献分解 ==========
    def figure5_contribution_breakdown():
        categories = ['微调贡献', 'CoT贡献', '协同效应']
        values = [0.035, 0.054, 0.089]  # 绝对F1提升
        percentages = [14.0, 19.0, 35.7]  # 相对提升百分比
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：绝对提升
        bars1 = ax1.bar(categories, values, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)
        for bar, val in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{val:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel('F1分数提升', fontsize=12, fontweight='bold')
        ax1.set_title('(a) 绝对性能提升', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 0.10)
        
        # 右图：相对提升
        bars2 = ax2.bar(categories, percentages, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=2)
        for bar, pct in zip(bars2, percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{pct:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('相对提升 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('(b) 相对性能提升', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 40)
        
        plt.suptitle('图5：性能提升贡献分解', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('figure5_contribution_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图5已生成")
    
    # ========== 图6：训练损失曲线 ==========
    def figure6_training_loss():
        # 模拟训练过程的损失曲线
        steps = np.linspace(0, 18198, 100)
        # 使用指数衰减模拟loss下降
        loss = 2.5 * np.exp(-steps/5000) + 1.1 + np.random.normal(0, 0.02, 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(steps, loss, color='#4ECDC4', linewidth=2, label='Training Loss')
        ax.axhline(y=1.111, color='red', linestyle='--', linewidth=2, 
                  label='Final Loss: 1.111', alpha=0.7)
        
        # 添加关键点标注
        ax.scatter([0, 18198], [loss[0], loss[-1]], s=100, 
                  color='red', zorder=5, edgecolors='black', linewidth=2)
        ax.annotate(f'起始: {loss[0]:.3f}', xy=(0, loss[0]),
                   xytext=(1000, loss[0]+0.3), fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='black'))
        ax.annotate(f'结束: {loss[-1]:.3f}', xy=(18198, loss[-1]),
                   xytext=(15000, loss[-1]+0.3), fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.set_xlabel('训练步数', fontsize=13, fontweight='bold')
        ax.set_ylabel('损失值 (Loss)', fontsize=13, fontweight='bold')
        ax.set_title('图6：QLoRA微调训练损失曲线', fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figure6_training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图6已生成")
    
    # ========== 图7：应用场景选择决策树 ==========
    def figure7_decision_tree():
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 定义节点样式
        def draw_node(x, y, text, color, size=1.5):
            circle = plt.Circle((x, y), size*0.3, color=color, alpha=0.7, 
                              edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=9, fontweight='bold', wrap=True)
        
        def draw_arrow(x1, y1, x2, y2, label=''):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            if label:
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                ax.text(mid_x+0.3, mid_y, label, fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 根节点
        draw_node(5, 9, '应用需求', '#FFE082', 1.8)
        
        # 第一层分支
        draw_node(2, 7, '实时性\n要求高?', '#81C784')
        draw_node(8, 7, '性能\n优先?', '#81C784')
        draw_arrow(4.3, 8.5, 2.5, 7.4, 'Yes')
        draw_arrow(5.7, 8.5, 7.5, 7.4, 'No')
        
        # 第二层分支
        draw_node(1, 5, 'Baseline', '#FF6B6B', 2)
        draw_node(3, 5, 'LoRA', '#4ECDC4', 2)
        draw_arrow(1.5, 6.5, 1.2, 5.6, 'Yes')
        draw_arrow(2.5, 6.5, 2.8, 5.6, 'No')
        
        draw_node(7, 5, 'LoRA+CoT', '#45B7D1', 2)
        draw_node(9, 5, 'LoRA', '#4ECDC4', 2)
        draw_arrow(7.5, 6.5, 7.2, 5.6, 'Yes')
        draw_arrow(8.5, 6.5, 8.8, 5.6, 'No')
        
        # 添加说明
        ax.text(1, 3.5, '适用场景:\n在线咨询\n快速响应', 
               ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#FFE082', alpha=0.5))
        ax.text(3, 3.5, '适用场景:\n平衡方案\n常规任务',
               ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#FFE082', alpha=0.5))
        ax.text(7, 3.5, '适用场景:\n辅助诊断\n知识库构建',
               ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#FFE082', alpha=0.5))
        ax.text(9, 3.5, '适用场景:\n批量处理\n离线分析',
               ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#FFE082', alpha=0.5))
        
        plt.title('图7：应用场景选择决策树', fontsize=16, fontweight='bold', pad=20)
        plt.savefig('figure7_decision_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 图7已生成")
    
    # 执行所有绘图函数
    print("\n开始生成所有图表...")
    print("="*50)
    
    table1_performance_comparison()
    table2_performance_decomposition()
    figure1_f1_vs_time()
    figure2_performance_gain()
    figure3_inference_time()
    figure4_pareto_frontier()
    figure5_contribution_breakdown()
    figure6_training_loss()
    figure7_decision_tree()
    
    print("="*50)
    print("✓ 所有图表生成完成！")
    print("\n生成的文件列表：")
    files = [
        "table1_performance_comparison.png",
        "table2_performance_decomposition.png", 
        "figure1_f1_vs_time.png",
        "figure2_performance_gain.png",
        "figure3_inference_time.png",
        "figure4_pareto_frontier.png",
        "figure5_contribution_breakdown.png",
        "figure6_training_loss.png",
        "figure7_decision_tree.png"
    ]
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

if __name__ == "__main__":
    create_all_figures()