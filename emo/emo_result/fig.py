import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
labels_path = 'labels.txt'
results_dir = './'  # 修改为存放模型结果文件的目录

# 读取GT标签
with open(labels_path, 'r') as f:
    labels = np.array(eval(f.read().strip()))  # 使用 eval 解析列表

# 获取模型结果文件列表
result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt') and f != 'labels.txt']

# 创建一个目录用于保存图表和指标
os.makedirs('metrics_visualization', exist_ok=True)
os.makedirs('metrics_results', exist_ok=True)

# 遍历所有模型文件，计算指标并生成可视化图表
for result_file in result_files:
    # 读取模型结果
    result_path = os.path.join(results_dir, result_file)
    with open(result_path, 'r') as f:
        predictions = np.array(eval(f.read().strip()))  # 使用 eval 解析列表

    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')

    # 打印结果
    print(f'Model: {result_file}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Classification Report:')
    print(classification_report(labels, predictions))
    print('---')

    # 将指标存储为CSV文件
    metrics = {
        'Model': result_file,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'metrics_results/{result_file.replace(".txt", "")}_metrics.csv', index=False)

    # 生成混淆矩阵
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {result_file}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'metrics_visualization/cm_{result_file.replace(".txt", "")}.png')
    plt.close()

    # 保存其他指标可视化图表
    metrics_df.plot(kind='bar', x='Model', legend=True)
    plt.title(f'Metrics for {result_file}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'metrics_visualization/metrics_{result_file.replace(".txt", "")}.png')
    plt.close()

print('Metrics calculation, storage, and visualization completed.')
