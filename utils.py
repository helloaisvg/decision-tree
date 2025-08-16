import numpy as np
import matplotlib.pyplot as plt

def generate_split_viz(root_indices, left_indices, right_indices, feature):
    """
    生成数据集分割的可视化
    
    参数:
        root_indices: 根节点的索引
        left_indices: 左分支的索引
        right_indices: 右分支的索引
        feature: 分割特征
    """
    feature_names = ["棕色菌盖", "锥形菌柄", "独生"]
    
    print(f"分割特征: {feature_names[feature]}")
    print(f"左分支 ({len(left_indices)} 个样本): 特征值 = 1")
    print(f"右分支 ({len(right_indices)} 个样本): 特征值 = 0")
    
    # 显示分割后的数据
    if len(left_indices) > 0:
        print("左分支样本:")
        for idx in left_indices:
            print(f"  样本{idx}: {X_train[idx]} -> 标签: {y_train[idx]}")
    
    if len(right_indices) > 0:
        print("右分支样本:")
        for idx in right_indices:
            print(f"  样本{idx}: {X_train[idx]} -> 标签: {y_train[idx]}")

def generate_tree_viz(root_indices, y_train, tree):
    """
    生成决策树结构的可视化
    
    参数:
        root_indices: 根节点的索引
        y_train: 训练标签
        tree: 树结构列表
    """
    print("\n决策树结构可视化:")
    print("=" * 50)
    
    # 计算根节点的统计信息
    root_edible = sum(y_train[i] for i in root_indices)
    root_total = len(root_indices)
    
    print(f"根节点: {root_total} 个样本, {root_edible} 个可食用, {root_total - root_edible} 个有毒")
    
    # 显示树的分支
    for i, (left_indices, right_indices, feature) in enumerate(tree):
        feature_names = ["棕色菌盖", "锥形菌柄", "独生"]
        
        left_edible = sum(y_train[j] for j in left_indices)
        right_edible = sum(y_train[j] for j in right_indices)
        
        print(f"\n深度 {i+1}:")
        print(f"  特征: {feature_names[feature]}")
        print(f"  左分支: {len(left_indices)} 个样本, {left_edible} 个可食用")
        print(f"  右分支: {len(right_indices)} 个样本, {right_edible} 个可食用")

def plot_entropy_curve():
    """
    绘制熵函数曲线
    """
    p1 = np.linspace(0.01, 0.99, 100)
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(p1, entropy, 'b-', linewidth=2)
    plt.xlabel('可食用样本比例 (p₁)')
    plt.ylabel('熵 H(p₁)')
    plt.title('熵函数曲线')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='最大熵 = 1')
    plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='p₁ = 0.5')
    plt.legend()
    plt.show()

def plot_information_gain_comparison():
    """
    绘制不同特征的信息增益比较
    """
    root_indices = list(range(10))
    features = [0, 1, 2]
    feature_names = ["棕色菌盖", "锥形菌柄", "独生"]
    
    info_gains = []
    for feature in features:
        # 这里需要导入compute_information_gain函数
        try:
            from decision_tree_lab import compute_information_gain
            info_gain = compute_information_gain(X_train, y_train, root_indices, feature)
            info_gains.append(info_gain)
        except ImportError:
            # 如果无法导入，使用预计算的值
            info_gains = [0.0349, 0.1245, 0.2781]
            break
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_names, info_gains, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.xlabel('特征')
    plt.ylabel('信息增益')
    plt.title('不同特征的信息增益比较')
    plt.ylim(0, max(info_gains) * 1.1)
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, info_gains):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.show()

def print_dataset_summary():
    """
    打印数据集摘要信息
    """
    print("蘑菇数据集摘要")
    print("=" * 30)
    print(f"总样本数: {len(X_train)}")
    print(f"特征数: {X_train.shape[1]}")
    print(f"可食用样本: {sum(y_train)}")
    print(f"有毒样本: {len(y_train) - sum(y_train)}")
    print(f"可食用比例: {sum(y_train)/len(y_train):.1%}")
    
    print("\n特征说明:")
    print("特征0: 棕色菌盖 (1=棕色, 0=红色)")
    print("特征1: 锥形菌柄 (1=锥形, 0=扩大)")
    print("特征2: 独生 (1=是, 0=否)")
    
    print("\n前5个样本:")
    for i in range(5):
        features = X_train[i]
        label = "可食用" if y_train[i] == 1 else "有毒"
        print(f"样本{i}: [{features[0]}, {features[1]}, {features[2]}] -> {label}")

# 全局变量（用于可视化函数）
X_train = None
y_train = None

def set_dataset(X, y):
    """设置数据集供可视化函数使用"""
    global X_train, y_train
    X_train = X
    y_train = y
