import numpy as np
import matplotlib.pyplot as plt
from decision_tree_lab import *
from utils import *

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_complete_demo():
    """运行完整的决策树演示"""
    print("🌳 决策树实践实验室 - 完整演示")
    print("=" * 60)
    
    # 设置数据集供可视化函数使用
    set_dataset(X_train, y_train)
    
    # 1. 数据集概览
    print("\n📊 数据集概览")
    print("-" * 30)
    print_dataset_summary()
    
    # 2. 运行所有练习
    print("\n🔬 运行所有练习")
    print("-" * 30)
    main()
    
    # 3. 可视化熵函数
    print("\n📈 可视化熵函数")
    print("-" * 30)
    print("正在绘制熵函数曲线...")
    plot_entropy_curve()
    
    # 4. 可视化信息增益比较
    print("\n📊 可视化信息增益比较")
    print("-" * 30)
    print("正在绘制信息增益比较图...")
    plot_information_gain_comparison()
    
    # 5. 详细的分割分析
    print("\n🔍 详细的分割分析")
    print("-" * 30)
    root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for feature in range(3):
        feature_names = ["棕色菌盖", "锥形菌柄", "独生"]
        print(f"\n分析特征 {feature}: {feature_names[feature]}")
        
        left_indices, right_indices = split_dataset(X_train, root_indices, feature)
        info_gain = compute_information_gain(X_train, y_train, root_indices, feature)
        
        print(f"  信息增益: {info_gain:.4f}")
        print(f"  左分支样本数: {len(left_indices)}")
        print(f"  右分支样本数: {len(right_indices)}")
        
        # 计算每个分支的纯度
        if len(left_indices) > 0:
            left_edible = sum(y_train[i] for i in left_indices)
            left_purity = left_edible / len(left_indices)
            print(f"  左分支可食用比例: {left_purity:.2%}")
        
        if len(right_indices) > 0:
            right_edible = sum(y_train[i] for i in right_indices)
            right_purity = right_edible / len(right_indices)
            print(f"  右分支可食用比例: {right_purity:.2%}")
    
    # 6. 决策树结构分析
    print("\n🌳 决策树结构分析")
    print("-" * 30)
    tree = []
    
    def build_tree_with_analysis(X, y, node_indices, branch_name, max_depth, current_depth):
        """构建树并进行分析"""
        if current_depth == max_depth:
            formatting = "  " * current_depth
            edible_count = sum(y[i] for i in node_indices)
            total_count = len(node_indices)
            purity = edible_count / total_count if total_count > 0 else 0
            print(f"{formatting}📄 {branch_name} 叶节点: {total_count} 个样本, 可食用比例: {purity:.2%}")
            return
        
        best_feature = get_best_split(X, y, node_indices)
        feature_names = ["棕色菌盖", "锥形菌柄", "独生"]
        
        formatting = "  " * current_depth
        print(f"{formatting}🔀 {branch_name} 深度 {current_depth}: 分割特征 {best_feature} ({feature_names[best_feature]})")
        
        left_indices, right_indices = split_dataset(X, node_indices, best_feature)
        tree.append((left_indices, right_indices, best_feature))
        
        build_tree_with_analysis(X, y, left_indices, "左", max_depth, current_depth+1)
        build_tree_with_analysis(X, y, right_indices, "右", max_depth, current_depth+1)
    
    build_tree_with_analysis(X_train, y_train, root_indices, "根", max_depth=2, current_depth=0)
    
    # 7. 预测示例
    print("\n🔮 预测示例")
    print("-" * 30)
    
    def predict_sample(sample, tree, root_indices):
        """使用决策树预测单个样本"""
        current_indices = root_indices.copy()
        
        for left_indices, right_indices, feature in tree:
            if sample[feature] == 1:
                current_indices = [i for i in current_indices if i in left_indices]
            else:
                current_indices = [i for i in current_indices if i in right_indices]
        
        if len(current_indices) == 0:
            return "无法确定"
        
        # 计算当前节点的多数类
        edible_count = sum(y_train[i] for i in current_indices)
        total_count = len(current_indices)
        
        if edible_count > total_count / 2:
            return "可食用"
        elif edible_count < total_count / 2:
            return "有毒"
        else:
            return "无法确定"
    
    # 测试几个样本
    test_samples = [
        [1, 1, 1],  # 棕色菌盖, 锥形菌柄, 独生
        [0, 0, 0],  # 红色菌盖, 扩大菌柄, 非独生
        [1, 0, 1],  # 棕色菌盖, 扩大菌柄, 独生
    ]
    
    sample_descriptions = [
        "棕色菌盖 + 锥形菌柄 + 独生",
        "红色菌盖 + 扩大菌柄 + 非独生", 
        "棕色菌盖 + 扩大菌柄 + 独生"
    ]
    
    for i, (sample, desc) in enumerate(zip(test_samples, sample_descriptions)):
        prediction = predict_sample(sample, tree, root_indices)
        print(f"样本 {i+1}: {desc}")
        print(f"  特征值: {sample}")
        print(f"  预测结果: {prediction}")
        print()
    
    print("🎉 演示完成！")
    print("\n💡 提示:")
    print("- 运行 'python test_decision_tree.py' 来验证实现")
    print("- 修改 'decision_tree_lab.py' 中的参数进行实验")
    print("- 查看 'README.md' 了解项目详情")

if __name__ == "__main__":
    run_complete_demo()
