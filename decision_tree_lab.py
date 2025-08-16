import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 训练数据
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

def compute_entropy(y):
    """
    计算节点的熵
    
    参数:
       y (ndarray): 表示每个样本是否可食用的numpy数组
           edible (`1`) 或 poisonous (`0`)
       
    返回:
        entropy (float): 该节点的熵
        
    """
    entropy = 0.
    
    # 检查数据是否为空
    if len(y) == 0:
        return 0
    
    # 计算p1，即可食用样本的比例
    p1 = np.sum(y) / len(y)
    
    # 如果p1为0或1，熵为0
    if p1 == 0 or p1 == 1:
        return 0
    
    # 计算熵: H(p1) = -p1*log2(p1) - (1-p1)*log2(1-p1)
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    
    return entropy

def split_dataset(X, node_indices, feature):
    """
    根据给定特征将节点数据分割为左右分支
    
    参数:
        X (ndarray):             形状为(n_samples, n_features)的数据矩阵
        node_indices (list):     包含活跃索引的列表。即在此步骤中考虑的样本
        feature (int):           要分割的特征索引
    
    返回:
        left_indices (list):     特征值 == 1的索引
        right_indices (list):    特征值 == 0的索引
    """
    
    left_indices = []
    right_indices = []
    
    # 遍历节点中的所有索引
    for index in node_indices:
        # 如果特征值为1，添加到左分支
        if X[index, feature] == 1:
            left_indices.append(index)
        # 如果特征值为0，添加到右分支
        else:
            right_indices.append(index)
        
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    """
    计算在给定特征上分割节点的信息增益
    
    参数:
        X (ndarray):            形状为(n_samples, n_features)的数据矩阵
        y (array like):         包含目标变量的n_samples列表或ndarray
        node_indices (ndarray): 包含活跃索引的列表。即在此步骤中考虑的样本
        feature (int):           要分割的特征索引
   
    返回:
        information_gain (float): 计算出的信息增益
    
    """    
    # 分割数据集
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # 一些有用的变量
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    # 计算节点熵
    node_entropy = compute_entropy(y_node)
    
    # 计算左分支熵
    left_entropy = compute_entropy(y_left)
    
    # 计算右分支熵
    right_entropy = compute_entropy(y_right)
    
    # 计算权重
    w_left = len(left_indices) / len(node_indices)
    w_right = len(right_indices) / len(node_indices)
    
    # 计算信息增益
    information_gain = node_entropy - (w_left * left_entropy + w_right * right_entropy)
    
    return information_gain

def get_best_split(X, y, node_indices):   
    """
    返回分割节点数据的最佳特征
    
    参数:
        X (ndarray):            形状为(n_samples, n_features)的数据矩阵
        y (array like):         包含目标变量的n_samples列表或ndarray
        node_indices (ndarray): 包含活跃索引的列表。即在此步骤中考虑的样本

    返回:
        best_feature (int):     最佳分割特征的索引
    """    
    
    # 一些有用的变量
    num_features = X.shape[1]
    
    best_feature = -1
    best_information_gain = -1
    
    # 遍历所有特征
    for feature in range(num_features):
        # 计算当前特征的信息增益
        information_gain = compute_information_gain(X, y, node_indices, feature)
        
        # 如果当前信息增益更大，更新最佳特征
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_feature = feature
            
    return best_feature

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    使用递归算法构建树，在每个节点将数据集分割为2个子组
    此函数仅打印树结构
    
    参数:
        X (ndarray):            形状为(n_samples, n_features)的数据矩阵
        y (array like):         包含目标变量的n_samples列表或ndarray
        node_indices (ndarray): 包含活跃索引的列表。即在此步骤中考虑的样本
        branch_name (string):   分支名称。['Root', 'Left', 'Right']
        max_depth (int):        结果树的最大深度
        current_depth (int):    当前深度。递归调用期间使用的参数
   
    """ 

    # 达到最大深度 - 停止分割
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # 否则，获取最佳分割并分割数据
    # 获取此节点上的最佳特征
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # 在最佳特征上分割数据集
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    # 继续分割左右子节点。增加当前深度
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

def main():
    """主函数，运行所有练习"""
    print("决策树实践实验室")
    print("=" * 50)
    
    # 查看变量
    print("X_train的前几个元素:")
    print(X_train[:5])
    print("X_train的类型:", type(X_train))
    print()
    
    print("y_train的前几个元素:", y_train[:5])
    print("y_train的类型:", type(y_train))
    print()
    
    # 检查变量维度
    print('X_train的形状:', X_train.shape)
    print('y_train的形状:', y_train.shape)
    print('训练样本数量 (m):', len(X_train))
    print()
    
    # 练习1: 计算熵
    print("练习1: 计算熵")
    print("-" * 30)
    root_entropy = compute_entropy(y_train)
    print("根节点的熵:", root_entropy)
    print()
    
    # 练习2: 分割数据集
    print("练习2: 分割数据集")
    print("-" * 30)
    root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    feature = 0
    
    left_indices, right_indices = split_dataset(X_train, root_indices, feature)
    print("案例1:")
    print("左索引:", left_indices)
    print("右索引:", right_indices)
    print()
    
    # 案例2
    root_indices_subset = [0, 2, 4, 6, 8]
    left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)
    print("案例2:")
    print("左索引:", left_indices)
    print("右索引:", right_indices)
    print()
    
    # 练习3: 计算信息增益
    print("练习3: 计算信息增益")
    print("-" * 30)
    info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
    print("在棕色菌盖上分割根节点的信息增益:", info_gain0)
    
    info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
    print("在锥形菌柄形状上分割根节点的信息增益:", info_gain1)
    
    info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
    print("在独生上分割根节点的信息增益:", info_gain2)
    print()
    
    # 练习4: 获取最佳分割
    print("练习4: 获取最佳分割")
    print("-" * 30)
    best_feature = get_best_split(X_train, y_train, root_indices)
    print("最佳分割特征:", best_feature)
    print()
    
    # 构建树
    print("构建决策树")
    print("-" * 30)
    tree = []
    build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
    
    print("\n决策树构建完成！")
    print("最佳分割特征顺序:")
    print("- 根节点: 特征2 (独生)")
    print("- 左分支: 特征0 (棕色菌盖)")
    print("- 右分支: 特征1 (锥形菌柄形状)")

if __name__ == "__main__":
    main()
