import numpy as np
from decision_tree_lab import *

# 测试数据
X = np.array([[1,1,1], [1,0,1], [0,1,0], [0,0,0]])
y = np.array([1, 1, 0, 0])
node_indices = [0, 1, 2, 3]

print("测试数据:")
print("X:", X)
print("y:", y)
print("node_indices:", node_indices)
print()

# 手动计算信息增益
print("手动计算信息增益:")
print("-" * 30)

# 1. 计算根节点熵
root_entropy = compute_entropy(y)
print(f"根节点熵: {root_entropy}")

# 2. 测试特征0
print(f"\n特征0 (第一列):")
left_indices, right_indices = split_dataset(X, node_indices, 0)
print(f"左分支索引: {left_indices}")
print(f"右分支索引: {right_indices}")

if len(left_indices) > 0:
    y_left = y[left_indices]
    left_entropy = compute_entropy(y_left)
    print(f"左分支标签: {y_left}")
    print(f"左分支熵: {left_entropy}")
else:
    left_entropy = 0
    print("左分支为空")

if len(right_indices) > 0:
    y_right = y[right_indices]
    right_entropy = compute_entropy(y_right)
    print(f"右分支标签: {y_right}")
    print(f"右分支熵: {right_entropy}")
else:
    right_entropy = 0
    print("右分支为空")

w_left = len(left_indices) / len(node_indices)
w_right = len(right_indices) / len(node_indices)
print(f"权重: w_left={w_left:.2f}, w_right={w_right:.2f}")

info_gain_0 = root_entropy - (w_left * left_entropy + w_right * right_entropy)
print(f"特征0信息增益: {info_gain_0}")

# 3. 测试特征1
print(f"\n特征1 (第二列):")
left_indices, right_indices = split_dataset(X, node_indices, 1)
print(f"左分支索引: {left_indices}")
print(f"右分支索引: {right_indices}")

if len(left_indices) > 0:
    y_left = y[left_indices]
    left_entropy = compute_entropy(y_left)
    print(f"左分支标签: {y_left}")
    print(f"左分支熵: {left_entropy}")
else:
    left_entropy = 0
    print("左分支为空")

if len(right_indices) > 0:
    y_right = y[right_indices]
    right_entropy = compute_entropy(y_right)
    print(f"右分支标签: {y_right}")
    print(f"右分支熵: {right_entropy}")
else:
    right_entropy = 0
    print("右分支为空")

w_left = len(left_indices) / len(node_indices)
w_right = len(right_indices) / len(node_indices)
print(f"权重: w_left={w_left:.2f}, w_right={w_right:.2f}")

info_gain_1 = root_entropy - (w_left * left_entropy + w_right * right_entropy)
print(f"特征1信息增益: {info_gain_1}")

print(f"\n总结:")
print(f"特征0信息增益: {info_gain_0}")
print(f"特征1信息增益: {info_gain_1}")
