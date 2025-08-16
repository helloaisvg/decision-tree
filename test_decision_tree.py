import numpy as np
from decision_tree_lab import *

def test_compute_entropy():
    """测试熵计算函数"""
    print("测试熵计算函数...")
    
    # 测试用例1: 所有样本都是可食用的
    y1 = np.array([1, 1, 1, 1, 1])
    entropy1 = compute_entropy(y1)
    expected1 = 0.0
    assert abs(entropy1 - expected1) < 1e-10, f"期望熵为{expected1}, 但得到{entropy1}"
    print("✓ 测试用例1通过: 所有样本可食用")
    
    # 测试用例2: 所有样本都是有毒的
    y2 = np.array([0, 0, 0, 0, 0])
    entropy2 = compute_entropy(y2)
    expected2 = 0.0
    assert abs(entropy2 - expected2) < 1e-10, f"期望熵为{expected2}, 但得到{entropy2}"
    print("✓ 测试用例2通过: 所有样本有毒")
    
    # 测试用例3: 一半样本可食用，一半有毒
    y3 = np.array([1, 1, 1, 0, 0, 0])
    entropy3 = compute_entropy(y3)
    expected3 = 1.0
    assert abs(entropy3 - expected3) < 1e-10, f"期望熵为{expected3}, 但得到{entropy3}"
    print("✓ 测试用例3通过: 一半样本可食用")
    
    # 测试用例4: 空数组
    y4 = np.array([])
    entropy4 = compute_entropy(y4)
    expected4 = 0.0
    assert entropy4 == expected4, f"期望熵为{expected4}, 但得到{entropy4}"
    print("✓ 测试用例4通过: 空数组")
    
    print("所有熵计算测试通过！\n")

def test_split_dataset():
    """测试数据集分割函数"""
    print("测试数据集分割函数...")
    
    X = np.array([[1,1,1], [1,0,1], [0,1,0], [0,0,1]])
    node_indices = [0, 1, 2, 3]
    
    # 测试特征0
    left_indices, right_indices = split_dataset(X, node_indices, 0)
    expected_left = [0, 1]
    expected_right = [2, 3]
    
    assert set(left_indices) == set(expected_left), f"期望左索引为{expected_left}, 但得到{left_indices}"
    assert set(right_indices) == set(expected_right), f"期望右索引为{expected_right}, 但得到{right_indices}"
    print("✓ 特征0分割测试通过")
    
    # 测试特征1
    left_indices, right_indices = split_dataset(X, node_indices, 1)
    expected_left = [0, 2]
    expected_right = [1, 3]
    
    assert set(left_indices) == set(expected_left), f"期望左索引为{expected_left}, 但得到{left_indices}"
    assert set(right_indices) == set(expected_right), f"期望右索引为{expected_right}, 但得到{right_indices}"
    print("✓ 特征1分割测试通过")
    
    print("所有数据集分割测试通过！\n")

def test_compute_information_gain():
    """测试信息增益计算函数"""
    print("测试信息增益计算函数...")
    
    # 使用更好的测试数据，确保有信息增益
    X = np.array([[1,1,1], [1,0,1], [0,1,0], [0,0,0]])
    y = np.array([1, 1, 0, 0])
    node_indices = [0, 1, 2, 3]
    
    # 测试特征0的信息增益
    info_gain = compute_information_gain(X, y, node_indices, 0)
    assert info_gain > 0, "信息增益应该大于0"
    print("✓ 特征0信息增益测试通过")
    
    # 测试特征1的信息增益
    info_gain = compute_information_gain(X, y, node_indices, 1)
    assert info_gain > 0, "信息增益应该大于0"
    print("✓ 特征1信息增益测试通过")
    
    print("所有信息增益测试通过！\n")

def test_get_best_split():
    """测试最佳分割特征选择函数"""
    print("测试最佳分割特征选择函数...")
    
    X = np.array([[1,1,1], [1,0,1], [0,1,0], [0,0,1]])
    y = np.array([1, 1, 0, 0])
    node_indices = [0, 1, 2, 3]
    
    best_feature = get_best_split(X, y, node_indices)
    assert best_feature in [0, 1, 2], f"最佳特征应该在[0,1,2]范围内，但得到{best_feature}"
    print(f"✓ 最佳分割特征测试通过: 特征{best_feature}")
    
    print("最佳分割特征测试通过！\n")

def test_mushroom_dataset():
    """测试蘑菇数据集上的完整流程"""
    print("测试蘑菇数据集上的完整流程...")
    
    # 测试根节点熵
    root_entropy = compute_entropy(y_train)
    expected_entropy = 1.0  # 5个可食用，5个有毒
    assert abs(root_entropy - expected_entropy) < 1e-10, f"期望根节点熵为{expected_entropy}, 但得到{root_entropy}"
    print("✓ 根节点熵测试通过")
    
    # 测试最佳分割特征
    root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    best_feature = get_best_split(X_train, y_train, root_indices)
    expected_best = 2  # 独生特征应该提供最大信息增益
    assert best_feature == expected_best, f"期望最佳特征为{expected_best}, 但得到{best_feature}"
    print("✓ 最佳分割特征测试通过")
    
    print("蘑菇数据集测试通过！\n")

def run_all_tests():
    """运行所有测试"""
    print("开始运行决策树测试...")
    print("=" * 50)
    
    try:
        test_compute_entropy()
        test_split_dataset()
        test_compute_information_gain()
        test_get_best_split()
        test_mushroom_dataset()
        
        print("🎉 所有测试都通过了！")
        print("决策树实现完全正确！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
