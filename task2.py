import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(op = [], ttl = None):
    plt.figure(figsize=(8, 4))
    plt.hist(op, edgecolor='black')
    plt.title(ttl)
    plt.xlabel('Opinion')


def test_func(num_people = 50, threshold = 0.2, beta = 0.2):
    print(f'total number of people in this test function is {num_people}')
    print(f'threshold set at {threshold}')
    print(f'beta set at {beta}')

    # 初始化意见
    opinions = np.random.rand(num_people)

    plot_hist(opinions,'Initial Opinions')
    plt.show()

    # 模拟意见更新
    for each in range(3):
        # 随机选择一个人
        person_idx = np.random.randint(0, num_people)
        print(f'------iteration {each}------')
        print(f'people No. {person_idx} chosen as Xi')
        
        # 随机选择左或右邻居
        direction = np.random.choice([-1, 1])
        print(f'direction {direction} chosen')
        neighbor_idx = (person_idx + direction) % num_people
        print(f'people No. {neighbor_idx} chosen as Xj')
        
        # 检查意见差异
        diff = abs(opinions[person_idx] - opinions[neighbor_idx])
        
        if diff < threshold:
            # 更新意见
            print('diff < threshold, updating opinions')
            print(f'Xi({each}) = {opinions[person_idx]}, Xj({each}) = {opinions[neighbor_idx]}')
            opinions[person_idx] += beta * (opinions[neighbor_idx] - opinions[person_idx])
            opinions[neighbor_idx] += beta * (opinions[person_idx] - opinions[neighbor_idx])
            print(f'Xi({each+1}) = {opinions[person_idx]}, Xj({each+1}) = {opinions[neighbor_idx]}')
        else:
            print('diff >= threshold, no changes made')

    # 可视化更新后的意见
    plot_hist(opinions,'Final Opinions')
    plt.show()

def main(num_people = 50, threshold = 0.2, beta = 0.2):

    # 初始化意见
    opinions = np.random.rand(num_people)

    plot_hist(opinions,'Initial Opinions')
    plt.show()

    # 存储每次迭代后的意见数据
    opinions_history = [opinions.copy()]

    # 模拟意见更新
    for each in range(num_people*100):
        # 随机选择一个人
        person_idx = np.random.randint(0, num_people)
        
        # 随机选择左或右邻居
        direction = np.random.choice([-1, 1])
        neighbor_idx = (person_idx + direction) % num_people
        
        # 检查意见差异
        diff = abs(opinions[person_idx] - opinions[neighbor_idx])
        
        if diff < threshold:
            # 更新意见
            opinions[person_idx] += beta * (opinions[neighbor_idx] - opinions[person_idx])
            opinions[neighbor_idx] += beta * (opinions[person_idx] - opinions[neighbor_idx])
        # 存储当前意见数据
        opinions_history.append(opinions.copy())

    # 可视化迭代过程中的意见变化
    plt.figure(figsize=(10, 6))
    for i, opinion in enumerate(opinions_history):
        plt.plot([i]*num_people, opinion, 'o', markersize=2, color='blue', alpha=0.5)

    plt.title('Opinions Evolution During Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Opinion')
    plt.show()

    # 可视化更新后的意见
    plot_hist(opinions,'Final Opinions')
    plt.show()

if __name__ == "__main__":
    num_people = 50
    threshold = 0.2
    beta = 0.2
    if '-threshold' in sys.argv:
        threshold_idx = sys.argv.index('-threshold')
        threshold = float(sys.argv[threshold_idx + 1])
    if '-beta' in sys.argv:
        beta_idx = sys.argv.index('-beta')
        beta = float(sys.argv[beta_idx + 1])
    if '-num_people' in sys.argv:
        num_people_idx = sys.argv.index('-num_people')
        num_people = int(sys.argv[num_people_idx + 1])
    if '-test_defuant' in sys.argv:
        test_func(num_people, threshold, beta)
    else:
        if '-defuant' in sys.argv:
            main(num_people, threshold, beta)