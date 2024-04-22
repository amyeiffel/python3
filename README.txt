# Network, Ising Model, and Defuant Model Simulation

这个项目包含几个模块：
- Network：用于创建和分析网络模型。
- Ising Model：用于模拟Ising模型。
- Defuant Model：用于模拟Defuant模型。

## 安装

这个项目需要Python 3和以下的库：
- numpy
- matplotlib

你可以使用pip安装所需的库：

pip install numpy matplotlib

Network模块
你可以创建不同类型的网络，如随机网络、环形网络和小世界网络。

network = Network()
network.make_random_network(10)
network.plot()

Ising Model模块
你可以模拟Ising模型的演变。
population = np.random.choice([-1, 1], size=(10, 10))
ising_main(population)

Defuant Model模块
你可以模拟Defuant模型的演变。
defuant_main()

测试
你可以运行以下命令来测试代码：
python assignment.py -test_defuant -threshold 0.2 -beta 0.2 -num_people 50

运行defuant模型
python assignment.py -defuant -threshold 0.2 -beta 0.2 -num_people 50 #将生成一个1D grid上的defuant模型，使用你给定的相关参数。
