import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import argparse
import sys

class Queue: #for breadth-first-search

	def __init__(self):
		self.queue = []

	def push(self,item):
		self.queue.append(item)

	def pop(self):
		return self.queue.pop(0)

	def is_empty(self):
		return len(self.queue) == 0

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

	def update_value(self, neighbor_value, threshold, beta):
		#####task 2#####
		if abs(self.value - neighbor_value) <= threshold:
			self.value += beta * (neighbor_value - self.value)

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		total_degree = 0

		for node in self.nodes:
			for i in node.connections:
				if i == 1:
					total_degree += 1
				else:
					continue
		total_nodes = len(self.nodes)
		mean_degree = total_degree / total_nodes

		return mean_degree

	def get_clustering(self):
		total_node_clustering = 0

		for node_index in range(len(self.nodes)): #for each node

			node = self.nodes[node_index]
			actual_connections = 0

			neighbours = [self.nodes[neighbour] for neighbour, connection in enumerate(node.connections) if connection]

			possible_connections = len(neighbours) * (len(neighbours) - 1) / 2
			#calculate possible connections

			for i, j in enumerate(neighbours): #and check each of its neighbours to see the actual connections
				for next_to in neighbours[i + 1:]:
					if j.connections[next_to.index]:
						actual_connections += 1

			if possible_connections != 0: #calculate clustering for current node and add value to totals
				node_clustering = actual_connections / possible_connections
				total_node_clustering += node_clustering

			mean_clustering = total_node_clustering / len(self.nodes)

		return mean_clustering
	
	def get_path_length(self):
		total_path_length = 0
		total_paths = 0

		for node in self.nodes:
			path = search_paths(self, node) #function defined as this algorithem is used many times for each node
			for path_length in path.values():

				total_path_length += path_length
				total_paths += 1 #the problem of counting path to current node itself is solved in search_paths()

		mean_path_length = total_path_length / total_paths

		return round(mean_path_length, 15) #due to test requirements of 15dp
	
	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here
		pass
	
	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here
		pass

	
	def plot(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

	def update_nodes(self, nds):
		#####task 2#####
		if len(self.nodes) == len(nds):
			self.nodes = nds
		else:
			print('Node length not equal')

def search_paths(network, start_node): #apply breadth-first-search

	#to search paths to all other nodes, return the distance to each other nodes

	paths = {}
	queue = Queue()
	queue.push((start_node,0))
	visited = []
	visited.append(start_node.index) #add the node being check into the visited list so it will not count itself and
									#the path to itself

	while not queue.is_empty():
		start_node, distance = queue.pop()

		for neighbour_index, connection in enumerate(start_node.connections):
			if connection and neighbour_index not in visited:
				neighbour = network.nodes[neighbour_index]
				paths[neighbour.index] = distance + 1
				queue.push((neighbour, distance + 1))
				visited.append(neighbour_index)

	return paths

def test_networks(): #given code

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")

def parse(arg):

	######task 3######
	network = 0
	test_network = 0

	if '-network' == arg[0]:
		network = 1

	if '-test_network' == arg[0]:
		test_network = 1

	return network, test_network

def flags():
	flag = argparse.ArgumentParser(description="type your flags")

	######task 3######
	flag.add_argument('-network', action='store_true')
	flag.add_argument('-test_network', action='store_true')


def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	#Your code for task 1 goes here

	return np.random.random() * population

def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1

	#Your code for task 1 goes here

def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''
	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''

	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==14), "Test 9"
	assert(calculate_agreement(population,1,1, -10)==-6), "Test 10"

	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def plot_hist(op = [], ttl = None):
	plt.figure(figsize=(8, 4))
	plt.hist(np.transpose(op), edgecolor='black')
	plt.title(ttl)
	plt.xlabel('Opinion')

def get_opinions(nw):
	opinions = [0 for _ in range(len(nw.nodes))]
	for node in nw.nodes:
		opinions[node.index] = node.value
	return opinions

def get_mean(opinions_history):
	mean_op = [0 for _ in range(len(opinions_history[0]))]
	for opinions in opinions_history:
		for i in range(len(opinions)):
			mean_op[i] += opinions[i]
	for j in range(len(mean_op)):
		mean_op[j] /= len(opinions_history)
	return mean_op

def defuant_main(num_people = 50, threshold = 0.2, beta = 0.2, nw = None):
	#Your code for task 2 goes here
	if nw is None:	
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
	else:
		#使用Network实现的Defuant model
		node_history = []
		for node in nw.nodes:
			node_history.append(node)
			neighbours = [nw.nodes[neighbour] for neighbour, connection in enumerate(node.connections) if connection]
			for neighbour in neighbours:
				node.update_value(neighbour.value, threshold, beta)
				neighbour.update_value(node.value, threshold, beta)
				#print('updating values')

		nw.update_nodes(node_history)
		return nw
	
def test_defuant(num_people = 50, threshold = 0.2, beta = 0.2):
	#Your code for task 2 goes here
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


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	#You should write some code for handling flags here
	if '-test_defuant' or '-defuant' in sys.argv:
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
			test_defuant(num_people, threshold, beta)
		else:
			if '-defuant' in sys.argv:
				if '-use_network' in sys.argv:
					network_size_idx = sys.argv.index('-use_network')
					network_size = int(sys.argv[network_size_idx+1])
					connectivity_p = 0.5
					nw = Network()
					Network.make_random_network(nw, network_size, connectivity_p)
					
					print("Initial Network:")
					nw.plot()
					opinions = get_opinions(nw)
					opinions_history = [opinions.copy()]
					plot_hist(opinions, 'Initial Opinions:')
					plt.show()

					for i in range(5):
						print(f"Iteration {i+1}:")
						nw = defuant_main(network_size, threshold, beta, nw)
						opinions = get_opinions(nw)
						opinions_history.append(opinions)

					def update(frame):
						opinions = opinions_history[frame]
						plt.hist(np.transpose(opinions), edgecolor='black')
						return opinions

					fig, ax = plt.subplots()
					# 创建动画，帧数为迭代次数，每帧间隔100毫秒
					ani = animation.FuncAnimation(fig, update, frames=range(len(opinions_history)), interval=100, repeat=False)
					plt.show()
					
					plot_hist(get_mean(opinions_history), 'Mean Opinions:')
					plt.show()
					
				else:
					defuant_main(num_people, threshold, beta)
			elif '-test_networks' in sys.argv:
				test_networks()
			else:
				print('No model specified')
		
	

if __name__=="__main__":
	main()
