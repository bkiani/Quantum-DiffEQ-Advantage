import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

default_n_classes = 2


# def states_to_indices(states, n_classes = default_n_classes):
# 	if len(states.shape) == 1:
# 		n_nodes = len(states)
# 		states = states.reshape(1,-1)
# 	else:
# 		n_nodes = states.shape[1]

# 	return np.sum( states*(n_classes**np.arange(n_nodes-1,-1,-1)).reshape(1,-1),
# 					axis = 1 )

def states_to_indices(states, n_classes = default_n_classes):
	if len(states.shape) == 1:
		n_nodes = len(states)
		states = states.reshape(1,-1)
	else:
		n_nodes = states.shape[1]

	return np.sum( states*(n_classes**np.arange(n_nodes)).reshape(1,-1),
					axis = 1 )

def indices_to_states(indices, n_nodes, n_classes = default_n_classes):
	remainders = n_classes**np.arange(1,n_nodes+1,1)
	dividers = n_classes**np.arange(n_nodes)
	# floor_indices = np.floor_divide(indices.reshape(-1,1), dividers)
	floor_indices = indices.reshape(-1,1)%remainders
	return np.floor_divide(floor_indices, dividers)

def enumerate_states(n_nodes, n_classes = default_n_classes):
	indices = np.arange(n_classes**n_nodes)
	return indices_to_states(indices, n_nodes, n_classes)

def sort_order(states, id_sort = 1):
	if isinstance(id_sort, int):
		id_sort = [id_sort]
	for id_i in id_sort:
		n_id = np.sum(states.astype(np.int) == id_i, axis = 1).astype(np.int)
	return np.argsort(n_id, kind = 'mergesort')

def create_SIS_matrix(G, r_SI, r_IS):
	# initialize
	n_classes = 2
	states = enumerate_states(len(G.nodes),n_classes)
	Q = np.zeros([states.shape[0],states.shape[0]])

	# add S to I transitions
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1).reshape(-1)
		for infected_k in infected:
			neighbors = G.neighbors(infected_k)
			for neighbor_j in neighbors:
				if state_i[neighbor_j] == 0:
					transition_state = np.copy(state_i)
					transition_state[neighbor_j] = 1
					transition_ind = states_to_indices(transition_state, n_classes)
					Q[transition_ind,i] += r_SI

	# add I to S transitions
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1)
		for infected_k in infected:
			transition_state = np.copy(state_i)
			transition_state[infected_k] = 0
			transition_ind = states_to_indices(transition_state, n_classes)
			Q[transition_ind,i] += r_IS

	# get diagonals	
	column_sums = np.sum(Q, axis = 0)
	np.fill_diagonal(Q, -column_sums)

	return Q

def create_SIS_matrix_w_distancing(G, r_SI, r_IS, reduction_percent = 0.2, n_infect_threshold = 3):
	# initialize
	n_classes = 2
	states = enumerate_states(len(G.nodes),n_classes)
	Q = np.zeros([states.shape[0],states.shape[0]])

	# add S to I transitions
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1).reshape(-1)
		if len(infected)>=n_infect_threshold:
			r_SI_state = r_SI*reduction_percent
		else:
			r_SI_state = r_SI
		for infected_k in infected:
			neighbors = G.neighbors(infected_k)
			for neighbor_j in neighbors:
				if state_i[neighbor_j] == 0:
					transition_state = np.copy(state_i)
					transition_state[neighbor_j] = 1
					transition_ind = states_to_indices(transition_state, n_classes)
					Q[transition_ind,i] += r_SI_state

	# add I to S transitions
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1)
		for infected_k in infected:
			transition_state = np.copy(state_i)
			transition_state[infected_k] = 0
			transition_ind = states_to_indices(transition_state, n_classes)
			Q[transition_ind,i] += r_IS

	# get diagonals	
	column_sums = np.sum(Q, axis = 0)
	np.fill_diagonal(Q, -column_sums)

	return Q


def create_SIS_matrix_variable(G, r_IS):
	# initialize
	n_classes = 2
	states = enumerate_states(len(G.nodes),n_classes)
	Q = np.zeros([states.shape[0],states.shape[0]])

	# add S to I transitions
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1).reshape(-1)
		for infected_k in infected:
			neighbors = G.neighbors(infected_k)
			for neighbor_j in neighbors:
				if state_i[neighbor_j] == 0:
					r_SI = G[infected_k][neighbor_j]['weight']
					transition_state = np.copy(state_i)
					transition_state[neighbor_j] = 1
					transition_ind = states_to_indices(transition_state, n_classes)
					Q[transition_ind,i] += r_SI

	# add I to S transitions
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1)
		for infected_k in infected:
			transition_state = np.copy(state_i)
			transition_state[infected_k] = 0
			transition_ind = states_to_indices(transition_state, n_classes)
			Q[transition_ind,i] += r_IS

	# get diagonals	
	column_sums = np.sum(Q, axis = 0)
	np.fill_diagonal(Q, -column_sums)

	return Q


def create_3_state_social_matrix_variable(G, r_IS):
	# initialize
	n_classes = 3
	states = enumerate_states(len(G.nodes),n_classes)
	Q = np.zeros([states.shape[0],states.shape[0]])

	# add 1 to 1 and 2 to 2 transitions (e.g. liberal to liberal)
	for i, state_i in enumerate(states):
		# from 0 to 1
		infected = np.argwhere(state_i == 1).reshape(-1)
		for infected_k in infected:
			neighbors = G.neighbors(infected_k)
			for neighbor_j in neighbors:
				if state_i[neighbor_j] == 0:
					r_SI = G[infected_k][neighbor_j]['weight']
					transition_state = np.copy(state_i)
					transition_state[neighbor_j] = 1
					transition_ind = states_to_indices(transition_state, n_classes)
					Q[transition_ind,i] += r_SI

		# repeat for 0 to 2
		infected = np.argwhere(state_i == 2).reshape(-1)
		for infected_k in infected:
			neighbors = G.neighbors(infected_k)
			for neighbor_j in neighbors:
				if state_i[neighbor_j] == 0:
					r_SI = G[infected_k][neighbor_j]['weight']
					transition_state = np.copy(state_i)
					transition_state[neighbor_j] = 2
					transition_ind = states_to_indices(transition_state, n_classes)
					Q[transition_ind,i] += r_SI

	# add 2 to 0 and 1 to 0 transitions (to undecided)
	for i, state_i in enumerate(states):
		infected = np.argwhere(state_i == 1)
		for infected_k in infected:
			transition_state = np.copy(state_i)
			transition_state[infected_k] = 0
			transition_ind = states_to_indices(transition_state, n_classes)
			Q[transition_ind,i] += r_IS

		infected = np.argwhere(state_i == 2)
		for infected_k in infected:
			transition_state = np.copy(state_i)
			transition_state[infected_k] = 0
			transition_ind = states_to_indices(transition_state, n_classes)
			Q[transition_ind,i] += r_IS

	# get diagonals	
	column_sums = np.sum(Q, axis = 0)
	np.fill_diagonal(Q, -column_sums)

	return Q




if __name__ == '__main__':
	from network_objects import G_7_branches
	G = G_7_branches
	
	sample_states = np.asarray([[0,1,2],[2,0,0],[0,0,2],[0,1,0]])
	sample_indices = states_to_indices(np.asarray([[0,1,2],[2,0,0],[0,0,2],[0,1,0]]), n_classes = 3) 
	print(sample_indices)
	sample_states = indices_to_states(sample_indices, 3, n_classes = 3)
	print(sample_states)

	print(enumerate_states(len(G.nodes),3).shape)