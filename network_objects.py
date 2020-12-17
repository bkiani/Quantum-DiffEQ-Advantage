import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def save_network(network_object, save_name):
	fig = plt.figure()
	edge_vmin = 0
	edge_vmax = 3
	labels = nx.get_edge_attributes(network_object,'weight')
	labels = np.array([labels[i] for i in labels])
	colors = labels/np.max(labels)
	widths = colors*(edge_vmax-edge_vmin)
	# labels = range(7)
	# print(labels)
	options = {
	    # "edge_color": colors,
	    "width": widths,
	    "edge_cmap": plt.cm.RdPu,
	    "with_labels": True,
	}
	nx.draw(network_object, **options)

	fig.set_size_inches(2.5, 2.5)
	plt.savefig(save_name)


def circular_line_graph(N):
	G = nx.Graph()
	G.add_nodes_from( range(N) )
	for i in range(N):
		G.add_edges_from([(i, (i+1) % N)], weight = 1.)

	if N > 3:
		rand_edge_start = np.random.randint(N)
		rand_edge_end = (rand_edge_start + np.random.randint(N-3)) % N
		print([(rand_edge_start, rand_edge_end)])
		G.add_edges_from([(rand_edge_start, rand_edge_end)], weight = 1.)

	return G


def erdos_renyi(n,p):
	return nx.erdos_renyi_graph(n,p)


## 7 node line to 3
G = nx.Graph()
G.add_nodes_from( range(7) )
G.add_edges_from( [ (0,1),
					(1,2),
					(2,3),
					(3,4),
					(3,5),
					(3,6),
					(4,5),
					(5,6)
					], weight = 1. )
G_7_line_out = G


## 7 node - 3 branches
G = nx.Graph()
G.add_nodes_from( range(7) )
G.add_edges_from( [ (0,1),
					(0,2),
					(1,3),
					(2,3),
					(4,5),
					(4,6)
					], weight = 1.5 )
G.add_edges_from( [ (0,4)
					], weight = 0.3 )
G_7_branches = G

## 7 node - 3 branches
G = nx.Graph()
G.add_nodes_from( range(7) )
G.add_edges_from( [ (0,1),
					(0,2),
					(1,3),
					(2,3),
					(4,5),
					(4,6)
					], weight = 1.5 )
G.add_edges_from( [ (0,4)
					], weight = 1.5 )
G_7_branches_equal = G


## 7 node - 2 branches
G = nx.Graph()
G.add_nodes_from( range(7) )
G.add_edges_from( [ (0,1),
					(1,2),
					(2,3)
					], weight = 1.6 )
G.add_edges_from( [ (2,4),
					(4,6)
					], weight = 0.8 )
G.add_edges_from( [ (1,5),
					(5,6)
					], weight = 0.4 )
G_7_three_variable = G

G = nx.Graph()
G.add_nodes_from( range(7) )
G.add_edges_from( [ (0,1),
					(1,2),
					(2,3)
					], weight = 1.0 )
G.add_edges_from( [ (2,4),
					(4,6)
					], weight = 1.0 )
G.add_edges_from( [ (1,5),
					(5,6)
					], weight = 1.0 )
G_7_three_variable_equal = G


if __name__ == '__main__':

	save_network(circular_line_graph(8),'circular_chord.pdf')
	# save_network(G_7_branches, 'branched_network_7_nodes.pdf')
	# save_network(G_7_line_out, 'line_out_network_7_nodes.pdf')
	# save_network(G_7_three_variable, 'three_variable_network_7_nodes.pdf')
	# # plt.show()