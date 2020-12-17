import numpy as np
import networkx as nx
from network_objects import G_7_line_out,G_7_branches,G_7_three_variable
from numerical_methods import forward_euler
from markov_model import states_to_indices
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from markov_model import  create_SIS_matrix, create_SIS_matrix_variable, enumerate_states, sort_order, create_SIS_matrix_w_distancing
from scipy.fftpack import fft,rfft, dct
import pywt


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

plot_style = ['seaborn-paper', './custom_style.mplstyle']

custom_3_colors = matplotlib.colors.ListedColormap(['gray', 'red', 'blue'])


def plot_full_states(a0, a1, G, x_t, states, h, xlabel = 'time (in days)', color_limits = [1E-4,1], frequency_domain = False, sort_ids = 1):
	cmap = plt.get_cmap('tab10_r')
	# cmap = custom_3_colors
	plt.style.use(plot_style)


	states_sort_order = sort_order(states, sort_ids)
	y_plot = np.arange(states.shape[0]+1)

	if frequency_domain:
		t_plot = np.fft.fftfreq(x_t.shape[1]+1, h)
		t_sort = np.argsort(t_plot)
		t_plot = t_plot[t_sort]
		t = np.fft.fftfreq(x_t.shape[1], h)
		x_t = x_t[:,np.argsort(t)]
	else:
		t_plot = np.arange(0,h*(x_t.shape[1]+1), h)

	
	states_plot = a0.pcolor(np.arange(len(G.nodes)+1),
							y_plot,
							states[states_sort_order],
							cmap = cmap)

	time_plot = a1.pcolor(	t_plot,
							y_plot,
							x_t[states_sort_order], 
							cmap = 'binary',
							# shading = 'flat',
							# norm = colors.Normalize(vmin=color_limits[0], vmax=color_limits[1]))
							norm=colors.LogNorm(vmin=color_limits[0], vmax=color_limits[1]))
	
	plt.colorbar(time_plot)
	a0.set_xticks([])
	a0.set_yticks([])
	a0.spines['top'].set_visible(False)
	a0.spines['right'].set_visible(False)
	# a0.spines['bottom'].set_visible(False)
	a0.spines['left'].set_visible(False)
	
	a1.set_yticks([])
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	# a1.spines['bottom'].set_visible(False)
	a1.spines['left'].set_visible(False)

	a1.autoscale(enable=True, axis='both', tight=True)
	a0.autoscale(enable=True, axis='both', tight=True)

	a0.set_xlabel('states')
	# a0.set_xlabel('nodes')
	# a0.set_title('states')
	a1.set_xlabel(xlabel)
	a1.set_title('probabilities of states')



def plot_singular_values(ax, s):
	plt.style.use(plot_style)

	ax.semilogy(s, color='gray', marker='o', linestyle = 'solid')

	ax.set_ylabel('singular value')
	ax.set_xlabel('singular value index')



def plot_multiple_left_singular_vectors(a0, a1, x, states, G, scales = None, n_vectors = 5, sort_ids = 1):
	cmap = plt.get_cmap('tab10_r')
	plt.style.use(plot_style)


	if scales is None:
		scales = np.ones((n_vectors))
	else:
		scales = scales[:n_vectors]


	states_sort_order = sort_order(states, sort_ids)
	y_plot = np.arange(states.shape[0]+1)

	
	states_plot = a0.pcolor(np.arange(len(G.nodes)+1),
							y_plot,
							states[states_sort_order],
							cmap = cmap)


	for vector_i in range(n_vectors):
		x_plot = x[:,vector_i].reshape(-1,1)*scales[vector_i] 
		a1.plot( x_plot[states_sort_order].reshape(-1), y_plot[:-1]+0.5, marker='.', linestyle = 'solid',
			 	 label = 'Vector {}'.format(vector_i+1))
	
	a0.set_xticks([])
	a0.set_yticks([])
	a0.spines['top'].set_visible(False)
	a0.spines['right'].set_visible(False)
	a0.spines['left'].set_visible(False)
	
	a0.set_xlabel('states')
	# a0.set_xlabel('nodes')
	# a0.set_title('states')
	a1.set_xlabel('vector value')
	a1.set_yticks([])
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	a1.spines['left'].set_visible(False)


	a1.legend()

	a0.set_ylim([y_plot[0],y_plot[-1]])
	a1.set_ylim([y_plot[0],y_plot[-1]])





def plot_right_matrix_probs(ax, v, h,xlim = [-0.1,100],xlabel = 'Haar Wavelet Number', ylabel = 'Probability of Measurement', 
							frequency_domain = False, log2axis = True, ylog = True):
	plt.style.use(plot_style)

	v = np.sum(v*v.conj(),axis=0)/np.sum(v*v.conj(),axis=(0,1))
	# print(v)
	if frequency_domain:
		t = np.fft.fftfreq(len(v), h)
		t_sort = np.argsort(t)
		t = t[t_sort]
	else:
		t = np.arange(len(v))*h
		t_sort = np.argsort(t)
		t = t[t_sort]

	ax.plot(t,v[t_sort], color='gray', marker='o', linestyle = 'solid')

	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	if ylog:
		ax.set_yscale('log')

	if log2axis:
		ax.set_xscale('symlog', basex=2)
	ax.set_xlim(xlim)


def plot_multiple_right_singular_vectors(ax, v, h, n_vectors, scales = None, 
										xlabel = 'time (in days)', frequency_domain = False,
										xlim = None, log2axis = False, 
										ylabel = 'singular vector value', ylog = False):
	plt.style.use(plot_style)


	if scales is None:
		scales = np.ones((n_vectors))
	else:
		scales = scales[:n_vectors]

	if frequency_domain:
		t = np.fft.fftfreq(v.shape[1], h)
		t_sort = np.argsort(t)
		t = t[t_sort]
	else:
		t = np.arange(len(v))*h
		t_sort = np.argsort(t)
		t = t[t_sort]

	for vector_i in range(n_vectors):
		v_plot = v[vector_i,:]*scales[vector_i]
		ax.plot(t,v_plot[t_sort], marker='.', linestyle = 'solid',
			 	 label = 'Vector {}'.format(vector_i+1))

	if xlim is not None:
		ax.set_xlim(xlim)

	if ylog:
		ax.set_yscale('log')

	ax.set_ylabel('singular vector value')
	ax.set_xlabel(xlabel)
	ax.legend()

	if log2axis:
		ax.set_xscale('symlog', basex=2)
	



def plot_haar_vectors(haar_ax,haar_mat,  n_rows = 4, n_cols = 4,
						h=1., xlabel = 'Time (days)'):
	plt.style.use(plot_style)

	t = np.arange(haar_mat.shape[0])*h
	ylim = [1.1*np.min(haar_mat[6,:]),1.1*np.max(haar_mat[6,:])]

	number_text = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']

	for i in range(n_rows):
		for j in range(n_cols):
			curr_ax = haar_ax[i][j]
			zeroth = False
			if j == 0:
				n_vec = 0
				if i == 0:
					zeroth = True
			else:
				n_vec = 2**(j-1)+i

			if (n_vec < 2**(j) and j > 0) or zeroth:
				v_plot = haar_mat[n_vec,:]
				curr_ax.plot(t,v_plot, marker='.', linestyle = 'solid', color = plt.cm.tab20b(j*4+i),
					 	 label = 'Vector {}'.format(n_vec))
				curr_ax.spines['top'].set_position('zero')
				curr_ax.spines['right'].set_visible(False)
				curr_ax.spines['bottom'].set_visible(False)
				curr_ax.spines['left'].set_position('zero')
				curr_ax.set_ylim(ylim)
				curr_ax.text(.5,1., number_text[n_vec], horizontalalignment='center', transform=curr_ax.transAxes,
						verticalalignment='top', fontsize=7, color = 'gray')
			else:
				curr_ax.spines['top'].set_visible(False)
				curr_ax.spines['right'].set_visible(False)
				curr_ax.spines['bottom'].set_visible(False)
				curr_ax.spines['left'].set_visible(False)
				
			curr_ax.axes.xaxis.set_ticklabels([])
			curr_ax.axes.yaxis.set_ticklabels([])
			curr_ax.axes.xaxis.set_ticks([])
			curr_ax.axes.yaxis.set_ticks([])

	curr_ax = haar_ax[0][-1]
	curr_ax.text(0,-0.1, '...', horizontalalignment='left', transform=curr_ax.transAxes,
			verticalalignment='top', fontsize=10, color = 'black', fontweight = 'bold')



def plot_haar_vectors_full(haar_ax,haar_mat,  n_rows = 4, n_cols = 4,
						h=1., xlabel = 'Time (days)'):
	plt.style.use(plot_style)

	t = np.arange(haar_mat.shape[0])*h
	ylim = [1.5*np.min(haar_mat[6,:]),1.5*np.max(haar_mat[6,:])]

	number_text = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', 
					'9th', '10th', '11th', '12th', '13th', '14th', '15th']

	for i in range(n_rows):
		for j in range(n_cols):
			curr_ax = haar_ax[i][j]
			zeroth = False
			n_vec = n_cols*j+i

			v_plot = haar_mat[n_vec,:]
			curr_ax.plot(t,v_plot, marker='.', linestyle = 'solid', color = 'gray',
				 	 label = 'Vector {}'.format(n_vec))
			curr_ax.spines['top'].set_position('zero')
			curr_ax.spines['right'].set_visible(False)
			curr_ax.spines['bottom'].set_visible(False)
			curr_ax.spines['left'].set_position('zero')
			curr_ax.set_ylim(ylim)
			curr_ax.text(.5,1., number_text[n_vec], horizontalalignment='center', transform=curr_ax.transAxes,
					verticalalignment='top', fontsize=7, color = 'gray')

				
			curr_ax.axes.xaxis.set_ticklabels([])
			curr_ax.axes.yaxis.set_ticklabels([])
			curr_ax.axes.xaxis.set_ticks([])
			curr_ax.axes.yaxis.set_ticks([])

	# curr_ax = haar_ax[0][-1]
	# curr_ax.text(0,-0.1, '...', horizontalalignment='left', transform=curr_ax.transAxes,
	# 		verticalalignment='top', fontsize=10, color = 'black', fontweight = 'bold')



def plot_network(ax, network_object):
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
	    "font_size": 7,
	    "node_size": 150
	}
	nx.draw(network_object, ax = ax, **options)


def plot_expected_numbers(a0,a1,states,X):
	plt.style.use(plot_style)
	r_vars = get_expected_numbers(states,X)
	n_states = np.max(np.max(states))+1


	for i in range(n_states):	
		a0.plot(r_vars[i][0], label = 'state '+str(i))
		a1.plot(r_vars[i][1], label = 'state '+str(i))

	a0.set_ylabel('Expected Number of Nodes in State')
	a0.set_xlabel('Time Step')

	a1.set_ylabel('Variance of Number of Nodes in State')
	a1.set_xlabel('Time Step')

	a0.legend()
	a1.legend()




def get_expected_numbers(states,X):
	n_states = np.max(np.max(states))+1
	results = []

	for state_i in range(n_states):
		N_i = np.sum(states == state_i,axis = 1).reshape(-1,1)
		E_X = np.sum( N_i*X,axis = 0)
		Var_X = np.sum( (N_i*N_i)*X, axis = 0) - E_X*E_X

		results.append([E_X,Var_X])

	return results



if __name__ == '__main__':
	states = np.eye(5).astype(int)
	states[-1,0] = 2
	probs = np.zeros((5,2))
	probs[0,0] = 1
	probs[0,1] = 0.3
	probs[2,1] = 0.2
	probs[4,1] = 0.5


	plot_expected_numbers(states,probs)