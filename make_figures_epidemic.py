import numpy as np
import networkx
from network_objects import G_7_line_out,G_7_branches,G_7_three_variable, G_7_three_variable_equal
from numerical_methods import forward_euler
from markov_model import states_to_indices
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
from markov_model import  create_SIS_matrix, create_SIS_matrix_variable, enumerate_states, sort_order, create_SIS_matrix_w_distancing
from scipy.fftpack import fft,rfft, dct
import pywt
import plot_utils


figures_subfolder = './figures/'

#using PNAS guide: https://blog.pnas.org/digitalart.pdf
fig_width_med = 4.5
fig_width_onecolum = 3.3
fig_width_max = 7.
fig_height_max = 6.0
fig_height_med = 3.42
fig_height_small = 2.5

plot_style = ['seaborn-paper', './custom_style.mplstyle']


def haarMatrix(n, normalized=True):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h


def plot_network_and_states(fig_name, network_object, x_t, states, h, xlabel = 'time (in days)', color_limits = [1E-3,1.], frequency_domain = False):

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	2, 2,
							height_ratios = [1,3], width_ratios = [1,6],
							wspace = 0.0, hspace = 0.01)

	a0 = fig.add_subplot(gs[0,1])
	# a0_text = fig.add_subplot(gs[0,0])
	a1 = fig.add_subplot(gs[1,0])
	a2 = fig.add_subplot(gs[1,1])	

	plot_utils.plot_network(a0, network_object)
	plot_utils.plot_full_states(a1, a2, network_object, x_t, states, h, xlabel = xlabel, 
								color_limits = color_limits, frequency_domain = frequency_domain)

	a0.text(0.0, 1.0, 'A', horizontalalignment='left', transform=a0.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a2.text(0.0, 1.0, 'B', horizontalalignment='left', transform=a2.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	
	fig.set_size_inches(fig_width_onecolum, 6.0)
	plt.savefig(figures_subfolder + fig_name, bbox_inches = 'tight')	
	


def plot_singular_values_and_vectors(fig_name, s, network_object, x_left, states_left, left_args, v_right, h_right, right_args):

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	2, 4,
							height_ratios = [1,1], width_ratios = [7,0.4,1,6],
							wspace = 0.0, hspace = 0.01)

	a0 = fig.add_subplot(gs[0,0])			# plotting singular values
	a1 = fig.add_subplot(gs[:,2])			# plotting states - left singular vectors
	a2 = fig.add_subplot(gs[:,3])			# plotting values - left singular vectors
	a3 = fig.add_subplot(gs[1,0])			# plotting right singular vectors

	plot_utils.plot_singular_values(a0, s)
	plot_utils.plot_multiple_left_singular_vectors(a1, a2, x_left, states_left, network_object, **left_args)
	plot_utils.plot_multiple_right_singular_vectors(a3, v_right, h_right, **right_args)

	a0.text(0.0, 1.0, 'A', horizontalalignment='left', transform=a0.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a2.text(0.0, 1.0, 'B', horizontalalignment='left', transform=a2.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a3.text(0.0, 1.0, 'C', horizontalalignment='left', transform=a3.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	
	fig.set_size_inches(fig_width_med, 4.)
	plt.savefig(figures_subfolder + fig_name, bbox_inches = 'tight')	



def plot_appendix_simulations(fig_name, x_t, h, s, network_object, states, x_left, left_args, v_right, h_right, right_args,
	 							xlabel_main = 'time (in days)', color_limits = [1E-3,1]):

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	2, 7,
							height_ratios = [1,1], width_ratios = [1,6,0.4,7,0.4,1,6],
							wspace = 0.0, hspace = 0.01)

	a0_0 = fig.add_subplot(gs[:,0])			# plotting states - full progression
	a0_1 = fig.add_subplot(gs[:,1])			# plotting values - full progression
	a1_0 = fig.add_subplot(gs[0,3])			# plotting singular values
	a2_0 = fig.add_subplot(gs[:,5])			# plotting states - left singular vectors
	a2_1 = fig.add_subplot(gs[:,6])			# plotting values - left singular vectors
	a3_0 = fig.add_subplot(gs[1,3])			# plotting right singular vectors

	plot_utils.plot_full_states(a0_0, a0_1, network_object, x_t, states, h, xlabel = xlabel_main, 
								color_limits = color_limits, frequency_domain = False)
	plot_utils.plot_singular_values(a1_0, s)
	plot_utils.plot_multiple_left_singular_vectors(a2_0, a2_1, x_left, states, network_object, **left_args)
	plot_utils.plot_multiple_right_singular_vectors(a3, v_right, h_right, **right_args)

	a0_0.text(0.0, 1.0, 'A', horizontalalignment='left', transform=a0.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a1_0.text(0.0, 1.0, 'B', horizontalalignment='left', transform=a2.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a2_0.text(0.0, 1.0, 'C', horizontalalignment='left', transform=a3.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a3_0.text(0.0, 1.0, 'D', horizontalalignment='left', transform=a3.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	
	fig.set_size_inches(fig_width_max, 5.0)
	plt.savefig(figures_subfolder + fig_name, bbox_inches = 'tight')	



def plot_haar_plots(fig_name, network_object, v, Vh, n_vectors):
	plt.style.use(plot_style)
	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	2, 2,
							height_ratios = [1,1], width_ratios = [20,1],
							wspace = 0.0, hspace = 0.01)

	a0 = fig.add_subplot(gs[0,0])
	a1 = fig.add_subplot(gs[1,0])

	plot_utils.plot_right_matrix_probs(a0, v, 1., xlim = [-0.1,100], 
										xlabel = 'Haar wavelet number', 
										ylabel = 'probability of measurement')
	plot_utils.plot_multiple_right_singular_vectors(a1,Vh, 1., n_vectors,
											'Haar wavelet number', 
											frequency_domain = False, 
											xlim = [-0.1,100], 
											log2axis = True)

	a0.text(0.95, 1.0, 'A', horizontalalignment='left', transform=a0.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a1.text(0.95, 1.0, 'B', horizontalalignment='left', transform=a1.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	
	fig.set_size_inches(fig_width_onecolum, 4.5)
	plt.savefig(figures_subfolder + fig_name, bbox_inches = 'tight')	


def plot_haar_plots_w_vectors(fig_name, network_object, v, Vh, n_vectors):
	plt.style.use(plot_style)
	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	4, 6,
							height_ratios = [1,1,1,1], width_ratios = [5,5,5,5,1,20],
							wspace = 0.0, hspace = 0.0)

	a0 = fig.add_subplot(gs[:2,-1])	# probabilities
	a1 = fig.add_subplot(gs[2:,-1])	# singular vectors
	haar_ax = [[fig.add_subplot(gs[i,j]) for i in range(4)] for j in range(4)] # haar vectors
	haar_mat = haarMatrix(len(v))

	plot_utils.plot_right_matrix_probs(a0, v, 1., xlim = [-0.1,100], 
										xlabel = 'Haar wavelet number', 
										ylabel = 'prob. of measurement (log axis)',
										ylog = True)
	plot_utils.plot_multiple_right_singular_vectors(a1,Vh, 1., n_vectors,
											xlabel = 'Haar wavelet number',
											ylabel = 'singular vector value' ,
											frequency_domain = False, 
											xlim = [-0.1,100], 
											log2axis = True,
											ylog = False)
	# plot_utils.plot_haar_vectors(haar_ax,haar_mat,  n_rows = 4, n_cols = 4)
	plot_utils.plot_haar_vectors_full(haar_ax,haar_mat,  n_rows = 4, n_cols = 4)


	haar_ax[3][0].text(0.95, 1.0, 'A', horizontalalignment='left', transform=haar_ax[3][0].transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a0.text(0.95, 1.0, 'B', horizontalalignment='left', transform=a0.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a1.text(0.95, 1.0, 'C', horizontalalignment='left', transform=a1.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	
	# plt.subplots_adjust(wspace=0.25, hspace=0.25)
	fig.set_size_inches(fig_width_med, 4.5)
	plt.savefig(figures_subfolder + fig_name, bbox_inches = 'tight')		

def plot_fft_plots(fig_name, network_object, v, Vh, h, n_vectors):
	plt.style.use(plot_style)
	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	2, 2,
							height_ratios = [1,1], width_ratios = [20,1],
							wspace = 0.0, hspace = 0.01)

	a0 = fig.add_subplot(gs[0,0])
	a1 = fig.add_subplot(gs[1,0])

	plot_utils.plot_right_matrix_probs(a0, v, h, xlim = [-10,10], 
										xlabel = 'Frequency (1/days)', 
										ylabel = 'probability of measurement',
										frequency_domain = True,
										log2axis = False,
										ylog = True)
	plot_utils.plot_multiple_right_singular_vectors(a1,Vh, h, n_vectors,
											xlabel = 'Frequency (1/days)', 
											ylabel = 'singular vector value' ,
											frequency_domain = True, 
											xlim = [-10,10], 
											log2axis = False,
											ylog = True)

	a0.text(0.95, 1.0, 'A', horizontalalignment='left', transform=a0.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	a1.text(0.95, 1.0, 'B', horizontalalignment='left', transform=a1.transAxes,
						verticalalignment='top', fontsize=9, fontweight = 'bold')
	
	fig.set_size_inches(fig_width_onecolum, 4.)
	plt.savefig(figures_subfolder + fig_name, bbox_inches = 'tight')	



def perform_fft(data_matrix, axis_choice = 1):
	return fft(data_matrix, axis = axis_choice)/np.sqrt(data_matrix.shape[abs(axis_choice)])

def perform_dct(data_matrix, axis_choice = 1):
	return dct(data_matrix, type =2, axis = axis_choice, norm = 'ortho') #/np.sqrt(data_matrix.shape[abs(axis_choice)])

def perform_haar(data_matrix, axis_choice = 1):
	haar_mat = haarMatrix(data_matrix.shape[axis_choice])
	temp = haar_mat@data_matrix.T
	return temp.T


def get_initial_state(n=7):
	x_0 = np.zeros((states.shape[0],1))
	# x_0[1,0] = 1.
	# x_0[states_to_indices(np.eye(7)).astype(int),0] = 1./7.
	state_probs = states.astype(np.double)
	state_probs[states == 0] = 1-p_start
	state_probs[states == 1] = p_start
	x_0[:,0] = np.prod(state_probs,axis = 1)
	return x_0


if __name__ == '__main__':
	name_of_simulation = 'main_1days_1daystart'
	n_days = 1
	n_t = 2**10
	h = n_days / n_t

	G = G_7_three_variable
	Q = create_SIS_matrix_variable(G,0.33)

	states = enumerate_states(len(G.nodes), 2)
	p_start = 0.35
	x_0 = get_initial_state()

	# x_0 = np.zeros((states.shape[0],1))
	# x_0[1,0] = 1.

	x_t = forward_euler(Q, x_0, h, n_t-1)
	x_0 = x_t[:,-1].reshape(-1,1)
	# print(x_0)





	# no transform
	x_t = forward_euler(Q, x_0, h, n_t-1)
	# print(x_t.shape)
	U, s, Vh = np.linalg.svd(x_t)

	# Done
	plot_network_and_states(name_of_simulation+'_network_and_transition.pdf', G, x_t, states, h, color_limits = [1E-3,1.])

	# Done
	left_args = {'scales':np.sqrt(s), 'n_vectors':4}
	right_args = {'scales':np.sqrt(s),'n_vectors':4}
	plot_singular_values_and_vectors(name_of_simulation+'_singular_stuff.pdf',
										s, G, U, states, left_args, Vh, h, right_args)



	x_t_haar = perform_haar(x_t)
	# print(x_t_haar.shape)


	U, s, Vh = np.linalg.svd(x_t_haar)
	# plot_haar_plots(name_of_simulation+'_Haar.pdf', 
	# 					G, x_t_haar, Vh, 4)
	plot_haar_plots_w_vectors(name_of_simulation+'_Haar_expanded.pdf', 
						G, x_t_haar, Vh, 4)


	x_t_fft = perform_fft(x_t)
	# print(x_t_fft.shape)


	U, s, Vh = np.linalg.svd(x_t_fft)
	plot_fft_plots(name_of_simulation+'_fft.pdf', 
						G, np.abs(x_t_fft), np.abs(Vh), h, 4)






	# # ####################### APPENDIX #############################


	# Model 2
	name_of_simulation = 'app_w_distancing'

	G = G_7_three_variable_equal
	Q = create_SIS_matrix_w_distancing(	G, 1.5, 0.33, 
										reduction_percent = 0.2,
										n_infect_threshold = 4)

	states = enumerate_states(len(G.nodes), 2)
	x_0 = get_initial_state()
	# print(x_0)


	x_t = forward_euler(Q, x_0, h, n_t-1)
	x_0 = x_t[:,-1].reshape(-1,1)
	# print(x_0)
	# no transform
	x_t = forward_euler(Q, x_0, h, n_t-1)
	# print(x_t.shape)
	U, s, Vh = np.linalg.svd(x_t)

	# Done
	plot_network_and_states(name_of_simulation+'_network_and_transition.pdf', G, x_t, states, h)

	# Done
	left_args = {'scales':np.sqrt(s), 'n_vectors':4}
	right_args = {'scales':np.sqrt(s),'n_vectors':4}
	plot_singular_values_and_vectors(name_of_simulation+'_singular_stuff.pdf',
										s, G, U, states, left_args, Vh, h, right_args)














	### Other models
	# # Model 1
	# name_of_simulation = 'app_line_out_base'
	# n_days = 10
	# n_t = 2**10
	# h = n_days / n_t

	# G = G_7_line_out

	# Q = create_SIS_matrix(G, 1.5, 0.33)

	# states = enumerate_states(len(G.nodes), 2)
	# x_0 = get_initial_state()
	# print(x_0)

	# # no transform
	# x_t = forward_euler(Q, x_0, h, n_t-1)
	# print(x_t.shape)
	# U, s, Vh = np.linalg.svd(x_t)

	# # Done
	# plot_network_and_states(name_of_simulation+'_network_and_transition.pdf', G, x_t, states, h)

	# # Done
	# left_args = {'scales':np.sqrt(s), 'n_vectors':4}
	# right_args = {'scales':np.sqrt(s),'n_vectors':4}
	# plot_singular_values_and_vectors(name_of_simulation+'_singular_stuff.pdf',
	# 									s, G, U, states, left_args, Vh, h, right_args)










	# # Model 2
	# name_of_simulation = 'app_line_out_w_distancing'
	# n_days = 10
	# n_t = 2**10
	# h = n_days / n_t

	# G = G_7_line_out
	# Q = create_SIS_matrix_w_distancing(	G, 1.5, 0.33, 
	# 									reduction_percent = 0.2,
	# 									n_infect_threshold = 4)

	# states = enumerate_states(len(G.nodes), 2)
	# x_0 = get_initial_state()
	# print(x_0)

	# # no transform
	# x_t = forward_euler(Q, x_0, h, n_t-1)
	# print(x_t.shape)
	# U, s, Vh = np.linalg.svd(x_t)

	# # Done
	# plot_network_and_states(name_of_simulation+'_network_and_transition.pdf', G, x_t, states, h)

	# # Done
	# left_args = {'scales':np.sqrt(s), 'n_vectors':4}
	# right_args = {'scales':np.sqrt(s),'n_vectors':4}
	# plot_singular_values_and_vectors(name_of_simulation+'_singular_stuff.pdf',
	# 									s, G, U, states, left_args, Vh, h, right_args)


























	# # Model 3
	# name_of_simulation = 'app_branch_model'
	# n_days = 10
	# n_t = 2**10
	# h = n_days / n_t

	# G = G_7_branches
	# Q = create_SIS_matrix_variable(G,0.33)

	# states = enumerate_states(len(G.nodes), 2)
	# x_0 = get_initial_state()
	# print(x_0)

	# # no transform
	# x_t = forward_euler(Q, x_0, h, n_t-1)
	# print(x_t.shape)
	# U, s, Vh = np.linalg.svd(x_t)

	# # Done
	# plot_network_and_states(name_of_simulation+'_network_and_transition.pdf', G, x_t, states, h)

	# # Done
	# left_args = {'scales':np.sqrt(s), 'n_vectors':4}
	# right_args = {'scales':np.sqrt(s),'n_vectors':4}
	# plot_singular_values_and_vectors(name_of_simulation+'_singular_stuff.pdf',
	# 									s, G, U, states, left_args, Vh, h, right_args)


	# # x_t_fft = perform_fft(x_t)

	# # U, s, Vh = np.linalg.svd(x_t_fft)

