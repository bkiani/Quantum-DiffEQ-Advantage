import numpy as np
from scipy.sparse import csr_matrix

def forward_euler(A, x_0, h = 0.1, n_times = 100, store_each_time = True, sparsify = False):
	if store_each_time:
		x_t = np.zeros((len(x_0), n_times+1))
		x_t[:,0] = x_0.reshape(-1)

	if sparsify:
		A = csr_matrix(A)

	x = x_0
	for i in range(n_times):
		if sparsify:
			x = x+h*A.dot(x)
		else:
			x = x+h*A@x
		if store_each_time:
			x_t[:,i+1] = x.reshape(-1)

	if store_each_time:
		return x_t
	else:
		return x


