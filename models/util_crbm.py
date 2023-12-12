import numpy as np
from scipy.spatial import cKDTree, distance

eps = np.finfo(np.float32).eps

def partition_into_batches(data: list[np.ndarray], batch_size: int,
                          batch_rng_seed: int = 1):
   """
   Partitions the data into batches

   @args
   - data: list[np.ndarray ~ (n_examples, n_vis)] << each element in the list
   represents a distinct element (i.e. input X, label Y, etc)
   - batch_size: int
   - batch_rng_seed: rng seed for shuffling the data

   @returns
   - output: list[list[np.ndarray ~ (batch_size, n_vis)]]
   - orig_idx: np.ndarray ~ (n_examples)
   """
   m = data[0].shape[0]
   rand_idx = np.arange(m)
   rng = np.random.default_rng(seed=batch_rng_seed)
   rng.shuffle(rand_idx)
   orig_idx = np.argsort(rand_idx)
   for i in range(len(data)):
       data[i] = data[i][rand_idx]
   num_batches = m // batch_size
   output = []
   for i in range(num_batches):
       start_idx = i * batch_size
       end_idx = start_idx + batch_size
       batch = []
       for j in range(len(data)):
           batch.append(data[j][start_idx:end_idx])
       output.append(batch)
   if num_batches * batch_size != m:
       final_batch = []
       for j in range(len(data)):
           final_batch.append(data[j][num_batches*batch_size:])
       output.append(final_batch)
   return output, orig_idx

def kth_smallest(matrix, k):
   kth_values = np.partition(matrix, k, axis=1)[:, k]
   return kth_values.reshape(-1, 1)

def k_nearest_neighbors2(data, query, k):
    distances = distance.cdist(query, data, 'euclidean')
    nn_bool = distances < kth_smallest(distances, k)
    # num_neighbors = np.sum(nn_bool, axis=1)
    # num_needed = k - num_neighbors
    # nn_equal = distances == kth_smallest(distances, k)
    # for i in range(nn_equal.shape[0]):
    #     true_indices = np.nonzero(nn_equal[i, :])[0]
    #     if len(true_indices) > num_needed[i]:
    #         selected_indices = np.random.choice(true_indices, size=num_needed[i], replace=False)
    #         nn_bool[i, selected_indices] = True
    #     else:
    #         nn_bool[i, true_indices] = True
    return nn_bool, distances

def inverse_distance_sum(distances, boolean_mask = None):
   if boolean_mask is None:
       return np.sum(1 / (distances + eps), axis=1)
   return np.sum(1 / (distances + eps) * boolean_mask, axis=1)

def k_nearest_neighbors(data, query, k=1):
    tree = cKDTree(data)
    distances, indices = tree.query(query, k=k)
    return indices, distances

# def inverse_distance_sum(distances):
#     return np.sum(1 / (distances + eps), axis=1)

def kth_nearest_neighbor_distance(data, query = None, k = 1):
    """
    If query is None, then query = data. In this case we exclude self when 
    computing nearest neighbors. 
    """
    if query is None:
        # when finding nearest neighbors for X_i, we can exclude X_i
        # from data by simpling adding one to k, since the 1-st 
        # nearest neighbor would be X_i. 
        query = data
        k += 1
    distances = distance.cdist(query, data, metric='euclidean')
    top_k_distances = np.sort(distances, axis=1)[:, :k]
    return top_k_distances[:, -1]

def approx_kl_div(p_samples: np.ndarray, q_samples: np.ndarray, k=1):
    """
    Estimates KL(p || q) given samples from p and q, as described in equation 5 
    of the 2009 paper on multi-dimensional KL divergence estimation, 
    https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf

    @args
    - p_samples: np.array ~ (n, d)
    - q_samples: np.array ~ (m, d)
    - k: k-NN

    @returns
    - float
    """
    n = len(p_samples)
    m = len(q_samples)
    d = p_samples.shape[1]
    nu = kth_nearest_neighbor_distance(q_samples, p_samples, k=k)
    rho = kth_nearest_neighbor_distance(p_samples, k=k)
    return (d * np.mean(np.log2(nu / rho), axis=0) + np.log2(m / (n - 1))).item()

if __name__ == "__main__":
    X = np.arange(7, 19, step=2).reshape(-1, 1)
    Y = np.arange(25).reshape(-1, 1)
    indices, distances = k_nearest_neighbors(Y, X, k=2)
    ind_cond = indices >= 6
    print(inverse_distance_sum(distances, ind_cond))
    print(inverse_distance_sum(distances))
    print()

    indices, distances = k_nearest_neighbors2(Y, X, k=2)
    ind_cond = indices.copy()
    ind_cond[:, :6] = False
    print(inverse_distance_sum(distances, ind_cond))
    print(inverse_distance_sum(distances, indices))

    # sigma_1 = 2
    # sigma_2 = 1
    # true_kl = np.log(sigma_2 / sigma_1) + (sigma_1 ** 2 / (2 * sigma_2 ** 2)) - 1/2
    # p_sample = np.random.standard_normal(size=10000).reshape(-1, 1) * sigma_1 + 10
    # print(p_sample)
    # q_sample = np.random.standard_normal(size=10000).reshape(-1, 1) + 10
    # print(q_sample)
    # print(approx_kl_div(p_sample, q_sample))
    # print(true_kl)




