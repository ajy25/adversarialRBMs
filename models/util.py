import numpy as np
from scipy.spatial import distance, cKDTree
from scipy.stats import entropy

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

def k_nearest_neighbors(data, query, k=1):
    tree = cKDTree(data)
    distances, indices = tree.query(query, k=k)
    return indices, distances

def inverse_distance_sum(distances):
    return np.sum(1 / (distances + eps), axis=1)

def kth_nearest_neighbor_distance(X, Y, k=1, distance_type="euclidean"):
    distances_matrix = distance.cdist(X, Y, distance_type)
    k_nearest_indices = np.argpartition(distances_matrix, k-1, axis=1)[:, :k]
    kth_distances = np.take_along_axis(distances_matrix, k_nearest_indices, axis=1)
    distances = np.min(kth_distances, axis=1)
    return distances

def approx_kl_div(p_samples: np.ndarray, q_samples: np.ndarray):
    """
    Estimates KL(p || q) given samples from p and q, as described in equation 5 
    of the 2009 paper on multi-dimensional KL divergence estimation, 
    https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf

    p_samples: np.array ~ (n, d)
    q_samples: np.array ~ (m, d)
    """
    n = len(p_samples)
    m = len(q_samples)
    d = p_samples.shape[1]
    nu = kth_nearest_neighbor_distance(p_samples, q_samples, k=1)
    rho = kth_nearest_neighbor_distance(p_samples, q_samples, k=1)
    return d * np.mean(np.log2(nu / rho)) + np.log2(m / n - 1)

if __name__ == "__main__":
    X = np.zeros(5).reshape(-1, 1)
    Y = np.arrange()
    print(k_nearest_neighbors(X, Y, 1))





