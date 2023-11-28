import numpy as np
from scipy.spatial import distance

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
   kth_values = np.partition(matrix, k, axis=1)[:,k]
   return kth_values.reshape(-1, 1)

def k_nearest_neighbors(data, query, k, distance_type='euclidean'):
   distances = distance.cdist(query, data, distance_type)
   nn_bool = distances <= kth_smallest(distances, k)
   return nn_bool, distances

def inverse_distance_sum(distances, index):
   return np.sum(1 / (distances + eps) * index, axis=1)









