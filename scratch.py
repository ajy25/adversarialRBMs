import numpy as np


def remove_diagonal_fast(matrix):
    # Get the matrix size
    n = matrix.shape[0]

    # Create an offset array to efficiently remove the diagonal elements
    offset = np.arange(1, n + 1)

    # Create an empty matrix to store the result
    result = np.zeros((n, n - 1))

    # Use advanced indexing to efficiently remove the diagonal elements
    result = matrix[np.arange(n)[:, None], offset - 1]

    return result


# Example usage
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
result = remove_diagonal_fast(matrix)
print(result)


