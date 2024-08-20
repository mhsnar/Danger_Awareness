import numpy as np

# Example data (replace these with your actual arrays)
u_H_values = np.array([
    [np.array([[-1.], [-1.]]), np.array([[-0.5], [-1.]]), np.array([[0.], [-1.]]), np.array([[0.5], [-1.]]), np.array([[1.], [-1.]])],
    [np.array([[-1.], [-0.5]]), np.array([[-0.5], [-0.5]]), np.array([[0.], [-0.5]]), np.array([[0.5], [-0.5]]), np.array([[1.], [-0.5]])],
    [np.array([[-1.], [0.]]), np.array([[-0.5], [0.]]), np.array([[0.], [0.]]), np.array([[0.5], [0.]]), np.array([[1.], [0.]])],
    [np.array([[-1.], [0.5]]), np.array([[-0.5], [0.5]]), np.array([[0.], [0.5]]), np.array([[0.5], [0.5]]), np.array([[1.], [0.5]])],
    [np.array([[-1.], [1.]]), np.array([[-0.5], [1.]]), np.array([[0.], [1.]]), np.array([[0.5], [1.]]), np.array([[1.], [1.]])]
], dtype=object)

u_H = np.array([[-1.], [0.5]])

# Find the index where u_H matches one of the arrays in u_H_values
def find_index(u_H_values, u_H):
    for i in range(u_H_values.shape[0]):
        for j in range(u_H_values.shape[1]):
            if np.array_equal(u_H_values[i, j], u_H):
                return (i, j)
    return None

index = find_index(u_H_values, u_H)
ss=index[0]
vssv=index[1]
print(f'The index of the matching array is: {index}')
