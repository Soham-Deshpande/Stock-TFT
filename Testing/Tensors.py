#testing tensors

import numpy as np

v1 = [1, 0, 0, 1]
v2 = [1, 0, 0, 1]

v1 = np.array(v1).reshape(2, 2)
v2 = np.array(v2).reshape(2, 2)

i = [[num1 * num2 for num1 in elem1 for num2 in v2[row]] for elem1 in v1 for row in range(len(v2))]

print(' ')

print("With list comprehension: ")
print(np.array(i).reshape(4, 4))

print(' ')

print("With numpy kron: ")
print(np.kron(v1, v2))
