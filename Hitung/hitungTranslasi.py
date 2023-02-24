import numpy as np
lm = [[1, -0, 0, 297.1108532],
      [0, 1, -0, 331.45641327],
      [0, 0, 1, -10.82884908],
      [0, 0, 0, 1]]
m = np.linalg.inv(lm)
p = [297.1108532, 331.45641327, -10.82884908, 1]
p = np.array(p)
m = np.array(m)
lm = np.array(lm)
pointNormalisation = np.matmul(m, p.transpose()).transpose()
print(pointNormalisation)

# translation example, this is like our normalisation
# because our normalisation using this
# point0 = lRes[9]
# point1 = deepcopy(point0)
# point2 = deepcopy(point0)
# point1[0] = point1[0] + 1
# point2[1] = point1[1] + 1
