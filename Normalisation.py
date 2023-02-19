import numpy as np
from copy import deepcopy


def normalisation_faceLandmark(faceLandMark: list, image):
    # the shape is [468, 3]
    heigth, width, dimension = image.shape
    # lRes artinya resolution
    lRes = np.zeros([468, 3])
    for i in range(468):
        lRes[i, 0] = faceLandMark[i][0]*heigth
        lRes[i, 1] = faceLandMark[i][1]*width

        # in this section we just use horizontal image
        if heigth >= width:
            lRes[i, 2] = faceLandMark[i][2]*heigth
        else:
            lRes[i, 2] = faceLandMark[i][2]*width

    point0 = lRes[9]
    # make normal X, Y axis, and make point0 become (0,0) coordinate
    point1 = deepcopy(point0)
    point2 = deepcopy(point0)
    point1[0] = point1[0] + 1
    point2[1] = point1[1] + 1

    # calculate the vector
    vx = point1 - point0
    vd = point2 - point1
    # make length of vx,vd become 1
    vx = vx/np.linalg.norm(vx)
    vd = vd/np.linalg.norm(vd)

    # get vector z
    vz = np.cross(vx, vd)
    # make length of vz become 1
    vz = vz/np.linalg.norm(vz)

    # get vector y
    vy = np.cross(vz, vx)
    # make length of vy become 1
    vy = vy/np.linalg.norm(vy)

    # calculate the matrix
    lm = np.zeros([4, 4])
    # print(vx, "vx")
    # print(vy, "vy")
    # print(vz, "vz")
    lm[0:3, 0] = vx
    lm[0:3, 1] = vy
    lm[0:3, 2] = vz
    lm[0:3, 3] = point0
    lm[3, 3] = 1
    m = np.linalg.inv(lm)

    # remember the shape is [468, 3], we made to 4 because for matrix multiplication
    pointDistance = np.ones([468, 4])
    pointPespective = point0
    pointDistance[:, 0:3] = pointPespective
    pointNormalisation = np.matmul(m, pointDistance.transpose()).transpose()

    return pointNormalisation
