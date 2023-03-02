import numpy as np
from copy import deepcopy
# Import math Library
import math


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

    # other reference
    # point0 = lRes[151]
    # point1 = lRes[337]
    # point2 = lRes[10]

    # !!! use this reference
    # make normal X, Y axis, and make point0 become (0,0) coordinate
    distance = math.dist(lRes[151], lRes[9])
    # distance = 1
    point0 = lRes[9]
    point1 = deepcopy(point0)
    point2 = deepcopy(point0)
    # Not Change the axis
    point1[0] = point1[0] + distance
    point2[1] = point1[1] + distance

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

    # print(vx, vy, vz, vd)
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
    # print("lm", lm)
    m = np.linalg.inv(lm)
    # print("m", m)
    # print("point0", point0)

    # print("LRES")
    # for lres in lRes:
    #     print(lres)

    # remember the shape is [468, 3], we made to 4 because for matrix multiplication
    pointDistance = np.ones([468, 4])
    previousPoint = lRes  # (Before Normalisation)
    pointDistance[:, 0:3] = previousPoint
    pointNormalisation = np.matmul(m, pointDistance.transpose()).transpose()

    pointResult = np.zeros(shape=(468, 3))
    numberCount = 0
    # print("pointNormalisation")
    for point in pointNormalisation:
        # ?? we want to make the coordinate axis bescome like we ussually know
        # ?? because in computer the Y axis is different, the Y become more positive when we down so we need to * -1
        # ?? for other axis, X is same, and Z is pretty same with note the 0,0 point is behind us when we facing webcam
        # ?? Z axis become more positif if toward us, and become more negative if moving toward the webcam
        point[1] = point[1] * -1
        pointResult[numberCount] = point[:3]
        numberCount = numberCount + 1

    # example point of face (9, 151)
    # Before Normalisation
    # print("9", lRes[9])
    # print("151", lRes[151])
    # After Normalisation
    # print("P9", pointNormalisation[9])
    # print("P151", pointNormalisation[151])

    return pointResult
