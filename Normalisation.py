import numpy as np


def nomralisation_faceLandmark(faceLandMark: list, image):
    # the shape is [468, 3]
    heigth, width, dimension = image.shape
    point0 = faceLandMark[9]
    # make normal X, Y axis, and make point0 become (0,0) coordinate
    point1 = point0
    point2 = point0
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
    vy = vz/np.linalg.norm(vy)

    pass
