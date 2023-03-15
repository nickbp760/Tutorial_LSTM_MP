import math


def calculateEAR(mesh_points):
    distanceE1 = math.dist(mesh_points[385], mesh_points[380])
    distanceE2 = math.dist(mesh_points[387], mesh_points[373])
    distanceE3 = math.dist(mesh_points[362], mesh_points[263])
    EAR = (distanceE1 + distanceE2) / 2 * distanceE3
    return EAR
