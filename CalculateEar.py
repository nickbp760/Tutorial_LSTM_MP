import math


# Left in picture like Iris.py
def calculateLeftEAR(mesh_points):
    distanceE1 = math.dist(mesh_points[385], mesh_points[380])
    distanceE2 = math.dist(mesh_points[387], mesh_points[373])
    distanceE3 = math.dist(mesh_points[362], mesh_points[263])
    EAR = (distanceE1 + distanceE2) / 2 * distanceE3
    return EAR


# right in picture like Iris.py
def calculateRightEAR(mesh_points):
    distanceE1 = math.dist(mesh_points[158], mesh_points[153])
    distanceE2 = math.dist(mesh_points[160], mesh_points[144])
    distanceE3 = math.dist(mesh_points[133], mesh_points[33])
    EAR = (distanceE1 + distanceE2) / 2 * distanceE3
    return EAR
