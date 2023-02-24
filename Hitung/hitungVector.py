# program to compute magnitude of a vector
# importing required libraries
import numpy
import math


# function definition to compute magnitude o f the vector
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


# computing and displaying the magnitude of the vector
print('Magnitude of the Vector:', magnitude(numpy.array([6.23336792, 137.70465851, -15.38446069])))
