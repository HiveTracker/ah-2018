import os
import math
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
dname = os.path.dirname(os.path.realpath(__file__))

viveidx = 0 # the vive controller measurements
hivesensoridx = 2 # the measurement used for comparison
axes = [0,2] # the axes (X and Z) used for tracing the floor polygon
measurements = ['../Data/controller_rig_measurements2017-06-27T19_52_31.csv',
                '../Data/sensor0_rig_measurements2017-06-27T19_53_34.csv',
                '../Data/sensor2_rig_measurements2017-06-27T19_53_34.csv',
                '../Data/sensorAvg_rig_measurements2017-06-27T19_53_34.csv']
data = [np.genfromtxt(path,delimiter=' ') for path in measurements]
vivedata = data[viveidx][:,axes]
hivedata = data[hivesensoridx][:,axes]

scale = 1
shiftx = -0.1
shifty = -0.05

plt.rcParams['font.size'] = 12
plt.plot(vivedata[:,0],vivedata[:,1],'b',label='controller')
plt.plot(hivedata[:,0]*scale+shiftx,
         hivedata[:,1]*scale+shifty,
         'orange',label='scaled/shifted average')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()

# piece-wise linear fit
controlpoints = np.array([(-0.595, 0.060),
                          (-0.393, 0.316),
                          (-0.033, 0.333),
                          (0.225, 0.095),
                          (0.050, -0.177),
                          (-0.374, -0.188)])

# Finds the closest point on a line-segment
# p - point to test
# a - line segment endpoint A
# b - line segment endpoint B
def closestpointonsegment(p,a,b):
    # Find projection of point p on the line defined by: a + t * (b - a)
    linevector = b - a
    t = np.dot(p - a, linevector) / np.dot(linevector, linevector)
    t = max(0, min(1, t)) # clamp to line segment
    return a + t * linevector

# Returns the distance from a point to a line-segment
# p - point to test
# a - line segment endpoint A
# b - line segment endpoint B   
def distancetolinesegment(p,a,b):
    closestpoint = closestpointonsegment(p,a,b)
    return norm(closestpoint - p)
    
# Computes the distance from a point to a closed polygon
# p - point to test
# poly - closed polygon defined by a list of points
def distancetopolygon(p,poly):
    polyB = np.concatenate((poly[1:],poly[0:1]))
    distances = [distancetolinesegment(p,a,b) for a,b in zip(poly,polyB)]
    return np.min(distances)

# Validation of distance calculations: distance map of grid to polygon
x = np.arange(-0.65,0.25,0.005)
y = np.arange(-0.25,0.4,0.005)
vs = np.array([[distancetopolygon(np.array([vx,vy]),
                                  controlpoints)
                                  for vx in x]
                                  for vy in y])
plt.figure()
plt.imshow(vs)
plt.show()

# Compute estimate of tracking error by average distance of points to polygon
viveerror = np.average([distancetopolygon(p,controlpoints) for p in vivedata])
hiveerror = np.average([distancetopolygon(p,controlpoints) for p in hivedata])
