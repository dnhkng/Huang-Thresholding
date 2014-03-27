import math
import numpy

def getAngle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))

def rotatePoint(point, theta):
    """docstring for rotate"""
    cs = math.cos(theta)
    sn = math.sin(theta)
    px = point[0] * cs - point[1] * sn
    py = point[0] * sn + point[1] * cs
    return (px, py)


def findCentroid(triangle):
    """The centroid of a finite set of points the arerage of each set of orthanogal point"""
    centroid = numpy.mean(triangle, axis=0)
    return centroid

def findScaleFactor(pointsA,pointsB):
    """Finds the length between points within a set, and gets the average ratio of the lengths between sets"""
    pointsA_rotated = numpy.vstack((pointsA[1:,:], pointsA[0,:]))-pointsA
    pointsB_rotated = numpy.vstack((pointsB[1:,:], pointsB[0,:]))-pointsB
    scale = numpy.mean(numpy.sqrt(numpy.square(pointsB_rotated[:,0])+numpy.square(pointsB_rotated[:,1]))/numpy.sqrt(numpy.square(pointsA_rotated[:,0])+numpy.square(pointsA_rotated[:,1])))
    return scale

def translate(l,c):
    """docstring for translate"""
    v = []
    for i in range(len(l)):
        p = [[],[]]
        p[0] = l[i][0] - c[0]
        p[1] = l[i][1] - c[1]
        v.append(p)
    return v

def findAngle(L1, L2):
    """docstring for center"""
    L1C = findCentroid(L1)
    L2C = findCentroid(L2)
    L1T = translate(L1,L1C)
    L2T = translate(L2,L2C)
    angle1 = getAngle(L1T[0],L2T[0])
    angle2 = getAngle(L1T[1],L2T[1])
    angle3 = getAngle(L1T[2],L2T[2])
    angle =  (angle1 + angle2 + angle3)/3
    return angle

def affineParams(L1, L2):
    R = findAngle(L1,L2)
    S = findScaleFactor(L1,L2)
    C1 = findCentroid(L1)
    C2 = findCentroid(L2)
    return (R,S,C1,C2)

def transform(L1, params):
    R = params[0]
    S = params[1]
    C1 = params[2]
    C2 = params[3]
    v = []
    for i in L1:
        p1 = [i[0]-C1[0], i[1]-C1[1]]
        p2 = rotatePoint(p1, -R)
        p3 = [p2[0]*S, p2[1]*S]
        p4 = [p3[0]+C2[0], p3[1]+C2[1]]
        v.append(p4)
    v = numpy.array(v)    
    return v





if __name__ == '__main__':
    pic = numpy.array([[424,792],[828,872],[1096,549],[911,158],[482,170],[296,482]])#,[708,545]])          #([[983.0,683.0],[1088.0,484.0],[922.0,531.0],[524.0,152.0],[448.0,780.0]])#([[845.0,134.0], [1054.0,301.0], [1079.0,546.0], [351.0,566.0], [501.0,806.0]])
    robot = numpy.array([[-20.0,43.0],[-31.2,11.6],[-64.5,6.3],[-83.6,36.1],[-64.8,64.6],[-35.7,64.3]])#,[-49.1,33.7]])         #([[-50.3,8.8],[-68.5,9.8],[-58.2,19.5],[-67.6,62.4],[-21.5,41.6]])#([[-81.4,40.7],[-78.6,20.5],[-63.3,8.5],[-32.4,56.7],[-22.4,36.9]])
    #params = affineParams(pic,robot)
    
    
    params = (2.1120208757310146, 0.080280775003210444, array([ 672.83333333,  503.83333333]), array([-49.96666667,  37.65      ]))
    out = transform(pic, params)

    print params
    p = (2.1163087380926116, 0.080631675816675547, numpy.array([ 793.,  526.]), numpy.array([-53.22,  28.42]))
    outt = transform([[409.0,509]], p)
    print outt

    x = pic[:,0]
    y = pic[:,1]
    x1 = robot[:,0]
    y1 = robot[:,1]
    x2 = out[:,0]
    y2 = out[:,1]

    import matplotlib.pyplot as plt

    #plt.scatter(x,y, color='r')
    plt.scatter(x1,y1, color='b')
    plt.scatter(x2,y2, color='g')
    plt.show()

