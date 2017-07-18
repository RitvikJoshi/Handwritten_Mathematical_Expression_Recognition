'''
Geometric features

Input: Stroke pair

Output: geometri feature

Author
Ritvik Joshi
Rahul Dashora

'''


import numpy as np


def Horizontal_offset(S1,S2):
    """
    Return 1 feature Horizontal offset
    :param S1:
    :param S2:
    :return:

    """
    starting_pt = S2[0]
    ending_pt = S1[len(S1)-1]

    dist = np.sqrt(np.sum((ending_pt-starting_pt)**2))
    return dist

def DistBBcenter(S1,S2):
    """
    Return 1 feature Distance between BB center
    :param S1:
    :param S2:
    :return:
    """

    minx_S1 = min(S1[:,0])
    maxx_S1 = max(S1[:,0])
    miny_S1 = min(S1[:,1])
    maxy_S1 = max(S1[:,1])

    bbcenter_s1 = np.array((((maxx_S1-minx_S1)/2+minx_S1),((maxy_S1-miny_S1)/2+miny_S1)))

    minx_S2 = min(S2[:,0])
    maxx_S2 = max(S2[:,0])
    miny_S2 = min(S2[:,1])
    maxy_S2 = max(S2[:,1])

    bbcenter_s2 = np.array((((maxx_S2-minx_S2)/2+minx_S2),((maxy_S2-miny_S2)/2+miny_S2)))

    dist = np.sqrt(np.sum((bbcenter_s1-bbcenter_s2)**2))

    return dist

def MinDistance_BBOverlapping(S1,S2):
    """
    Return 2 feature Min distance between the Bounding box and BB Overlapping
    :param S1:
    :param S2:
    :return:
    """

    maxx_s1 = max(S1[:,0])
    index = list(S1[:,0]).index(maxx_s1)

    max_pt_1 = S1[index]

    minx_s2 = min(S2[:,0])
    index = list(S2[:,0]).index(minx_s2)
    min_pt_S2= S2[index]

    dist = np.sqrt(np.sum((max_pt_1-min_pt_S2)**2))
    if(max_pt_1[0]<min_pt_S2[0]):
        return dist,0
    else:
        return 0,dist


def DistAverageCenter(S1,S2):
    """
    Return 1 feature distance between average center of strokes
    :param S1:
    :param S2:
    :return:
    """

    avg_s1 = np.array((np.average(S1[:,0]),np.average(S1[:,1])))
    avg_s2 = np.array((np.average(S2[:,0]),np.average(S2[:,1])))

    dist = np.sqrt(np.sum((avg_s2-avg_s1)**2))

    return dist

def MaximalPairDist(S1,S2):
    """
    Return 1 feature Distance between the farthest points
    :param S1:
    :param S2:
    :return:
    """

    minx_s1 = min(S1[:,0])
    index = list(S1[:,0]).index(minx_s1)

    min_pt_1 = S1[index]

    maxx_s2 = max(S2[:,0])
    index = list(S2[:,0]).index(maxx_s2)

    max_pt_2 = S2[index]

    dist = np.sqrt(np.sum((min_pt_1-max_pt_2)**2))

    return dist

def VertDistBBCenter(S1,S2):
    """
    Return 1 feature vertical distance between center of BB
    :param S1:
    :param S2:
    :return:
    """
    miny_S1 = min(S1[:,1])
    maxy_S1 = max(S1[:,1])

    centerY_S1 = (maxy_S1-miny_S1)/2

    miny_S2 = min(S2[:,1])
    maxy_S2 = max(S2[:,1])

    centerY_S2 = (maxy_S2-miny_S2)/2

    Vdist=centerY_S2-centerY_S1

    return Vdist

def DistBeginpts(S1,S2):
    """
    Retrun 1 feature distance between begin pts
    :param S1:
    :param S2:
    :return:
    """
    starting_pt_S2 = S2[0]
    starting_pt_S1 = S1[0]

    dist = np.sqrt(np.sum((starting_pt_S1-starting_pt_S2)**2))
    return dist

def DistEndpts(S1,S2):
    """
    Retrun 1 feature distance between end pts
    :param S1:
    :param S2:
    :return:
    """
    end_pt_S2 = S2[len(S2)-1]
    end_pt_S1 = S1[len(S1)-1]

    dist = np.sqrt(np.sum((end_pt_S1-end_pt_S2)**2))
    return dist


def Vert_offset(S1,S2):
    """
    Returns 2 features vertical offset between starting pts and ending points
    :param S1:
    :param S2:
    :return:
    """
    starting_pt_S2 = S2[0]
    starting_pt_S1 = S1[0]

    vert_dist_start = np.sqrt((starting_pt_S1[1] - starting_pt_S2[1])**2)

    end_pt_S2 = S2[len(S2)-1]
    end_pt_S1 = S1[len(S1)-1]

    vert_dist_end = np.sqrt((end_pt_S1[1] - end_pt_S2[1])**2)

    return vert_dist_start,vert_dist_end

def BackwardMovement(S1,S2):
    """
    Return 1 feature Backward Movement
    :param S1:
    :param S2:
    :return:
    """
    end_pt_S1 = S1[len(S1)-1]

    starting_pt_S2 = S2[0]

    Manhattan_dist =np.sum (end_pt_S1-starting_pt_S2)

    return Manhattan_dist

def Parallelity(S1,S2):
    """
    Return 4 feature Parallelity
    :param S1:
    :param S2:
    :return:
    """

    s1_vect = np.array((S1[0],S1[len(S1)-1]))
    s2_vect = np.array((S2[0],S2[len(S2)-1]))

    if((s1_vect[0][0]==s1_vect[1][0] and s1_vect[0][1]==s1_vect[1][1]) or (s2_vect[0][0]==s2_vect[1][0] and s2_vect[0][1]==s2_vect[1][1])):
        return [0,0,0,0]

    dot_prod = np.dot(s1_vect,s2_vect)
    s1_vect_mag = np.linalg.norm(s1_vect)
    s2_vect_mag = np.linalg.norm(s2_vect)
    denominator=(s1_vect_mag*s2_vect_mag)
    theta = np.arccos((dot_prod/denominator))

    return theta.flatten()

def WritingSlope(S1,S2):
    """
    retrun 1 feature angle between the ending pt of s1 and starting pt os s2
    :param S1:
    :param S2:
    :return:
    """

    end_pt_S1 = S1[len(S1)-1]

    starting_pt_S2 = S2[0]

    diff = end_pt_S1 -starting_pt_S2

    theta = np.arctan2(diff[1],diff[0])

    return theta

def sizediff(S1,S2):
    """
    1 feature Difference in parameter of BB
    :param S1:
    :param S2:
    :return:
    """

    minx_S1 = min(S1[:,0])
    maxx_S1 = max(S1[:,0])
    miny_S1 = min(S1[:,1])
    maxy_S1 = max(S1[:,1])

    length_s1 = maxy_S1-miny_S1
    breadth_s1 = maxx_S1-minx_S1

    parameter_s1 = 2*(length_s1+breadth_s1)

    minx_S2 = min(S2[:,0])
    maxx_S2 = max(S2[:,0])
    miny_S2 = min(S2[:,1])
    maxy_S2 = max(S2[:,1])

    length_s2 = maxy_S2-miny_S2
    breadth_s2 = maxx_S2-minx_S2

    parameter_s2 = 2*(length_s2+breadth_s2)

    diff =parameter_s1-parameter_s2

    return diff

def StrokeAngle(S1,S2):

    minx_S1 = min(S1[:,0])
    maxx_S1 = max(S1[:,0])
    miny_S1 = min(S1[:,1])
    maxy_S1 = max(S1[:,1])

    # c1Y = (maxy_S1-miny_S1)/2+miny_S1
    # breadth_s1 = (maxx_S1-minx_S1)/2 +minx_S1

    # parameter_s1 = 2*(length_s1+breadth_s1)

    minx_S2 = min(S2[:,0])
    maxx_S2 = max(S2[:,0])
    miny_S2 = min(S2[:,1])
    maxy_S2 = max(S2[:,1])

    c1 = (maxy_S2-miny_S1)/2 + miny_S1
    c2= (maxx_S2-minx_S1)/2 + minx_S1




    firstS1 = S1[0]
    lastS1 = S1[-1]
    firstS2 = S2[0]
    lastS2 = S2[-1]


    theta1 = np.arctan2(lastS1[1]-c2,lastS1[0]-c1)
    theta2 = np.arctan2(lastS2[1] - c2, lastS2[0] - c1)
    theta3 = np.arctan2(firstS1[1] - c2, firstS1[0] - c1)
    theta4 = np.arctan2(firstS2[1] - c2, firstS2[0] - c1)

    return [theta1,theta2,theta3,theta4]