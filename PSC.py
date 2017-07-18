"""
Parzen Shape Context Feature

Input : All stroke information
Output : Shape Context Feature of size (30x3)

Author
Ritvik Joshi
Rahul Dashora
"""

import math
import numpy as np
def getAllPSC(normalizedData,pair):
    """

    :param normalizedData:
    :return:
    """
    PSC_feature=np.zeros((0))
    mat = normalizedData

    s1 = mat[pair[0]]
    s2 = mat[pair[1]]
    center,rad = getPSCParam([s1,s2])
    feature_s1=getPSC(center,rad,[s1])
    PSC_feature=np.append(PSC_feature,feature_s1)
    feature_s2=getPSC(center, rad, [s2])
    PSC_feature=np.append(PSC_feature,feature_s2)
    otherStrokes = mat[:pair[0]]+mat[pair[0]+1:pair[1]]+mat[pair[1]:]
    feature_s3=getPSC(center, rad, otherStrokes)
    PSC_feature=np.append(PSC_feature,feature_s3)

    return PSC_feature

def getPSCParam(strokePair):

    maxY = 0
    minY = 99999
    maxX = 0
    minX = 99999
    for m in strokePair:
        maxX = max(max(m[:, 0]), maxX)
        minX = min(min(m[:, 0]), minX)
        maxY = max(max(m[:, 1]), maxY)
        minY = min(min(m[:, 1]), minY)
    rangeX = maxX-minX
    rangeY = maxY-minY
    center = [(rangeX/2) +minX , (rangeY/2)+minY]
    rad = math.sqrt(((rangeX/2)**2 + (rangeY/2)**2))
    return center,rad

def getPSC(center,rad,strokeList,angle=6,dist=5):

    numOfBins = angle * dist
    polarHist = [0]*numOfBins
    distBinSize = rad/dist
    totalPoints = 1
    for stroke in strokeList:
        for stk in stroke:
            xDist = (stk[0]-center[0])
            yDist = (stk[1]-center[1])
            distBin = int(math.sqrt((xDist**2 + yDist**2))/distBinSize)
            if distBin >= dist:
                continue
            totalPoints+=1
            phase = math.degrees(math.atan2(yDist,xDist))

            if phase < 0 : phase = 360 + phase
            angleBin = int(phase/int(360/angle))

            binIdx = distBin * angle + angleBin
            if(binIdx>=30):
                continue
            polarHist[binIdx]+=1
    #print(polarHist)
    polarHist = np.divide(polarHist,totalPoints,dtype='float32')
    #print(polarHist)
    return polarHist