"""
Line of Sight Algorithm
Program creates a line of sight graph for each stroke/symbol
Input
Stroke/symbol information

Output
Line of Sight graph with 1 indicating as visibility between the strokes/symbol

Author
Rahul Dashora
Ritvik Joshi
"""


import numpy as np
import math
import copy
#from scipy.spatial import ConvexHull
import cv2




def getLOSGraph(allStrokes):
    E = []
    EdgeGraph =  np.zeros((len(allStrokes),len(allStrokes)))
    BBCenter = []

    strokeDist = np.zeros((len(allStrokes),len(allStrokes)))

    # Get bounding box center for each strokw
    for stroke in allStrokes:
        maxXS = max(stroke[:,0])
        minXS = min(stroke[:,0])
        maxYS = max(stroke[:,1])
        minYS = min(stroke[:,1])
        BBCenter.append([(maxXS-minXS)/2+minXS,(maxYS-minYS)/2+minYS])

    #Get inter stroke distances
    #   p1 r1 r2 r3
    #   p2 r1 r2 r3
    #   p3 r1 r2 r3
    #
    idx = 0
    for center1 in BBCenter:
        dist = []
        index=0
        for center2 in BBCenter:
            #dist.append(math.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2))
            dist.append((index,math.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)))
            index+=1
        x =  copy.deepcopy(dist)
        #print(x)
        x.sort(key= lambda x:x[1])
        #print(x)
        # Get rank for each element
        idx1 =0
        for e in x:
            #i = x.index(e)
            strokeDist[idx,idx1] = e[0]
            idx1 +=1
        idx+=1
    #print(strokeDist)

    numOfStrokes = len(allStrokes)
    #for each stroke that belongs to set S
    for sIdx in range(numOfStrokes):
        #U = set([(0,2*math.pi)])
        U = range(360)
        #rank each stroke by distance from stroke s
        ranks = list(strokeDist[sIdx])
        #print(ranks,numOfStrokes)
        for r in range(numOfStrokes):
            tIdx=0

            tIdx = int(ranks[r])
            if sIdx == tIdx : continue
            #print(tIdx)
            t = allStrokes[tIdx]
            flag=True
            thetaMin = 999999
            thetaMax = -99999
            hullmin = 0
            hullmax = 0
            #convexhull and its min max
            CH = getConvexHull(t)
            v= np.asarray(CH)
            v=list(v[:,1])
            maxind=v.index(max(v))
            minind=v.index(min(v))
            CH=[CH[maxind],CH[minind]]
            #########################3
            for node in CH:
                w =[node[0]-BBCenter[sIdx][0],node[1]-BBCenter[sIdx][1]]
                #h = [1,0]
                # calculating theta
                # print('BBC',BBCenter[sIdx])
                # print('node',node)
                # print('w',w)
                theta = math.atan2(-w[1],w[0])
                #print('theta',math.degrees(theta))
                if theta <0 : theta = 2*math.pi + theta

                # if(flag):
                #     #thetaMin = theta
                #     #thetaMax = theta
                #     #hullmax=node
                #     #hullmin=node
                #     flag=False
                if thetaMin > theta:
                    thetaMin = theta
                    #print('theraMin',thetaMin)
                    #print('hullMin',hullmin)
                    hullmin=node
                if thetaMax < theta:
                    thetaMax=theta
                    #print('thetaMax',thetaMax)
                    hullmax=node


                #thetaMin = min(thetaMin,theta)
                #thetaMax = max(thetaMax,theta)
            thetaMin = int(math.degrees(thetaMin))
            thetaMax = int(math.degrees(thetaMax))
            hullInterVal = [thetaMin,thetaMax]
            #hullInterVal = range(thetaMin, thetaMax)
            #V = U.copy()
            #print("I:",hullInterVal,U)
            #for u in U:

            #u = list(u)

            leftOver= overlap(U,hullInterVal)
            #print("L:",leftOver)

            if len(U) != len(leftOver):

            #Display Line of Sight for each BBC
                #displayLOS(hullmin, hullmax, BBCenter, sIdx, allStrokes)

                U = leftOver
                E.append([sIdx,tIdx])
                E.append([tIdx,sIdx])
                EdgeGraph[sIdx,tIdx] = 1
            #EdgeGraph[tIdx,sIdx] = 1

           # V.add(leftOver[1])

            
    #print(E)
    #print(EdgeGraph)
    return(EdgeGraph)

def getConvexHull(stroke):
    """
    Return convex hull of stroke
    """
    #print('len',len(stroke))
    '''
    try:
        if(len(stroke)<5):
            return stroke
        hull = ConvexHull(stroke)
        hullCor =[]
        for h in hull.vertices:
            hullCor.append(stroke[h])
    except Exception as e:
        return stroke
    '''
    return stroke


def overlap(U,minMax):
    shadow = []
    if 0<=minMax[0]<=90 and (minMax[1] - minMax[0] > 180):
        shadow = set(range(360))-set(range(minMax[0],minMax[1]))
    else:
        shadow = set(range(minMax[0], minMax[1]))
    leftover =   list(set(U) - shadow )
    #print('leftover',leftover)
    return  leftover
def displayLOS(hullmin,hullmax,BBCenter,sIdx, allStrokes):


    frame = np.zeros((200, int(2000)))

    dataArray = np.array([hullmin, BBCenter[sIdx]], dtype=int)
    # print(dataArray)
    cv2.polylines(frame, [dataArray], 0, (255), thickness=1)
    for s in allStrokes:
        dataArray = np.array(s, dtype=int)
        cv2.polylines(frame, [dataArray], 0, (255), thickness=1)

    cv2.imshow('testfile', frame)
    cv2.waitKey()

    dataArray = np.array([hullmax, BBCenter[sIdx]], dtype=int)
    # print(dataArray)
    cv2.polylines(frame, [dataArray], 0, (255), thickness=1)
    for s in allStrokes:
        dataArray = np.array(s, dtype=int)
        cv2.polylines(frame, [dataArray], 0, (255), thickness=1)

    cv2.imshow('testfile', frame)
    cv2.waitKey()
