"""
Perform Symbol classification task 

Input 
Stroke Information
Output
Symbol Label

Author 
Ritvik Joshi
Rahul Dashora
"""
import numpy as np
import cv2
import MergeClassifier
def normalizeSymbol(AllData):
    # Converts all character into 100 * 100 symbol
    # Aspect Ratio has been preserved and the character has been maintained in the center for each sample to generate uniform features
    combinedData = []
    #print(AllData)
    for data in AllData:
        for idx in range(len(data)):
            d = data[idx]
            t1 = (float(d[0]))
            t2 = (float(d[1]))

            data[idx] = [t1,t2]
            combinedData.append([t1,t2])

    dataMat = np.matrix(combinedData)
    #print(dataMat)
    minX = min(dataMat[:, 0])[0, 0]
    maxX = max(dataMat[:, 0])
    #print(minX,maxX)
    rangeX = (maxX - minX)[0, 0]
    #print(rangeX)
    if rangeX == 0: rangeX =1
    minY = min(dataMat[:, 1])[0, 0]
    maxY = max(dataMat[:, 1])
    rangeY = (maxY - minY)[0, 0]
    if rangeY == 0: rangeY = 1
    offsetX = 0
    offsetY = 0

    maxBit = 100
    if rangeX > rangeY:
        offsetY =int( (maxBit/2)*(1 - ((rangeY/float(rangeX))))-1)

    elif rangeX < rangeY:
        offsetX = int((maxBit / 2) * (1 - ((rangeX / float(rangeY))))-1)

    divFactor =float( maxBit/max(rangeX,rangeY))

    for data in AllData:

        for idx in range(len(data)):
            '''
            data[idx][0] -= minX
            data[idx][0] *= divFactor#int((100) / divFactor)
            data[idx][0] += offsetX
            data[idx][0] = int(data[idx][0] )
            data[idx][1] -= minY
            data[idx][1] *= divFactor#int((100) / divFactor)
            data[idx][1] += offsetY
            data[idx][1] = int(data[idx][1])
            '''

            data[idx][0] = int((data[idx][0] - minX)*(divFactor)+offsetX)
            #temp = data[idx][1]
            data[idx][1] = int((data[idx][1] - minY) * (divFactor) + offsetY)
    #features = [minX,minY,maxX,maxY]
    aspectRatio = rangeY-rangeX/max((rangeY+rangeX),1)
    return  AllData,aspectRatio


def Symbol_feature_extraction(strokes):
    Feature = []

    # data, filename, aspectRatio = getSVG(path + '/' + file[0])
    data ,aspectRatio = normalizeSymbol(strokes)
    #data = strokes

    # print('normaized data',data)
    # print('Original Strokes',strokes)
    # aspectRatio = getASpectratio(data)
    first = data[0][0]

    last = data[-1][-1]

    dist = ((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2) ** (0.5)
    theta = (last[1] - first[1]) / max((last[0] - first[0]), 1)

    firstorgdist = ((first[0] - 50) ** 2 + (first[1] - 50) ** 2) ** (0.5)
    firstorgtheta = (first[1] - 50) / max((first[0] - 50), 1)
    lastorgdist = ((last[0] - 50) ** 2 + (last[1] - 50) ** 2) ** (0.5)
    lastorgtheta = (last[1] - 50) / max((last[0] - 50), 1)
    # theta = (last[1] - first[1]) / (last[0] - first[0])
    onlineFeat = [dist, theta, firstorgdist, firstorgtheta, lastorgdist, lastorgtheta]
    #print(len(strokes))
    resampledData = resampling(strokes)

    img = np.zeros((100, 100))

    for idx in range(len(data)):
        dataArray = np.array(data[idx],dtype='int32')
        #print(dataArray)
        cv2.polylines(img, [dataArray], 0, (255), thickness=5)

    rowHist = []
    colHist = []
    # rowHist  =np.zeros(20)
    # colHist = np.zeros(20)
    # countBin =0
    for rowIdx in range(0, 100, 5):
        temp = 0

        tempc = 0
        for i in range(5):
            temp += np.count_nonzero(img[rowIdx + i])
            # print(temp)
            tempc += np.count_nonzero(img[:, rowIdx + i])
        colHist.append(tempc)
        rowHist.append(temp)
        # countBin +=1
    # subImg = np.zeros((32,32))
    # subImg = cv2.resize(img,(32,32),interpolation = cv2.INTER_AREA)
    # cv2.imshow('chotu',subImg)
    # cv2.waitKey()
    histFeature = rowHist + colHist

    # hogFeature = HOG(subImg)
    # hogFeature = list(hogFeature)


    curlinessFeature = Curliness(data)

    firstDerivative = derivative(resampledData)
    secondDerive = secondDerivative(firstDerivative)

    Feature = [curlinessFeature] + firstDerivative + [aspectRatio] + histFeature + secondDerive + onlineFeat  # +hogFeature
    return Feature

def Curliness(data):
    #Curliness computes the curves of a symbol by calculating the length of the storke and the aspect ratio
    #more the length , more will be the curliness
    feature = []
    count = 0
    total =0
    for stroke in data:
        strokesSize = (len(stroke))
        count+=1
        #lenT = []
        for idx in range(0, strokesSize - 1):
            #lenT.append(((stroke[idx][0]-stroke[idx+1][0])**2 + (stroke[idx][1]-stroke[idx+1][1])**2)**(0.5))
            total+=((stroke[idx][0]-stroke[idx+1][0])**2 + (stroke[idx][1]-stroke[idx+1][1])**2)**(0.5)
        #feature.append(sum(lenT)/100)
        ans=total/(count*100)
    return ans

def derivative(normalizedData):
    #
    # Calculate derivative of elements of resampled data , this is calculated using weighted difference of the neibour if each
    # point which is normalized. This feature helps to handle speed variance while writting
    feature = []
    #print(len(normalizedData))
    picsize = 100
    normalizedData = np.asarray(normalizedData)
    for stroke in [normalizedData]:
        strokesSize = (len(stroke))
      #  print(strokesSize)
        for idx in range(2,strokesSize-2):
            '''
            xm2 = stroke[idx - 2][0]
            xm1 = stroke[idx-1][0]
            x = stroke[idx][0]
            x1 = stroke[idx+1][0]
            x2 = stroke[idx + 2][0]
            ym2 = stroke[idx - 2][1]
            ym1 = stroke[idx - 1][1]
            y = stroke[idx][1]
            y1 = stroke[idx + 1][1]
            y2 = stroke[idx + 2][1]
            '''
            x,y = stroke[idx][0],stroke[idx][1]
            xx,yy =  stroke[idx-2:idx+3,0],stroke[idx-2:idx+3,1]

            wts = [-2,-1,0,1,2]
            xx,yy = np.multiply(xx,wts),np.multiply(yy,wts)

            delta = (x**2 + y**2)**(0.5)
            if delta == 0:
                delta = 1
            '''
            xPrime = ((x1-xm1) + 2*(x2-xm2))/(5*delta)
            yPrime = ((y1 - ym1) + 2 * (y2 - ym2)) / (5 * delta)
            '''
            xPrime,yPrime = sum(xx)/(5*delta), sum(yy)/(5*delta)
            #temp.append([picsize+int(xPrime*picsize),picsize+int(yPrime*picsize)])
            feature.append(xPrime)
            feature.append(yPrime)
        #feature.append(temp)
    return feature

def resampling(data):
    # Create a new vector of coordinates that contains fixed number of samples from original vector
    # this allows to used some features as attributes
    numOfPts = 40
    newData = []
    stroke = []
    for s  in data:
        for k in s:
            stroke.append(k)
    #print(stroke)
    strokeSize = len(stroke)
    for idx in range(numOfPts):
        i = int((idx*strokeSize)/numOfPts)
        newData.append([stroke[i][0],stroke[i][1]])
    #newData.append(newStroke)
      #  print(newStroke)
    return newData
def secondDerivative(firstDerivative):
    feature = []
    strokesSize = (len(firstDerivative))

    # print(len(normalizedData))
    picsize = 100
    x=[]
    y = []
    for idx in range(0,strokesSize,2):
        x.append(firstDerivative[idx])
        y.append(firstDerivative[idx+1])
    stroke = np.asarray(firstDerivative)

    strokesSize = (len(x))

    for idx in range(2, strokesSize - 2):

        x1, y1 = x[idx], y[idx]
        xx, yy = x[idx - 2:idx + 3], y[idx - 2:idx + 3]

        wts = [-2, -1, 0, 1, 2]
        xx, yy = np.multiply(xx, wts), np.multiply(yy, wts)

        delta = (x1 ** 2 + y1 ** 2) ** (0.5)
        if delta == 0:
            delta = 1
        '''
        xPrime = ((x1-xm1) + 2*(x2-xm2))/(5*delta)
        yPrime = ((y1 - ym1) + 2 * (y2 - ym2)) / (5 * delta)
        '''
        xPrime, yPrime = sum(xx) / (5 * delta), sum(yy) / (5 * delta)
        # temp.append([picsize+int(xPrime*picsize),picsize+int(yPrime*picsize)])
        feature.append(xPrime)
        feature.append(yPrime)
        # feature.append(temp)
    return feature

