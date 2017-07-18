'''
Feature Extraction Program
Extract features for Symbol and Segmentor Classifier
Run
FeatureExtraction.py <Train .inkml Directory>

Input:
Train .inkml Directory - Directory where .inkml files are present for training purpose 

Output:
SegmentFeatures - .csv file containing features extracted for segmentation classifier 
SymbolFeatures - .csv file containing features extracted for Symbol classifier

Author
Ritvik Joshi
Rahul Dashora
'''

import glob
import os
import LOS_v2
import PSC

from xml.etree import cElementTree
import numpy as np

import SymbolClassifier
import geometric as geo
import sys


target=open('SegmentorFeatures.csv','w')
target2 = open('SymbolFeatures.csv','w')
def read_inkml(filename):
    """
    Read inkml file and generate class dict,stroke dict and expression graph
    :param filename: file path
    :param exp_dict: expression dict for the files
    :return:
    """

    try:
        tree = cElementTree.ElementTree(file=filename)

        #Read Strokes in the file
        strokes=[]

        for traces in tree.findall('{http://www.w3.org/2003/InkML}trace'):
            strokes_id = traces.items()[0][1]

            strokes_text = traces.text

            s_list = strokes_text.split(',');
            strokes_array = np.empty((len(s_list),2))
            for  index in range(len(s_list)):
                s_list[index] = s_list[index].strip()
                xy = s_list[index].split(' ')
                if(len(xy)==2):
                    strokes_array[index] = np.asarray(xy,dtype='float')
                else:
                    strokes_array[index] = np.asarray(xy[:2],dtype='float')
            strokes.append(strokes_array)

        #print(strokes)

        #Read Class and respective stroke_id
        classes={}

        for symbol in tree.findall("{http://www.w3.org/2003/InkML}traceGroup"):
            for sym in symbol.getchildren():
                sub=sym.tag[30:]
                #print(sub)
                if(sub=='annotation'):
                        #print('\t'+sym.text)
                    pass
                if(sub=='traceGroup'):
                    symb=''
                    sym_stroke=[]
                    for sym2 in sym.getchildren():
                        sub=sym2.tag[30:]
                        #print('\t'+sub)
                        if(sub=='annotation'):
                            #print('\t'*2+sym2.text)
                            symb=sym2.text
                            if(not classes.has_key(symb)):
                                classes[symb]=[]
                        if(sub=='traceView'):
                            #print('\t'*2+sym2.attrib['traceDataRef'])
                            stroke_id =int(sym2.attrib['traceDataRef'])
                            sym_stroke.append(stroke_id)
                    classes[symb].append(sym_stroke)

        #print(classes)

        #Read UID of the file
        files_data={}
        UID=''
        for UI in tree.findall("{http://www.w3.org/2003/InkML}annotation"):
             if(UI.items()[0][1]=='UI'):
                 files_data[UI.text]=[classes]
                 UID=UI.text

        labeledGraph = [['-' for _ in range(len(strokes))]for _ in range(len(strokes))]
        labeled_graph =createLabeledGraph(classes,labeledGraph)
        #print(labeled_graph)
        return UID,classes,strokes,labeled_graph
    except Exception as e:
        print('Exception:',e)
        print(filename)
    return None,None,None,None




def createLabeledGraph(symbols,labeledGraph):
    """
    Create Label graph for the math expression
    :param symbols: class dict with stroke information
    :param strokes_id: stroke id list
    :return:
    """

    #print(symbols)
    for symbol in symbols.keys():
        strokes_list = symbols[symbol]
        for strokes in strokes_list:
            for iter in range(len(strokes)):
                for jiter in range(iter,len(strokes)):
                    if(iter == jiter):
                        labeledGraph[strokes[iter]][strokes[jiter]] =symbol
                    else:
                        labeledGraph[strokes[iter]][strokes[jiter]] = '*'
                        labeledGraph[strokes[jiter]][strokes[iter]] = '*'
    # for row in labeledGraph:
    #  print(row)


    #relationship in edges
    for iter in range(len(labeledGraph)-1):
        for jiter in range(iter,len(labeledGraph)):
            if(labeledGraph[jiter][jiter] != labeledGraph[iter][iter] and labeledGraph[jiter][jiter]!='-'):
                next_symbol = labeledGraph[jiter][jiter]
                strokes = symbols[next_symbol]
                for stroke in strokes:
                    for stroke_id in stroke:
                        labeledGraph[iter][stroke_id] = 'R'

                break;
            elif(jiter-1>=0 and labeledGraph[jiter][jiter]==labeledGraph[iter][iter] and (labeledGraph[jiter][jiter-1]=='-')):
                labeledGraph[jiter][jiter-1] = 'R'
                labeledGraph[jiter-1][jiter] = 'R'

    # for row in labeledGraph:
    #      print(row)

    return labeledGraph



def normalizaion(strokePts):
    mat = strokePts
    maxY=0
    minY = 99999
    maxX = 0
    minX = 99999
    for m in mat:
        maxX = max(max(m[:,0]),maxX)
        minX = min(min(m[:,0]),minX)
        maxY = max(max(m[:, 1]), maxY)
        minY = min(min(m[:, 1]), minY)
   # print(maxX,minX,maxY,minY)

    rangeX = maxX - minX
    rangeY = maxY - minY

#   print(rangeX,rangeY)

    yFactor = 200/rangeY
    xFactor = 200/rangeY

    for m in mat:
        np.subtract(m[:, 0], minX,m[:,0])
        np.multiply(m[:,0],xFactor,m[:,0])
        #print(m[:,0])
        np.subtract(m[:, 1], minY,m[:,1])
        np.multiply(m[:,1],yFactor,m[:,1])

    return mat,int(rangeX*xFactor)




def pairgeneration(LOS,GT):
    SLT = []

    for iter in range(len(GT)):
        for jiter in range(iter+1,len(GT)):
            #if(LOS[iter][jiter]==1 and (GT[iter][jiter]=='*' or GT[iter][jiter]=='R')):
            if((GT[iter][jiter]=='*' or GT[iter][jiter]=='R')):
                SLT.append([iter,jiter])

    return SLT

def feature_extraction(strokes,GT,SLT):
    Label=0
    Feature=[]

    try:
        for pair in SLT:
            #print(pair)
            geo_features=geometric_features(strokes,pair)
            shape_context=PSC.getAllPSC(strokes,pair)
            final_feature=np.append(geo_features,shape_context)
            if(GT[pair[0]][pair[1]]=='*' ):
                Label = 1
            else:
                Label= 0
            final_feature=np.append(final_feature,Label)
            Feature.append(final_feature)
    except Exception as e:
        print(e)

    return Feature


def geometric_features(strokes,pair):
    #pair=[0,1]

    BM=geo.BackwardMovement(strokes[pair[0]],strokes[pair[1]])
    HO=geo.Horizontal_offset(strokes[pair[0]],strokes[pair[1]])
    DBC=geo.DistBBcenter(strokes[pair[0]],strokes[pair[1]])
    MD,BO =geo.MinDistance_BBOverlapping(strokes[pair[0]],strokes[pair[1]])
    DAC=geo.DistAverageCenter(strokes[pair[0]],strokes[pair[1]])
    MPD=geo.MaximalPairDist(strokes[pair[0]],strokes[pair[1]])
    VDBC=geo.VertDistBBCenter(strokes[pair[0]],strokes[pair[1]])
    DBp= geo.DistBeginpts(strokes[pair[0]],strokes[pair[1]])
    DEp=geo.DistEndpts(strokes[pair[0]],strokes[pair[1]])
    VDs,VDe=geo.Vert_offset(strokes[pair[0]],strokes[pair[1]])
    SD =geo.sizediff(strokes[pair[0]],strokes[pair[1]])
    Parallel=list(geo.Parallelity(strokes[pair[0]],strokes[pair[1]]))
    WS=geo.WritingSlope(strokes[pair[0]],strokes[pair[1]])
    #STA = geo.StrokeAngle(strokes[pair[0]],strokes[pair[1]])
    geo_features=[BM,HO,DBC,MD,BO,DAC,MPD,VDBC,DBp,DEp,VDs,WS,SD]+Parallel

    geo_features=np.asarray(geo_features)

    return geo_features


def normalizeGeoMetirc(features,rangeX):
    #print(features)
    for idx in range(13):
        #maxF = max(features[:,idx])
        #minF = min(features[:,idx])
        #NC  = maxF - minF
        #NC = max(NC,1)
        features[:,idx] = np.divide(features[:,idx],rangeX)
    return  features


def SymbClassfier(UID,Symbols,Strokes):

    for symb in Symbols.keys():
        stroke_ids=Symbols[symb]
        if symb==',':
            symb='COMMA'
        for str in stroke_ids:
            stroke_list=[]
            #print(symb,str)
            for st in str:
                stroke_list.append(Strokes[int(st)])
            #print(stroke_list)
            symb_features = SymbolClassifier.Symbol_feature_extraction(stroke_list)
            writeSymbFeature(UID,symb_features,symb)

def writeSymbFeature(UID,features,symb):
    count=0
    row=UID+","
    count+=1
    for data_idx in range(len(features)):
            row+=str(features[data_idx])+','

    row+=symb+'\n'

    target2.write(row)

def pipeline_function(filename):
    print(filename)

    UID,Symbols,Strokes,GTgraph=read_inkml(filename)
    print(UID)
    if Symbols!=None:
        SymbClassfier(UID,Symbols,Strokes)
    if UID==None or len(Strokes)<3:
        return
    normalizedData,rangeX =normalizaion(Strokes)
    #print(normalizedData)
    Losgraph=LOS_v2.getLOSGraph(normalizedData)

    SLT=pairgeneration(Losgraph,GTgraph)

    Feature=feature_extraction(normalizedData,GTgraph,SLT)

    Feature=np.asarray(Feature)
    Normalized_features = normalizeGeoMetirc(Feature,rangeX)
    write_data(Normalized_features,UID)


    #print(Losgraph)
    #RenderImage.getImage(normalizedData,rangeX)
    #print(exp_dict)

def write_data(features,UID):

    row=""
    count=0
    for feature in features:
        row=UID+","
        count+=1
        for data_idx in range(len(feature)):
            if(data_idx+1==len(feature)):
                row+=str(feature[data_idx])+'\n'
            else:
                row+=str(feature[data_idx])+','

        target.write(row)

    #print(count)


def read_files(dir):

    for curr_dir,subdir,files in os.walk(dir):
        count=0
        for filename in glob.glob(os.path.join(curr_dir, '*.inkml')):
                pipeline_function(filename)
                count+=1
        print(count)





def main():

    if(len(sys.argv)<2):
        print('Please use the following way to run the program')
        print('FeatureExtraction.py <Train .inkml Directory>')
        print('Train .inkml Directory - Directory where .inkml files are present for training purpose ')
    else:
        filename = sys.argv[1]
        read_files(filename)


main()

#pipeline_function('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\TrainINKML\\expressmatch\\101_fujita.inkml')