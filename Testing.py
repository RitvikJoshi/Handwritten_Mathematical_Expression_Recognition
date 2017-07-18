"""
Handwritten Mathematical Expression Recognition
Creates Symbol Layout Tree
Run
Testing.py <Test .inkml Directory> <Output Directory> <Symbol classifier pickle> <Segmentor classifier pickle> <Parser classifier pickle>

Input:
Test .inkml Directory - Directory where .inkml files are present for testing purpose 
Output Directory - Directory where output .lg files will be created
Symbol classifier pickle - Random forest pickle for symbol classification
Segmentation classifier pickle - Random forest pickle for Segmentation classification
Parser classifier pickle - Random forest pickle for Relationship classification

Output:
 .lg files - .lg files in the Output directory for all the .inkml file present in Test Directory

Author
Ritvik Joshi
Rahul Dashora
"""

import glob
import os
import pickle
import warnings
import ParserTest
import LOS_v2
import PSC
from xml.etree import cElementTree
import numpy as np
import SymbolClassifier
import geometric as geo
import sys
from sklearn.externals import joblib
import copy

if (len(sys.argv)<6):
    print('Please use the following way to run the program')
    print('Testing.py <Test .inkml Directory> <Output Directory> <Symbol classifier pickle> <Segmentor classifier pickle>')
    print('Test .inkml Directory - Directory where .inkml files are present for testing purpose')
    print('Output Directory - Directory where output .lg files will be created')
    print('Symbol classifier pickle - Random forest pickle for symbol classification')
    print('Segmentation classifier pickle - Random forest pickle for Segmentation classification')
    sys.exit(0)
else:
    testdir = sys.argv[1]
    outputdir = sys.argv[2]

    segmentpickle = sys.argv[4]
    symbolpickle = sys.argv[3]
    parserpickle=sys.argv[5]

    segmentfile = open(segmentpickle,'rb')
    rf = pickle.load(segmentfile)

    rfSymb = joblib.load(symbolpickle)

    Parserfile = open(parserpickle,'rb')
    rfParser = pickle.load(Parserfile)
    #outputdir = 'C:\\Users\\ritvi\\PycharmProjects\\PatternRecParser\\crohmelib\\bin\\AllParts'


# symbolfile = open('SymbolClassifier.p', 'rb')
# rfSymb = pickle.load(symbolfile )


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

        #Read UID of the file
        # files_data={}
        # UID=''
        # for UI in tree.findall("{http://www.w3.org/2003/InkML}annotation"):
        #      if(UI.items()[0][1]=='UI'):
        #          UID=UI.text
        temp_list = filename.split('\\')
        UID = temp_list[len(temp_list)-1].split(".")

        return UID[0],strokes
    except Exception as e:
        print('Exception:',e)
        print(filename)
    return None,None



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


def pairgeneration(LOS):
    SLT = []

    for iter in range(len(LOS)):
        for jiter in range(iter+1,len(LOS)):
            #if(LOS[iter][jiter]==1):
               SLT.append([iter,jiter])

    return SLT

def baseLinePair(LOS):
    SLT=[]

    for iter in range(0,len(LOS)-1):
        SLT.append([iter,iter+1])

    return SLT

def feature_extraction(strokes,SLT):
    Label=0
    Feature=[]
    #print("SLT::",SLT)
    try:
        for pair in SLT:
            #print(pair)
            geo_features=geometric_features(strokes,pair)
            shape_context=PSC.getAllPSC(strokes,pair)
     #       print('s:',shape_context)
            final_feature=np.append(geo_features,shape_context)
            Feature.append(final_feature)
    except Exception as e:
        print(e)
    #print(Feature)
    return np.asarray(Feature)


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
    #print('g:',geo_features)
    return geo_features


def normalizeGeoMetirc(features,rangeX):
   # print(features)

    for idx in range(13):
        features[:,idx] = np.divide(features[:,idx],rangeX)
    return  features




def PipeLine(filename):
    #Reading strokes from inkml files
    UID,Strokes =read_inkml(filename)
    Strokes_copy = copy.deepcopy(Strokes)
    print(UID)
    if len(Strokes)<2:
        RDF_test(UID,Strokes,True,None,None)
        return

    #Normalizing strokes
    normalizedData,rangeX = normalizaion(Strokes_copy)

    #print(UID)
    Losgraph=LOS_v2.getLOSGraph(normalizedData)
    #print(Losgraph)
    SLT=pairgeneration(Losgraph)
    #SLT=baseLinePair(Losgraph)
    #print(SLT)
    Features=feature_extraction(normalizedData,SLT)
    NormalizedFeatures = normalizeGeoMetirc(Features,rangeX)
    Symbols=RDF_test(UID,copy.deepcopy(Strokes),False,NormalizedFeatures,SLT)
    #print(SLT)
    #print(Symbols)
    ParserTest.ParserPipeline(UID,Symbols,Strokes,rfParser,outputdir)




def RDF_test(UID,Strokes,flag,data_array=None,SLT=None):
    classification_pairs=[]
    if flag==True:
        classification_pairs.append([0])
    else:
        #leaf_indices = rdtree.predict(data_array)
        leaf_indices = rf.predict(data_array)

        # for i in range(len(data_array)):
        #     print(SLT[i],leaf_indices[i])
        #
        labeledGraph = [['-' for _ in range(len(Strokes))]for _ in range(len(Strokes))]

        for i in range(len(data_array)):
            if(leaf_indices[i]==1):
                labeledGraph[SLT[i][0]][SLT[i][1]] = '*'
                labeledGraph[SLT[i][1]][SLT[i][0]] = '*'

        # for label in labeledGraph:
        #     print(label)


        visited={}
        classification_index=0
        for iter in range(len(labeledGraph)):
            if(iter not in visited):
                pair=[iter]
                visited[iter]=classification_index

                for jiter in range(iter+1,len(labeledGraph)):
                    if(labeledGraph[iter][jiter]=='*'):
                        pair.append(jiter)
                        visited[jiter]=classification_index
                classification_pairs.append(pair)
                classification_index+=1;
            else:
                cl_index=visited[iter]
                pair_list=classification_pairs[cl_index]
                for jiter in range(iter+1,len(labeledGraph)):
                    if(labeledGraph[iter][jiter]=='*' and jiter not in pair_list):
                        pair_list.append(jiter)
                        visited[jiter]=cl_index


    # print(classification_pairs)

    final_result={}

    for pairs in classification_pairs:
        classification_Strokes=[]
        for value in pairs:
            classification_Strokes.append(Strokes[value])

        pairFeatures = SymbolClassifier.Symbol_feature_extraction(classification_Strokes)
        #print(len(pairFeatures))
        symbol = rfSymb.predict(pairFeatures)
        #print(pairs,symbol)
        if(symbol[0] not in final_result):
            final_result[symbol[0]]=[pairs]
        else:
            final_result[symbol[0]].append(pairs)
    # print(final_result)
    #OR_fromat(UID,final_result)
    return final_result


def OR_fromat(filename,symbol):

    UID=filename
    print(UID)
    #target = open('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\crohmelib\\bin\\Adtest2\\'+UID+'.lg','w')
    #target = open('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\crohmelib\\bin\\baseLine\\'+UID+'.lg','w')
    target = open(outputdir+'\\'+UID+'.lg','w')
    num_object = len(symbol)


    target.write("# IUD, "+UID+'\n')
    target.write("# Objects("+str(num_object)+"):\n")
    weight=1.0
    for keys in symbol.keys():

        raw_label = keys.split('\\')
        if(len(raw_label)==2):
            label=raw_label[1]
        else:
            label=raw_label[0]
        label_counter=0
        for values in symbol.get(keys):
            final_label=label+'_'+chr(49+label_counter)
            label_counter+=1
            out_string='O,'+final_label+','+keys+','+str(weight)
            for strokes in values:
                out_string+=','+str(strokes)

            target.write(out_string+'\n')


    target.close()


def read_files(dir):

    for curr_dir,subdir,files in os.walk(dir):
        count=0
        for filename in glob.glob(os.path.join(curr_dir, '*.inkml')):
                PipeLine(filename)
                count+=1
        print(count)

#PipeLine('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\AdityaSplit\\train\\65_alfonso.inkml')

warnings.filterwarnings("ignore",category=DeprecationWarning)
#read_files('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\AdityaSplit\\test')
read_files(testdir)
