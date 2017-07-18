"""
Ground Truth Parser
Performs Symbol Parsing and creates Symbol Layout tree
Requirement - The test .inkml file should have stroke as well as symbol level information(which strokes are used to form which symbol).
Run
GTParser.py <Test .inkml Directory> <Output Directory> <Parser classifier pickle>

Input:
Test .inkml Directory - Directory where .inkml files with symbol stroke relationship are present for testing purpose
Output Directory - Directory where output .lg files will be created
Parser classifier pickle - Random forest pickle for Relationship classification

Output:
.lg files - .lg files in the Output directory for all the .inkml file present in Test Directory

Authors:
Ritvik Joshi
Rahul Dashora
"""


import glob
from xml.etree import cElementTree

import sys

import geometric as geo
import numpy as np
import LOS_v3
import PSC
import os
import pickle
import copy


if (len(sys.argv)<4):
    print('Please use the following way to run the program')
    print('GTParser.py <Test .inkml Directory> <Output Directory> <Parser classifier pickle>')
    print('Test .inkml Directory - Directory where .inkml files with symbol stroke relationship are present for testing purpose')
    print('Output Directory - Directory where output .lg files will be created')
    print('Parser classifier pickle - Random forest pickle for Relationship classification')
    sys.exit(0)
else:
    testinkmldir = sys.argv[1]
    outputdir = sys.argv[2]

    parserpickle=sys.argv[3]
    Parserfile = open(parserpickle,'rb')
    rf = pickle.load(Parserfile)


#Parserfile = open('ParserClassifier_v27.p','rb')
# Parserfile = open('ParserClassifier_v64.p','rb')
# rf = pickle.load(Parserfile)
# outputdir = 'C:\\Users\\ritvi\\PycharmProjects\\PatternRecParser\\crohmelib\\bin\\AllParts'
#outputdir = 'C:\\Users\\ritvi\\PycharmProjects\\PatternRecParser'

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
                            if(symb not in classes):
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
        # for UI in tree.findall("{http://www.w3.org/2003/InkML}annotation"):
        #      if(UI.items()[0][1]=='UI'):
        #          files_data[UI.text]=[classes]
        #          UID=UI.text
        temp_list = filename.split('\\')
        UID = temp_list[len(temp_list) - 1].split(".")

        labeledGraph = [['-' for _ in range(len(strokes))]for _ in range(len(strokes))]
        labeled_graph =createLabeledGraph(classes,labeledGraph)
        #print(labeled_graph)
        return UID[0],classes,strokes,labeled_graph
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
    #print(geo_features)
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

def read_files(Inkmldir):

    for curr_dir,subdir,files in os.walk(Inkmldir):
        count=0
        for filename in glob.glob(os.path.join(curr_dir, '*.inkml')):
                ParserPipeline(filename)
                count+=1
        print(count)

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



'''
****************************************************************************************
New Code 
****************************************************************************************
'''

def readRelations(filename):
    target = open(filename,'r')
    SymbLabelStroke={}
    SymbRel={}
    LabelSymb={}
    for line in target:
        line=line.strip()
        line=line.split(',')

        if(line[0]=='O'):
            symb_label = line[1].split(' ')[1]
            symb = line[2].split(' ')[1]
            stroke_list=[]
            LabelSymb[symb_label]=symb
            for stroke_id in range(4,len(line)):
                stroke = line[stroke_id].split(' ')[1]
                stroke_list.append(int(stroke))
            SymbLabelStroke[symb_label]=stroke_list
        elif(line[0]=='EO'):
            curr_symLabel = line[1].split(' ')[1]
            next_symLabel = line[2].split(' ')[1]
            Relation = line[3].split(' ')[1]
            if(curr_symLabel not in SymbRel):
                SymbRel[curr_symLabel]=[(next_symLabel,Relation)]
            else:
                SymbRel[curr_symLabel].append((next_symLabel,Relation))

    #print(SymbLabelStroke)
    #print(LabelSymb)
    #print(SymbRel)

    return SymbLabelStroke, LabelSymb, SymbRel

def ParserPipeline(fileinkml):
#def ParserPipeline(UID,Symbols,Strokes,rfParser,outputdir):
    UID, Symbols, Strokes, GTgraph = read_inkml(fileinkml)
    print(UID)
    if len(Strokes)<3 or len(Symbols)<2:
        return
    #SymbLabelStroke, LabelSymb, SymbRel = readRelations(dirlg+'\\'+UID+'.lg')
    #print(filelg)
    normalizedData, rangeX = normalizaion(Strokes)

    CombinedStrokes,SymbLabelList,LabelToSymb,LabelToStroke = getSymbStrokes(Symbols,normalizedData)

    Losgraph = LOS_v3.getLOSGraph(CombinedStrokes)
    #print(Losgraph)
    pairs=pairGeneration(Losgraph)
    Feature = featureExtraction(pairs, CombinedStrokes)

    #writeSymbFeature(UID, Feature)
    #print(Feature)

    Feature = np.asarray(Feature)
    NormalizedFeatures = normalizeGeoMetirc(Feature, rangeX)


    '''
    call random forest to get relations
    '''
    RelationGraph,SLT=RDFTest(SymbLabelList, NormalizedFeatures, pairs)
    OR_fromat(UID, LabelToSymb, LabelToStroke, RelationGraph, SLT)


# def writeSymbFeature(UID,NormalizedFeatures):
#     target2=open('wrongfeature.csv','w')
#     count=0
#     for features in NormalizedFeatures:
#         row = UID + ","
#         for data_idx in range(len(features)):
#                 row+=str(features[data_idx])+','
#         row+='\n'
#
#     target2.write(row)


def RDFTest(SymbLabelList,NormalizedFeatures,pairs):

    Rellabeledgraph = [['~' for _ in range(len(SymbLabelList))] for _ in range(len(SymbLabelList))]
    #Weightedlabeledgraph=[[0.0 for _ in range(len(SymbLabelList))] for _ in range(len(SymbLabelList))]
    Weightedlabeledgraph =np.zeros((len(SymbLabelList),len(SymbLabelList)))
    probab = rf.predict_proba(NormalizedFeatures)
    allClasses = rf.classes_


    for index in range(len(NormalizedFeatures)):
        rowNum = pairs[index][0]
        colNum = pairs[index][1]

        pair_result = list(probab[index])
    #    print(pair_result)
        max_prob = max(pair_result)
        max_index = pair_result.index(max_prob)
        labelId = allClasses[max_index]
        labelName = getRelation(labelId)
        Rellabeledgraph[rowNum][colNum]=labelName
        Weightedlabeledgraph[rowNum][colNum]=max_prob


    for iter in range(len(SymbLabelList)):
        Rellabeledgraph[iter][iter]=SymbLabelList[iter]
        #Weightedlabeledgraph[iter][iter]=SymbLabelList[iter]


    # for row in Rellabeledgraph:
    #     print(row)
    # for row in Weightedlabeledgraph:
    #     print(row)

    SLT=SymbolLayoutTree(Rellabeledgraph,Weightedlabeledgraph)

    return Rellabeledgraph,SLT


def featureExtraction(pairs,CombinedStrokes):
    Feature = []

    try:
        for pair in pairs:
            # print(pair)
            geo_features = geometric_features(CombinedStrokes, pair)
            shape_context = PSC.getAllPSC(CombinedStrokes, pair)
            final_feature = np.append(geo_features, shape_context)
            #print(Label)
            Feature.append(final_feature)
    except Exception as e:
        print(e)

    return Feature

def getRelation(relation):
    #print(relation)
    if(relation==0):
        return 'Right'
    elif(relation==1):
        return 'Above'
    elif (relation == 2):
        return 'Below'
    elif (relation == 3):
        return 'Sup'
    elif (relation == 4):
        return 'Sub'
    elif (relation == 5):
        return 'Inside'
    elif (relation == 6):
        return 'Undef'


def pairGeneration(losgraph):
    pair=[]
    for iter in range(len(losgraph)):
        for jiter in range(len(losgraph)):
            if(losgraph[iter][jiter]== 1):
                pair.append((iter,jiter))
    return pair


def SymbolLayoutTree(Rellabeledgraph,Weightedlabeledgraph):
    TempSLT = copy.deepcopy(Weightedlabeledgraph)
    # for row in labeledgraph:
    #     print(row)

    for iter in range(len(Rellabeledgraph)):
        for jiter in range(len(Rellabeledgraph)):
            if(Rellabeledgraph[iter][jiter]=='Undef'):
                TempSLT[iter][jiter]=0.0
    SLTIndexList=[]
    for iter in range(TempSLT.shape[0]):
        colList =  list(TempSLT[:,iter])
    #    print(colList)
        max_index = colList.index(max(colList))
        SLTIndexList.append(max_index)
    #print(SLTIndexList)
    SLT = np.zeros(TempSLT.shape)
    for iter in range(len(SLTIndexList)):
        SLT[SLTIndexList[iter],iter] = 1.0

    # for row in SLT:
    #     print(row)

    return SLT

def getSymbStrokes(Symbols,Strokes ):
    #print(Strokes)
    SymbList = list(Symbols.keys())
    AllStrokes=[]
    SymbLabelList=[]
    LabelToSymb={}
    LabelToStroke={}
    for symb in SymbList:
        symbStrokeIdList = Symbols[symb]
        counter=1
        if symb==',':
            symb='COMMA'
        for strokeIdList in symbStrokeIdList:
            stroke=np.empty((0,2))

            for strokeId in strokeIdList:
                stroke=np.concatenate((stroke,Strokes[strokeId]),axis=0)
            currSymbLabel=symb+'_'+str(counter)
            SymbLabelList.append(currSymbLabel)
            LabelToSymb[currSymbLabel]=symb
            LabelToStroke[currSymbLabel]=strokeIdList
            counter+=1

            AllStrokes.append(stroke)

    return AllStrokes,SymbLabelList,LabelToSymb,LabelToStroke


def OR_fromat(filename,LabelToSymbol,LabelToStroke,RelationGraph,SLT):

    UID=filename
    print(UID)
    labels = ['Right', 'Above', 'Below', 'Sup', 'Sub', 'Inside']
    #target = open('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\crohmelib\\bin\\Adtest2\\'+UID+'.lg','w')
    #target = open('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\crohmelib\\bin\\baseLine\\'+UID+'.lg','w')
    #outputdir = 'C:\\Users\\ritvi\\PycharmProjects\\PatternRecParser'
    target = open(outputdir+'\\'+UID+'.lg','w')
    num_object = len(LabelToSymbol)


    target.write("# IUD, "+UID+'\n')
    target.write("# Objects("+str(num_object)+"):\n")
    weight=1.0
    for keys in LabelToSymbol.keys():

        raw_label = keys.split('\\')
        if(len(raw_label)==2):
            label=raw_label[1]
        else:
            label=raw_label[0]
        label_counter=0
        out_string='O,'+keys+','+LabelToSymbol[keys]+','+str(weight)
        for strokes in LabelToStroke[keys]:
            out_string+=','+str(strokes)

        target.write(out_string+'\n')

    target.write("\n# Relations from SRT:\n")

    for iter in range(len(RelationGraph)):
        for jiter in range(len(RelationGraph)):
            rel = 'EO, '
            if(SLT[iter][jiter]==1 and RelationGraph[iter][jiter] in labels):
                rel+=RelationGraph[iter][iter]+', '+RelationGraph[jiter][jiter]+', '+RelationGraph[iter][jiter]+', '+'1.0'
                target.write(rel+'\n')

    target.close()


#ParserPipeline('C:\\Users\\ritvi\\PycharmProjects\\PatternRecproject2\\AdityaSplit\\train\\65_alfonso.inkml')
read_files(testinkmldir)
