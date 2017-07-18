'''
Parser Feature Extraction
Extract Feature for Parser Clasiifier

Run
ParserFeatureExtraction.py <Train .inkml Directory> <Train .lg Directory>

Input:
Train .inkml Directory - Directory where .inkml files are present for training purpose 
Train .lg Directory - Directory where .lg files are present for training purpose 

Output:
ParserFeatures - .csv file containing features extracted for Parser classifier 

Author
Ritvik Joshi
Rahul Dashora

'''
import glob
from xml.etree import cElementTree

import sys

import geometric as geo
import numpy as np
import LOS_v3
import PSC
import os


target=open('ParserFeatures.csv','w')

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

def read_files(Inkmldir,Lgdir):

    for curr_dir,subdir,files in os.walk(Inkmldir):
        count=0
        for filename in glob.glob(os.path.join(curr_dir, '*.inkml')):
                ParserPipeline(filename,Lgdir)
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

def ParserPipeline(fileinkml,dirlg):
    UID, Symbols, Strokes, GTgraph = read_inkml(fileinkml)
    print(UID)
    if len(Strokes)<3 or len(Symbols)<2:
        return
    SymbLabelStroke, LabelSymb, SymbRel = readRelations(dirlg+'\\'+UID+'.lg')
    #print(filelg)
    normalizedData, rangeX = normalizaion(Strokes)
    CombinedStrokes,SymbLabelList = getSymbStrokes(SymbLabelStroke,normalizedData)

    Losgraph = LOS_v3.getLOSGraph(CombinedStrokes)
    #print(Losgraph)
    labeledgraph=labelLOSgraph(SymbLabelList,Losgraph,SymbRel)

    pairs = pairGeneration(labeledgraph)
    Feature=featureExtraction(pairs,CombinedStrokes,labeledgraph)
    #print(Feature)
    Feature = np.asarray(Feature)
    Normalized_features = normalizeGeoMetirc(Feature, rangeX)
    #write_data(Normalized_features, UID)





def featureExtraction(pairs,CombinedStrokes,labeledgraph):
    Feature = []

    try:
        for pair in pairs:
            # print(pair)
            geo_features = geometric_features(CombinedStrokes, pair)
            shape_context = PSC.getAllPSC(CombinedStrokes, pair)
            final_feature = np.append(geo_features, shape_context)
            Label = getRelationId(labeledgraph[pair[0]][pair[1]])
            #print(Label)
            final_feature = np.append(final_feature, Label)
            Feature.append(final_feature)
    except Exception as e:
        print(e)

    return Feature

def getRelationId(relation):
    #print(relation)
    if(relation=='Right'):
        return 0
    elif(relation=='Above'):
        return 1
    elif (relation == 'Below'):
        return 2
    elif (relation == 'Sup'):
        return 3
    elif (relation == 'Sub'):
        return 4
    elif (relation == 'Inside'):
        return 5
    elif (relation == 'Undef'):
        return 6


def pairGeneration(labeledgraph):
    pair=[]
    labels=['Right','Above','Below','Undef','Sup','Sub','Inside']
    for iter in range(len(labeledgraph)):
        for jiter in range(len(labeledgraph)):
            if(labeledgraph[iter][jiter] in labels):
                pair.append((iter,jiter))
    return pair


def labelLOSgraph(SymbLabelList,Losgraph,SymbRel):
    labeledgraph = [['~' for _ in range(len(SymbLabelList)) ] for _ in range(len(SymbLabelList)) ]
    # for row in labeledgraph:
    #     print(row)
    for symb_index in range(len(SymbLabelList)):
        labeledgraph[symb_index][symb_index] = SymbLabelList[symb_index]


    for iter in range(len(SymbLabelList)):
        if SymbLabelList[iter] not in SymbRel:
            continue
        rel = SymbRel[SymbLabelList[iter]]
        for rel_tuples in rel:
            next_symb = rel_tuples[0]
            index = SymbLabelList.index(next_symb)
            labeledgraph[iter][index]=rel_tuples[1]

    for iter in range(len(Losgraph)):
        for jiter in range(len(Losgraph)):
            if(Losgraph[iter][jiter]==1 and labeledgraph[iter][jiter]=='~'):
                labeledgraph[iter][jiter]='Undef'

    # for row in labeledgraph:
    #     print(row)

    return labeledgraph

def getSymbStrokes(SymbLabelStroke,Strokes ):
    #print(Strokes)
    SymbLabelList = list(SymbLabelStroke.keys())
    AllStrokes=[]
    for symbLabel in SymbLabelList:
        strokeIdList = SymbLabelStroke[symbLabel]
        stroke=np.empty((0,2))
        for strokeId in strokeIdList:
            stroke=np.concatenate((stroke,Strokes[strokeId]),axis=0)

        AllStrokes.append(stroke)

    return AllStrokes,SymbLabelList


def main():

    if(len(sys.argv)<3):
        print('Please use the following way to run the program')
        print('FeatureExtraction.py <Train .inkml Directory>')
        print('Train .inkml Directory - Directory where .inkml files are present for training purpose ')
    else:
        Inkmlname = sys.argv[1]
        Lgname=sys.argv[2]
        read_files(Inkmlname,Lgname)
        # read_files('C:\Users/ritvi\PycharmProjects\PatternRecproject2\AdityaSplit/train',
        #            'C:\Users/ritvi\PycharmProjects\PatternRecParser\crohmelib/bin\GTfiles')


main()
