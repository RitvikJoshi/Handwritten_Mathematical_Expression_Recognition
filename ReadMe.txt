Read Me:

Authors:
Ritvik Joshi
Rahul Dashora

Runs on python version 3.4x64bit
Requirement- numpy,skitlearn,scipy,Opencv
 
Please follow the instruction to run the programs
***********************************************************************************************
For Feature Extraction
***********************************************************************************************
Run
FeatureExtraction.py <Train .inkml Directory>

Input:
Train .inkml Directory - Directory where .inkml files are present for training purpose 

Output:
SegmentFeatures - .csv file containing features extracted for segmentation classifier 
SymbolFeatures - .csv file containing features extracted for Symbol classifier

Run
ParserFeatureExtraction.py <Train .inkml Directory> <Train .lg Directory>

Input:
Train .inkml Directory - Directory where .inkml files are present for training purpose 
Train .lg Directory - Directory where .lg files are present for training purpose 

Output:
ParserFeatures - .csv file containing features extracted for Parser classifier 

***********************************************************************************************
For RandomForest Training
***********************************************************************************************
Run
Training.py <Feature.csv> <sym/seg/par indicator>

Input:
Feature.csv - Feature files to train random forest 
sym - To train for symbol classifier[Works only with symbol feature file (SymbolFeatures.csv)] 
seg - To train for segmentor classifier[Works only with segmentor feature file (SegmentFeatures.csv)]
par - To train for Parser classifier[Works only with Parser feature file (ParserFeatures.csv)]	
Output:
RandomForest.pickle - Returns trained random forest pickle file

***********************************************************************************************
For Testing
***********************************************************************************************
***********************************************************************************************
Ground Truth Parser (Does not require segementor and Symbol classifier)
***********************************************************************************************

Requirement - The test .inkml file should have stroke as well as symbol level information(which strokes are used to form which symbol).

Run
GTParser.py <Test .inkml Directory> <Output Directory> <Parser classifier pickle>

Input:
Test .inkml Directory - Directory where .inkml files with symbol stroke relationship are present for testing purpose
Output Directory - Directory where output .lg files will be created
Parser classifier pickle - Random forest pickle for Relationship classification

Output:
.lg files - .lg files in the Output directory for all the .inkml file present in Test Directory
***********************************************************************************************
Raw Stroke Parser (Require segementor and Symbol classifier pickle)
**********************************************************************************************

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



