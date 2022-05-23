#--------------------------------------------------------------------------
#                    accuracy_assess_v2021-7.py
#--------------------------------------------------------------------------
# Assess accuracy by generating a full "map" for each fold.
# This version tests evaluating the stratified accuracies using the full set of predictions from the cross validation after running all the loops

from __init__ import *
import pandas as pd
import numpy as np
import os, glob, pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
#from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from Modules import accuracy_and_sampling_lib2 as accuracySamplingLib
import pickle, ee
ee.Initialize()
from Modules.LCMSVariables import *

#------------------------------------------------------------------------------
#                                   Options
#------------------------------------------------------------------------------
studyArea = 'COASTAL_AK'
runname = 'Final_Delivery_Prep' # directory name to organize saved figures and pickle file.
mapName = 'Land_Cover' # 'Change', 'Land_Cover', 'Land_Use'

modelingDict = paramDict[studyArea]['Modeling']
trainingDict = paramDict[studyArea]['Training_Data']

# save to pickle or load from pickle - so that the run results can be saved and accessed more easily
# Step 1: Generate dictionary of KFoldInfo for each fold
save_or_load_KFoldInfo = 'save'

k = 10 # Folds for Group K-Fold 
num_iterations = 1
n_jobs = 8

# Change Options
if mapName == 'Change':
    modelToRun = modelingDict['changeClasses']
    tuningFileName = modelingDict['changeTuningFileName']
applyTreeMask = False # For Change Only
applyForestMask = False # For Change Only
buffer = True # For Change Only - apply a one year buffer to loss and gain

# Land Cover Options
clump_landcover_classes = False # For Land Cover only
if mapName == 'Land_Cover':
    modelToRun = ['DOM_SEC_LC']
    tuningFileName = modelingDict['landcoverTuningFileName']
    
# Land Use Options
if mapName == 'Land_Use':
    modelToRun = ['DOM_LU']
    tuningFileName = modelingDict['landuseTuningFileName']
apply_landuse_rules = False # For Land Use only, True for CONUS and False for AK
dev_years = 1 # For Land Use only, apply_landuse_rules only
urban_mask = True # For Land Use only, apply_landuse_rules only
# Use LandUse_Rules_Exporter.js to make this file:
urban_mask_file = r'Z:\Production\00_CONUS\06_CONUS_2021-7\03_Assessment\Accuracies\Urban_Mask.csv'


#------------------------------------------------------------------------------
#                    Paths & Models to Run
#------------------------------------------------------------------------------
# These are the classes that will be printed in the confusion matrix etc.
# The classes that are actually modeled are listed in the LCMSVariables modelingDict
if mapName == 'Land_Cover' and clump_landcover_classes:
    classDict = {
        'Land_Cover': ['TREES','SHRUBS','GRASS','BARREN-IMP','SNOW','WATER']}
else:
    classDict = {
        'Change': ['Stable','Slow_Loss','Fast_Loss','Gain'],
        'Land_Cover': ['TREES','TS-TREES','SHRUBS-TRE','GRASS-TREE','BARREN-TRE','TS','SHRUBS','GRASS-SHRU','BARREN-SHR','GRASS','BARREN-GRA','BARREN-IMP','SNOW','WATER'],
        'Land_Use': ["Agriculture", "Developed", "Forest", "Non-forest Wetland", "Other", "Rangeland"]}



strataFile = modelingDict['timesyncStrataFile']
strataDict = modelingDict['strataDict']
traindir = trainingDict['localTrainingDataPath'] # root directory for the paths below
summaryDir = os.path.join(traindir,'TuningParameters','Summaries') # where to find the Tuning_Summary that contains run parameters 
accuracyDir = os.path.join(paramDict[studyArea]['Assessment']['localAssessmentPath'],'Accuracies') # root for where to save final accuracy files
savedir = os.path.join(accuracyDir, runname) # actual save directory
picklePath = os.path.join(savedir, 'Pickle_Files') 
indir =  modelingDict['trainingTableFolder'] # yearly exported tables with base-learner & composite values extracted for each timesync plot
plotColumn = 'PLOTID'
stratPlotColumn = modelingDict['strataFileIDColumn'] # name of plot id column in strataFile
stratColumn = modelingDict['strataFileStratColumn'] #  name of the strata column in strataFile

if mapName == 'Land_Use':
    remapDict = {"Agriculture": 1, "Developed": 2, "Forest": 3, "Non-forest Wetland": 4, "Other": 5, "Rangeland": 6}
elif mapName == 'Land_Cover':# and studyArea == 'COASTAL_AK':
    remapDict = {'TREES': 1, 
                'TS-TREES': 2,
                'SHRUBS-TRE': 3,
                'GRASS-TREE': 4,
                'BARREN-TRE': 5,
                'TS': 6,
                'SHRUBS': 7, 
                'GRASS-SHRU': 8,
                'BARREN-SHR': 9, 
                'GRASS': 10,
                'BARREN-GRA': 11,
                'BARREN-IMP': 12,
                'SNOW': 13,
                'WATER': 14}
# elif mapName == 'Land_Cover':
#     remapDict = {'TREES': 1, 
#                 'SHRUBS-TRE': 3,
#                 'GRASS-TREE': 4,
#                 'BARREN-TRE': 5,
#                 'SHRUBS': 7, 
#                 'GRASS-SHRU': 8,
#                 'BARREN-SHR': 9, 
#                 'GRASS': 10,
#                 'BARREN-GRA': 11,
#                 'BARREN-IMP': 12,
#                 'SNOW': 13,
#                 'WATER': 14}
    clumpDict_primary = {'TREES': 'TREES', 
                        'SHRUBS-TRE': 'SHRUBS',
                        'GRASS-TREE': 'GRASS',
                        'BARREN-TRE': 'BARREN-IMP',
                        'SHRUBS': 'SHRUBS', 
                        'GRASS-SHRU': 'GRASS',
                        'BARREN-SHR': 'BARREN-IMP', 
                        'GRASS': 'GRASS',
                        'BARREN-GRA': 'BARREN-IMP',
                        'BARREN-IMP': 'BARREN-IMP',
                        'SNOW': 'SNOW',
                        'WATER': 'WATER'}
    clumpDict_secondary = {'TREES': 'TREES', 
                        'SHRUBS-TRE': 'TREES',
                        'GRASS-TREE': 'TREES',
                        'BARREN-TRE': 'TREES',
                        'SHRUBS': 'SHRUBS', 
                        'GRASS-SHRU': 'SHRUBS',
                        'BARREN-SHR': 'SHRUBS', 
                        'GRASS': 'GRASS',
                        'BARREN-GRA': 'GRASS',
                        'BARREN-IMP': 'BARREN-IMP',
                        'SNOW': 'SNOW',
                        'WATER': 'WATER'}
elif mapName == 'Change':
    remapDict = {'Stable': 1,
                'DND_Slow': 2,
                'DND_Fast': 3,
                'RNR': 4,
                'Slow_Loss': 2,
                'Fast_Loss': 3,
                'Gain': 4}

remapDictFlipped = {str(y):x for x,y in remapDict.items()}
#------------------------------------------------------------------------------
#           Do initial set up
#------------------------------------------------------------------------------
if not os.path.isdir(accuracyDir):
    os.mkdir(accuracyDir)
if not os.path.isdir(savedir):
    os.mkdir(savedir)
if not os.path.isdir(picklePath):
    os.mkdir(picklePath)
print('Run Name: ', os.path.basename(savedir))

outputName = '_'.join([mapName, runname, studyArea])
pickleName = '_'.join([mapName, studyArea])
accuracyFile = os.path.join(savedir, 'kfold_accuracies_'+outputName+'.txt')
if os.path.isfile(accuracyFile):
    os.remove(accuracyFile)
if mapName == 'Land_Cover' and clump_landcover_classes:
    finalAccuracyFilePath = os.path.join(savedir, '_'.join(['final_stats_clump', outputName])+'.txt')
elif mapName == 'Land_Use' and apply_landuse_rules:
    finalAccuracyFilePath = os.path.join(savedir, '_'.join(['final_stats_LU_Rules', outputName])+'.txt')
else:
    finalAccuracyFilePath = os.path.join(savedir, '_'.join(['final_stats', outputName])+'.txt')
if os.path.isfile(finalAccuracyFilePath):
    os.remove(finalAccuracyFilePath)

if save_or_load_KFoldInfo == 'save':

    # TimeSync training data format: annualized or vertex (raw)
    TS_dataFormat = 'annualized'

    print('Reading in Training Data')
    # Read in data table
    tabledir = glob.glob(os.path.join(indir,'*.csv'))
    allTrainingData = pd.concat((pd.read_csv(f, index_col=None, header=0) for f in tabledir), axis=0, ignore_index=True)

    # Drop any plots with NaNs
    allNames = allTrainingData.columns
    indNames = [name for name in allNames if (('_LT_' in name) or ('_VT_' in name) or ('_Comp_' in name) or ('_CCDC_' in name) or ('terrain' in name))]
    allTrainingData.dropna(axis=0, subset=indNames, inplace=True)
    
    print('Reading in Strata File')
    strata = pd.read_csv(strataFile)

    # Make sure all Strata Plots are actually in Training Data
    strata = strata[strata[stratPlotColumn].isin(allTrainingData['PLOTID'])]

    if studyArea == 'COASTAL_AK':
        allTrainingData = allTrainingData.rename(columns = {'DND_Slow': 'Slow_Loss', 'DND_Fast': 'Fast_Loss', 'RNR': 'Gain'})
####################################################################################
#                       Run Models and Save Probabilities:
###################################################################################

#-----------------------------------------------------------------------------
#                       Import Run Parameters
#-----------------------------------------------------------------------------
if save_or_load_KFoldInfo == 'save':
    
    indDict = {}
    thresholdDict = {}
    for key in modelToRun:

        # "keyName" is the used to find the correct Tuning Summary file only
        indDict[key] = {}
        if key == 'LU':
            keyName = 'DOM_LU' 
        elif key == 'LC':
            keyName = 'DOM_LC'
        else:
            keyName = key
        
        # Summary File: load variables, mtry, and threshold from .csv files
        summaryFile = os.path.join(summaryDir, '_'.join(['Tuning_Summary', studyArea, keyName, tuningFileName]))

        with open(summaryFile) as f: # this assumes only one line in file - will default to the last line if that isn't true
            next(f)
            for line in f:

                if len(line) > 1:
                    variables = line.split(',')[9:]
                    indDict[key]['variables'] = [v.replace('"','').replace('\n','') for v in variables]                
                    indDict[key]['accuracy'] = float(line.split(',')[5].replace('"',''))
                    indDict[key]['kappa'] = float(line.split(',')[8].replace('"',''))                
                    indDict[key]['run_description'] = line.split(',')[0].replace('"','')
                    indDict[key]['nTrees'] = float(line.split(',')[2].replace('"',''))
                    indDict[key]['min_samples_leaf']=  int(line.split(',')[3])

                    # We are only using the threshold for Change, not Land Cover and Land Use
                    if mapName == 'Change': 
                        thresholdDict[key] = float(line.split(',')[4].replace('"',''))
                    
                    indDict[key]['mtry'] = int(line.split(',')[1].replace('"',''))

    #-----------------------------------------------------------------------------
    #     Get Strata for each plot
    #-----------------------------------------------------------------------------
    # Get Strata:
    if not modelingDict['strataFileStratColumn'] in allTrainingData.columns:
        strataTable = pd.read_csv(strataFile)
        if modelingDict['strataFileIDColumn'] != 'PLOTID':
            strataTable['PLOTID'] = strataTable[modelingDict['strataFileIDColumn']]
        allTrainingData = allTrainingData.merge(strataTable[['PLOTID', modelingDict['strataFileStratColumn']]], how='inner', on='PLOTID', copy=False)
    strata = allTrainingData[stratColumn].squeeze()
    groups = allTrainingData['PLOTID'].squeeze()

    #-----------------------------------------------------------------------------
    #     Prep Dependent and Independent data for each model so it doesn't have to be redone each fold.
    #-----------------------------------------------------------------------------
    for model in modelToRun:
        print()
        print(model)                
        print()

        # Prep Training Data Columns
        modelColumn = model
        if model == 'BARREN-IMP':
            allTrainingData[modelColumn] = allTrainingData['BARREN'].add(allTrainingData['IMPERVIOUS'])

        elif mapName == 'Land_Use' and model != 'DOM_LU':
            allTrainingData[modelColumn] = allTrainingData['DOM_LU'].eq(model).astype(int)
        
        # This is not needed for AK but it is needed for CONUS...not sure why. 
        elif mapName == 'Land_Cover' and (model == 'DOM_SEC_LC' or clump_landcover_classes) and studyArea == 'CONUS':
            landcoverClasses = modelingDict['landcoverClasses']
            allTrainingData['BARREN-IMP'] = allTrainingData['BARREN'].add(allTrainingData['IMPERVIOUS'])
            dom_sec_lc_index = np.argmax(allTrainingData[landcoverClasses].to_numpy(), axis = 1)
            allTrainingData['DOM_SEC_LC'] = [landcoverClasses[i] for i in dom_sec_lc_index] 
        # For AK, get rid of grass-ts and barren-ts class, which are too small and are not in our final product.
        allTrainingData['DOM_SEC_LC'] = allTrainingData['DOM_SEC_LC'].replace({'GRASS-TS': 'GRASS', 'BARREN-TS': 'BARREN-IMP'})

        # Get Dependent and Independent Variables for Training            
        inddata = allTrainingData[indDict[model]['variables']]
        depdata = allTrainingData[modelColumn]
        X, y = inddata, depdata.squeeze()
        indDict[model]['X'] = X
        indDict[model]['y'] = y

    #-----------------------------------------------------------------------------
    #     Run all Iterations and Folds of the Model
    #-----------------------------------------------------------------------------
    for iter in range(num_iterations):    
        KFoldInfo = {}
        kfoldinfo_pickle_filename = pickleName+'_Iteration_'+str(iter)+'.p'
        KFoldInfo['TrainingData'] = allTrainingData.copy()

        print()
        print('Iteration '+str(iter))

        # Find Indices for KFolds
        # K-fold iterator variant with non-overlapping groups.
        # The same group will not appear in two different folds (the number of distinct groups has to be at least equal to the number of folds).
        # The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold. 
        gkf = GroupKFold(n_splits=k)
        foldNum = 1
        for train_index, test_index in gkf.split(allTrainingData, allTrainingData, groups):
            KFoldInfo[str(foldNum)] = {}
            print()
            print('Fold Number: '+str(foldNum))
            print()

            # Indices of training and test samples
            KFoldInfo[str(foldNum)]['Indices'] = {\
                'Train': train_index,
                'Test': test_index}

            # Strata of training and test samples
            gk_strata_train, gk_strata_test = strata.iloc[train_index], strata.iloc[test_index]
            KFoldInfo[str(foldNum)]['Strata'] = {\
                'Train': gk_strata_train,
                'Test': gk_strata_test}

            # Run model and predict probabilities
            KFoldInfo[str(foldNum)]['Probabilities'] = {}
            KFoldInfo[str(foldNum)]['Predictions'] = {}
            KFoldInfo[str(foldNum)]['Model'] = {}
            for model in modelToRun:
                print()
                print(model)                
                print()

                # Get X and Y points for each group  
                gkx_train, gkx_test = indDict[model]['X'].iloc[train_index], indDict[model]['X'].iloc[test_index]
                gky_train, gky_test = indDict[model]['y'].iloc[train_index], indDict[model]['y'].iloc[test_index]
                
                # Fit and Train model
                classifier_gkfold = RandomForestClassifier(n_estimators = int(indDict[model]['nTrees']), max_features = int(indDict[model]['mtry']), 
                    min_samples_leaf = int(indDict[model]['min_samples_leaf']), n_jobs = n_jobs, oob_score = True, verbose = 1)
                classifier_gkfold.fit(gkx_train, gky_train)

                # Get Predicted Probabilities for each Test Point
                if model in ['DOM_SEC_LC','DOM_LC','DOM_LU']:
                    gky_pred = classifier_gkfold.predict(gkx_test)
                    KFoldInfo[str(foldNum)]['Predictions'][model] = gky_pred
                else:
                    gky_proba = classifier_gkfold.predict_proba(gkx_test)
                    KFoldInfo[str(foldNum)]['Probabilities'][model] = gky_proba

                # Save results                
                KFoldInfo[str(foldNum)]['Model'][model] = classifier_gkfold

            foldNum = foldNum + 1

        # Save Change Thresholds
        if mapName == 'Change':
            KFoldInfo['ThresholdDict'] = thresholdDict

        # Save Model Info and Output Probabilities to Pickle (This is most important for CONUS, which takes forever to run)
        pickle.dump(KFoldInfo, open(os.path.join(picklePath, kfoldinfo_pickle_filename), 'wb'))
        KFoldInfo = None

####################################################################################
#                   Assemble "Final Maps" for Each Fold and Assess Accuracies
###################################################################################
# Function for Tree Mask and Forest Mask
def count_consecutive_ones(s):
    v = np.diff(np.r_[0, s.values.flatten(), 0])
    s = pd.value_counts(np.where(v == -1)[0] - np.where(v == 1)[0])
    s.index.name = "num_consecutive_ones"
    s.name = "count"
    return s

if mapName == 'Land_Use' and apply_landuse_rules:
    urbanFrame = pd.read_csv(urban_mask_file)
    urbanMasks = urbanFrame.groupby('PLOTID').first()

print()
print('Assembling "Final Maps"')
with open(accuracyFile, 'a+') as accFile:  

    accFile.write('\n')
    accFile.write('#------------------------------\n')
    accFile.write(outputName+'\n')
    accFile.write('#------------------------------\n')
    accFile.write('\n')

    iterAccuracy = []
    iterBalancedAccuracy = []
    iterKappa = []
    iterF1Score = []
    iterUsers = []
    iterProducers = []
    for iter in range(num_iterations):    
        print('Iteration '+str(iter))
        accFile.write('#------------Iteration '+str(iter)+'-------------\n')
        accFile.write('\n')

        kfoldinfo_pickle_filename = pickleName+'_Iteration_'+str(iter)+'.p'
        print('Loading Pickle File: ', kfoldinfo_pickle_filename)
        KFoldInfo = pickle.load(open(os.path.join(picklePath, kfoldinfo_pickle_filename), 'rb'))
        allTrainingData = KFoldInfo['TrainingData']

        #---------------------------------------------------------------------
        #                 Prep. Reference Data
        #---------------------------------------------------------------------


        if mapName == 'Land_Use':
            referenceColumn = 'DOM_LU'
            
        elif mapName == 'Land_Cover':
            # # These lines only if aggregating barren-imp after modeling
            # allTrainingData['BARREN-IMP'] = allTrainingData['BARREN'].add(allTrainingData['IMPERVIOUS'])
            # modelToRun = [model for model in modelToRun if model not in ['BARREN', 'IMPERVIOUS']]
            # modelToRun.append('BARREN-IMP')
                          
            if clump_landcover_classes:
                allTrainingData['DOM_LC_primary'] = allTrainingData['DOM_SEC_LC'].replace(clumpDict_primary)
                allTrainingData['DOM_LC_secondary'] = allTrainingData['DOM_SEC_LC'].replace(clumpDict_secondary)
                referenceColumn = 'DOM_LC_primary'
            else:
                referenceColumn = 'DOM_SEC_LC'   


        elif mapName == 'Change':
            
            trainingModels = ['Stable'] + modelToRun
            allTrainingData['Stable'] = np.where(np.amax(allTrainingData[modelToRun], axis = 1) == 0, 1, 0)
            change_index = np.argmax(allTrainingData[trainingModels].to_numpy(), axis = 1)
            allTrainingData['CHANGE'] = [trainingModels[i] for i in change_index]
            referenceColumn = 'CHANGE'

            if applyTreeMask:
                treeNoTree = allTrainingData[['TREES','SHRUBS-TRE','GRASS-TREE','BARREN-TRE','PLOTID']].set_index('PLOTID').sum(axis = 1).gt(0).astype(int)
                counts = treeNoTree.groupby('PLOTID').apply(count_consecutive_ones).reset_index()
                idsToInclude = counts[counts['num_consecutive_ones'] >= 3].PLOTID.unique() # this is used later
            if applyForestMask:
                treeNoTree = allTrainingData[['DOM_LU','PLOTID']].set_index('PLOTID').eq('Forest').astype(int)
                counts = treeNoTree.groupby('PLOTID').apply(count_consecutive_ones).reset_index()
                idsToInclude = counts[counts['num_consecutive_ones'] >= 3].PLOTID.unique() # this is used later

        value_counts = allTrainingData[referenceColumn].value_counts()

        if modelToRun in [['DOM_SEC_LC'],['DOM_LC'],['DOM_LU']]:
            emptyClasses = [className for className in classDict[mapName] if className not in list(value_counts.keys())]
        else:
            emptyClasses = [className for className in classDict[mapName] if className not in modelToRun]
        if 'Stable' in emptyClasses:
            emptyClasses.remove('Stable')
        if len(emptyClasses) > 0:
            for className in emptyClasses:
                allTrainingData[className] = 0

        #---------------------------------------------------------------------
        #                   Assemble Map
        #---------------------------------------------------------------------
        
        probabilities = {}
        predictions = {}
        for key in modelToRun:
            probabilities[key] = []
            predictions[key] = []
        test_index = []
        gk_strata_test = []
        for foldNum in range(1, k+1):
            accFile.write('# KFold '+str(foldNum)+': \n')

            for model in modelToRun:
                if model in ['DOM_SEC_LC','DOM_LC','DOM_LU']:
                    predictions[model].append(KFoldInfo[str(foldNum)]['Predictions'][model])
                else:
                    probabilities[model].append(KFoldInfo[str(foldNum)]['Probabilities'][model])
            test_index.append(KFoldInfo[str(foldNum)]['Indices']['Test'])
            gk_strata_test.append(KFoldInfo[str(foldNum)]['Strata']['Test'])

        test_index = np.concatenate(test_index)
        gk_strata_test = np.concatenate(gk_strata_test)
        for key in modelToRun:
            if key in ['DOM_SEC_LC','DOM_LC','DOM_LU']:
                predictions[key] = np.concatenate(predictions[key])
            else:
                probabilities[key] = np.concatenate(probabilities[key])   
        
        if modelToRun in [['DOM_SEC_LC'],['DOM_LC'],['DOM_LU']]:
            classNames = list(value_counts.keys())
        else:
            classNames = modelToRun.copy()
        classNums = range(0, len(classNames))

        # Assemble Map Here
        probArray = []
        if mapName == 'Change':
            for className in modelToRun:
                # Apply Threshold
                thisProbArray = probabilities[className][:,1]
                belowThresh = thisProbArray < KFoldInfo['ThresholdDict'][className]
                thisProbArray[belowThresh] = 0
                probArray.append(thisProbArray)
            stableMask =  np.where(np.amax(probArray, axis = 0) == 0, 1, 0)
            classNames.insert(0,'Stable')
            probArray.insert(0, stableMask)
            gky_pred_index = np.argmax(probArray, axis = 0)
        else:
            if not (modelToRun in [['DOM_SEC_LC'],['DOM_LC'],['DOM_LU']]):               
                for className in classNames:
                    probArray.append(probabilities[className][:,1])
                gky_pred_index = np.argmax(probArray, axis = 0)

        if modelToRun in [['DOM_SEC_LC'],['DOM_LC'],['DOM_LU']]:  
            gky_pred_names = predictions[modelToRun[0]]            
        else:   
            gky_pred_names = [classNames[i] for i in gky_pred_index]
        gky_test_names =  allTrainingData[referenceColumn].iloc[test_index]

        #---------------------------------------------------------------------
        # Apply Special Post-Processing - Land-Use Rules and Clump Land Cover
        #---------------------------------------------------------------------
        if mapName == 'Land_Use' and apply_landuse_rules:
            predFrame = pd.DataFrame(columns = {'PredName', 'PLOTID', 'YEAR', 'Developed'})
            predFrame['PredName'] = gky_pred_names
            predFrame['PLOTID'] = allTrainingData['PLOTID'].iloc[test_index].to_numpy()
            predFrame['YEAR'] = allTrainingData['YEAR'].iloc[test_index].to_numpy()
            predFrame['UNIQUEID'] = predFrame['PLOTID'].astype(str)+'_'+predFrame['YEAR'].astype(str)
            predFrame['Developed'] = (predFrame['PredName'] == 'Developed').astype('int')
            predFrame['Forest'] = (predFrame['PredName'] == 'Forest').astype('int')
            predFrame['DOM_LU'] = allTrainingData['DOM_LU'].iloc[test_index].to_numpy()
            # if urban_mask:
            #     pdb.set_trace()
            #     predFrame['Urban_Mask'] = urbanMasks.loc[predFrame['Urban_Mask'], 'Urban_Mask']
            #     #predFrame = predFrame.merge(urbanFrame[['PLOTID', 'Urban_Mask']], how='inner', on='PLOTID', copy=False)
            #     predFrame['Urban_Mask'] = predFrame['Urban_Mask'].astype('bool')

            plotidValues = predFrame['PLOTID'].unique()

            stack = []
            for plotid in plotidValues:
                plotFrame = predFrame[predFrame['PLOTID'] == plotid].sort_values('YEAR')
                plotFrame['Urban_Mask'] = urbanMasks.loc[plotid, 'Urban_Mask'].astype('bool')

                # Get Developed Segments
                leftArray = np.insert(plotFrame['Developed'].to_numpy(), 0, 0)
                rightArray = np.append(plotFrame['Developed'].to_numpy(), [0])
                diff = np.subtract(rightArray, leftArray)
                if plotFrame['Developed'].sum() > 0: # and plotFrame['Developed'].sum() < 20 and plotFrame['Forest'].iloc[-4:-1].sum() > 0:
                    # Find length of developed segment
                    endVals = np.where(diff == -1)
                    startVals = np.where(diff == 1)
                    segLengths = np.subtract(endVals, startVals)

                    # If segment is long enough, change subsequent forest pixels to developed
                    if np.max(segLengths) >= dev_years:
                        # Get end year of oldest developed segment
                        endYears = np.add(endVals, plotFrame['YEAR'].min())
                        endYears = endYears[np.where(segLengths >= dev_years)]
                        endYear = np.min(endYears)
                        plotFrame['Dev_Mask'] = plotFrame['YEAR'].ge(endYear)
                        # Replace forest values
                        if urban_mask:
                            plotFrame.loc[(plotFrame['PredName'] == 'Forest') & plotFrame['Dev_Mask'] & plotFrame['Urban_Mask'],'PredName'] = 'Developed'
                        else:
                            plotFrame.loc[(plotFrame['PredName'] == 'Forest') & plotFrame['Dev_Mask'],'PredName'] = 'Developed'

                stack.append(plotFrame)
            unstacked = pd.concat(stack).sort_index()
            gky_pred_names = unstacked['PredName'].to_numpy()

        elif mapName == 'Land_Cover' and clump_landcover_classes:
            gky_pred_names = [clumpDict_secondary[i] for i in gky_pred_names]
            classNames = allTrainingData[referenceColumn].value_counts().keys()
        
        #---------------------------------------------------------------------
        #     Apply 1 Or 2 Year Buffer
        #---------------------------------------------------------------------
        if mapName == 'Change' and buffer:
            print('Buffering Predictions')
            orig_preds = gky_pred_names
            gky_frame = allTrainingData.iloc[test_index][['PLOTID','YEAR']]
            gky_frame['pred'] = gky_pred_names
            gky_frame['orig_pred'] = gky_frame['pred']
            gky_frame['test'] = gky_test_names
            gky_frame['2before'] = gky_frame.groupby(['PLOTID'])['pred'].shift(2)
            gky_frame['1before'] = gky_frame.groupby(['PLOTID'])['pred'].shift(1)
            gky_frame['1after'] = gky_frame.groupby(['PLOTID'])['pred'].shift(-1)
            gky_frame['2after'] = gky_frame.groupby(['PLOTID'])['pred'].shift(-2)

            #test = gky_frame[gky_frame['PLOTID'] == 14351]
            # # 2 Year Buffer
            # for val in ['Slow_Loss','Fast_Loss','Gain']:
            #     gky_frame['match'] = gky_frame.apply(lambda row: val in row[['2before','1before','1after','2after']].to_numpy() and row.loc['test'] == val, axis = 1)
            #     gky_frame.loc[gky_frame['match'], 'pred'] = val

            # 1 Year Buffer
            for val in ['Slow_Loss','Fast_Loss','Gain']:
                gky_frame['match'] = gky_frame.apply(lambda row: val in row[['1before','1after']].to_numpy() and row.loc['test'] == val, axis = 1)
                gky_frame.loc[gky_frame['match'], 'pred'] = val

            gky_pred_names = gky_frame['pred'].to_numpy()
            gky_test_names = gky_frame['test'].to_numpy()
            test_index = gky_frame.index.to_numpy()

        #---------------------------------------------------------------------
        #      Remap to Values
        #---------------------------------------------------------------------                
        gky_pred = [remapDict[i] for i in gky_pred_names]
        gky_test = [remapDict[i] for i in gky_test_names]
        classValues = [remapDict[i] for i in classNames]

        #---------------------------------------------------------------------
        #     Apply Tree Mask or Forest Mask if Selected
        #---------------------------------------------------------------------
        if mapName == 'Change' and (applyTreeMask or applyForestMask):
            gky_plotids = allTrainingData['PLOTID'].iloc[test_index]
            mask = gky_plotids.isin(idsToInclude)
            npMask = mask.to_numpy()

            gky_pred = [i for ind, i in enumerate(gky_pred) if npMask[ind]]
            gky_test = [i for ind, i in enumerate(gky_test) if npMask[ind]]
            gky_pred_names = [i for ind, i in enumerate(gky_pred_names) if npMask[ind]]
            gky_test_names = gky_test_names[mask]
            gk_strata_test = gk_strata_test[mask] 

        #---------------------------------------------------------------------
        #       Get Accuracies and write to initial file
        #---------------------------------------------------------------------
        accuracy, balanced_accuracy, users, producers, kappa, f1_score, areas, accuracy_error, usersError, producersError, area_errors = \
            accuracySamplingLib.get_write_stratified_accuracies(pd.Series(gky_test), gky_pred, gk_strata_test, strataDict, 
                                                                classValues, '', accFile)

        cf_matrix = confusion_matrix(gky_test_names, gky_pred_names, labels = classDict[mapName])#classNames)
        df_cm = pd.DataFrame(cf_matrix, index = classDict[mapName], columns = classDict[mapName])#classNames)
        
        accFile.write(df_cm.to_string()+'\n')
        accFile.write('\n')
        accFile.write('# Overall Numbers for Iteration '+str(iter)+': \n')  
        accFile.write('Accuracy: '+str(accuracy)+'\n')
        accFile.write('Balanced Accuracy: '+str(balanced_accuracy)+'\n')
        accFile.write('Kappa: '+str(kappa)+'\n')
        accFile.write('F1 Score: '+str(f1_score)+'\n')

        accFile.write('Users Accuracy: \n')
        for c in users.keys():
            accFile.write(remapDictFlipped[str(c)]+': '+str(users[c])+'\n')
        for className in emptyClasses:
            accFile.write(className+': Not Modeled\n')

        accFile.write('Producers Accuracy: \n')
        for c in producers.keys():
            accFile.write(remapDictFlipped[str(c)]+': '+str(producers[c])+'\n')  
        for className in emptyClasses:
            accFile.write(className+': Not Modeled\n')
        
        iterAccuracy.append(accuracy)
        iterBalancedAccuracy.append(balanced_accuracy)
        iterKappa.append(kappa)
        iterF1Score.append(f1_score)
        iterUsers.append(users)
        iterProducers.append(producers)


#-----------------------------------------------------------------------------
#                     Save Final Stats to File
#-----------------------------------------------------------------------------

finalAccFile = open(finalAccuracyFilePath, 'w')

finalAccuracy = np.mean(iterAccuracy)   
finalBalancedAccuracy = np.mean(iterBalancedAccuracy)
finalKappa = np.mean(iterKappa)
finalF1Score = np.mean(iterF1Score)
finalUsers = {}
finalProducers = {}
for i in iterUsers[0].keys():
    finalUsers[i] = np.mean([dict[i] for dict in iterUsers])
for i in iterProducers[0].keys():
    finalProducers[i] = np.mean([dict[i] for dict in iterProducers]) 

finalAccFile.write(mapName+'\n')  
finalAccFile.write('\n')  
finalAccFile.write('Accuracy: '+"{:.2f}".format(finalAccuracy*100)+'\n')
finalAccFile.write('Balanced Accuracy: '+"{:.2f}".format(finalBalancedAccuracy*100)+'\n')
finalAccFile.write('Kappa: '+"{:.2f}".format(finalKappa)+'\n')
#finalAccFile.write('F1 Score: '+"{:.2f}".format(finalF1Score)+'\n')
finalAccFile.write('\n')

finalAccFile.write('Users Accuracy (100%-Commission Error): \n')
for className in classDict[mapName]:
    if className in emptyClasses:
        finalAccFile.write(className+': Not Modeled\n')
    else:
        finalAccFile.write(className+': '+"{:.2f}".format(finalUsers[remapDict[className]]*100)+'\n')
finalAccFile.write('\n')

finalAccFile.write('Producers Accuracy (100%-Omission Error): \n')
for className in classDict[mapName]:
    if className in emptyClasses:
        finalAccFile.write(className+': Not Modeled\n')
    else:
        finalAccFile.write(className+': '+"{:.2f}".format(finalProducers[remapDict[className]]*100)+'\n')
finalAccFile.write('\n')

finalAccFile.write('Number of Samples in each class: \n')
for className in classDict[mapName]:
    if className in emptyClasses:
        finalAccFile.write(className+': Not Modeled\n')
    else:
        finalAccFile.write(className+': '+str(value_counts[className])+'\n')
finalAccFile.write('\n')

finalAccFile.write('#------------------------------------------------------\n')
finalAccFile.write('Example Confusion Matrix from 1 out of '+str(num_iterations)+' iterations of the '+str(k)+'-fold cross-validation:\n')
finalAccFile.write(df_cm.to_string()+'\n')
finalAccFile.close()
