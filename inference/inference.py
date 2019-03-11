import matplotlib
matplotlib.use('Agg')
from bigdl.util.common import *
from bigdl.nn.layer import Model
from bigdl.dlframes.dl_classifier import DLModel
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import argparse
import sys

sns.set(style="darkgrid")

def loadParquet(spark, filePath, featureCol, label):
    df = spark.read.format('parquet') \
            .load(filePath) \
            .select(featureCol + [label])
    return df

def createSample(df, featureCol, labelCol):
    rdd = df.rdd \
            .map(lambda row: Sample.from_ndarray(
                [np.asarray(row.__getitem__(feature)) for feature in featureCol],
                np.asarray(row.__getitem__(labelCol)) + 1
            ))
    return rdd

def loadModel(modelPath):
    model = Model.loadModel(modelPath=modelPath+'.bigdl', weightPath=modelPath+'.bin')
    return model

def computeAUC(y_true, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print("\t*AUC score for tt selector: {:.4f}\n\n".format(roc_auc[0]))
    return fpr,tpr,roc_auc 

def savePlot(plotDir, values, modelType):
    plt.figure()
    for (fpr,tpr,roc_auc),modelName in zip(values, modelType):
        plt.plot(fpr[0], tpr[0],
            lw=2, label=modelName+' classifier (AUC) = %0.4f' % roc_auc[0])
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Background Contamination (FPR)')
    plt.ylabel('Signal Efficiency (TPR)')
    plt.title('$tt$ selector')
    plt.legend(loc="lower right")
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor')
    plt.savefig(plotDir+'/roc.pdf')


def inference(spark, args):
    
    ## Lists containing fpr, tpr, auc for each model
    results = []
    models = []

    for model in args.models:

        print("\n\nProcessing "+model)

        ## Full path to the model
        model = args.modelsDir + "/" +model
        ## Check the type of model
        featureCol = None
        label = 'encoded_label'

        if 'hlf' in model:
            models.append('HLF')
            featureCol = ['HLF_input']
        elif 'gru' in model:
            models.append('Particle-sequence')
            featureCol = ['GRU_input']
        elif 'inclusive' in model:
            models.append('Inclusive')
            featureCol = ['GRU_input', 'HLF_input']
        else:
            sys.exit("Error, Invalid model type")

        print("\t*Loading test dataset")
        testDF = loadParquet(spark, args.testDataset, featureCol, label)
        testRDD = createSample(testDF,featureCol,label)
        model = loadModel(model)
        print("\t*Predicting")
        pred = model.predict(testRDD)

        ## Collect results
        print("\t*Collecting results")
        y_true = np.asarray(testDF.rdd \
                    .map(lambda row: row.encoded_label) \
                    .collect())
        y_pred = np.asarray(pred.collect())

        results.append(computeAUC(y_true, y_pred))

        del y_true, y_pred  
    
    savePlot(args.plotDir, results, models)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--testDataset', type=str,
                        help ='path to the test dataset')
    parser.add_argument('--models', nargs='+', type=str,
                        help='List of models to compare')
    parser.add_argument('--modelsDir', type=str,
                        help='Path to the directory containing models')
    parser.add_argument('--plotDir', type=str,
                        help='Directory where to store plots')
    
    args = parser.parse_args()

    ## Create SparkContext and SparkSession
    sc = SparkContext(
        appName="Prediction",
        conf=create_spark_conf())
    spark = SQLContext(sc).sparkSession
    init_engine()
    sc.setLogLevel("ERROR")

    inference(spark,args)