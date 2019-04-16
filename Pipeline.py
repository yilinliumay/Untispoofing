
from os import listdir
from os.path import isfile, join
from MFCCFeaturizer import MFCCFeaturizer
from LogbankFeaturizer import LogbankFeaturizer
from fbankFeaturizer import FbankFeaturizer
from SSCFeaturizer import SSCFeaturizer
from GMM import GMM
from SVM import SVM
from DNN import DNN
from Accuracy import Accuracy
from EER import EER
import numpy as np
import csv





class Pipeline(object):

    def __init__(self, trainFilePath, devFilePath,valFilePath,
                 train_laFielPath,dev_laFielPath,val_laFielPath,
                 featurizerMethod,classifierMethod,evaluation):

        # for different feature extraction method and different classify mehtod
        self.featurizer = featurizerMethod
        self.classifier = classifierMethod

        # get all audio files and labels
        train_audio,train_label=self.data_label(trainFilePath,train_laFielPath)
        dev_audio,dev_label = self.data_label(devFilePath, dev_laFielPath)
        val_audio, val_label = self.data_label(valFilePath, val_laFielPath)

        clf=self.untispoofing_train(train_audio,train_label,"ASVspoof2017_V2_train/")

        train_result = self.untispoofing_predict(train_audio, train_label, "ASVspoof2017_V2_train/",clf)
        print("Evaluation of Training Set: "+str(train_result))
        dev_result=self.untispoofing_predict(dev_audio,dev_label,"ASVspoof2017_V2_dev/",clf)
        print("Evaluation of Development Set: "+str(dev_result))
        val_result = self.untispoofing_predict(val_audio, val_label,"ASVspoof2017_V2_eval/",clf)
        print("Evaluation of Evaluation Set: "+str(val_result))



    def data_label(self,dataFilePath,labelFilePath):

        # get all audio files
        audio_files = [f for f in listdir(dataFilePath) if isfile(join(dataFilePath, f))]

        # get corresponding labels

        with open(labelFilePath) as inf:
            reader = csv.reader(inf, delimiter=" ")
            second_col = list(zip(*reader))[1]
            second_col=np.array(second_col)

        return audio_files,second_col

    def untispoofing_train(self,audio,label,typeFilePath):

        print("Start Training")
        # feature
        feature = self.featurizer.getFeatureRepresentation(audio,typeFilePath)
        # train
        clf = self.classifier.buildClassifier(feature, label)
        print("Training Done")

        return clf

    def untispoofing_predict(self,audio,label,typeFilePath,clf):
        print("Start Predicting")
        # feature
        feature = self.featurizer.getFeatureRepresentation(audio, typeFilePath)
        # predict
        label_predict, label_predict_proba = self.classifier.predict(feature, clf)
        # print(label_predict)

        # evaluation
        result = evaluation.compute(label, label_predict, label_predict_proba)
        print("Predicting Done")
        return result





if __name__ == '__main__':
    trainFilePath="ASVspoof2017_V2_train"
    devFilePath="ASVspoof2017_V2_dev"
    evalFilePath="ASVspoof2017_V2_eval"
    train_laFielPath='protocol_V2/ASVspoof2017_V2_train.trn.txt'
    dev_laFielPath ="protocol_V2/ASVspoof2017_V2_dev.trl.txt"
    eval_laFielPath ="protocol_V2/ASVspoof2017_V2_eval.trl.txt"
    featurizerInstance =SSCFeaturizer()
    classifierInstance =DNN()
    evaluation=EER()


    trainInstance=Pipeline(trainFilePath,devFilePath,evalFilePath,train_laFielPath,dev_laFielPath,eval_laFielPath,featurizerInstance,classifierInstance,evaluation)

