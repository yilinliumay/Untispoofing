from Featurizer import Featurizer
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import numpy as np

class LogbankFeaturizer(Featurizer):

    def getFeatureRepresentation(self, audio_file,typeFilePath):
        print("LogBank Feature Extraction....")
        logbank_list=list()

        for audio in audio_file:
            (rate, sig) = wav.read(typeFilePath+audio)
            logbank_feat = logfbank(sig, rate)
            a=np.concatenate(logbank_feat[0:12,:], axis=None)
            # print(a)
            logbank_list.append(a)

        mfcc_list=np.array(logbank_list)
        return mfcc_list