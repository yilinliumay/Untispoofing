from Featurizer import Featurizer
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np

class MFCCFeaturizer(Featurizer):
    def getFeatureRepresentation(self, audio_file,typeFilePath):
        print("MFCC Feature Extraction....")
        mfcc_list=list()

        for audio in audio_file:
                (rate, sig) = wav.read(typeFilePath+audio)
                mfcc_feat = mfcc(sig, rate)
                a=np.concatenate(mfcc_feat[0:12,:], axis=None)

                mfcc_list.append(a)

        mfcc_list=np.array(mfcc_list)
        return mfcc_list