from Featurizer import Featurizer
import scipy.io.wavfile as wav
from python_speech_features import ssc
import numpy as np

class SSCFeaturizer(Featurizer):
    def getFeatureRepresentation(self, audio_file,typeFilePath):
        print("SSC Feature Extraction....")

        ssc_list=list()

        for audio in audio_file:
            (rate, sig) = wav.read(typeFilePath+audio)
            ssc_feat = ssc(sig, rate)
            a=np.concatenate(ssc_feat[0:12,:], axis=None)
            # print(a)
            ssc_list.append(a)

        ssc_list=np.array(ssc_list)
        return ssc_list