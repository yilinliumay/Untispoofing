from Featurizer import Featurizer
import scipy.io.wavfile as wav
from python_speech_features import fbank
import numpy as np

class FbankFeaturizer(Featurizer):
    def getFeatureRepresentation(self, audio_file,typeFilePath):
        print("Fbank Feature Extraction....")
        fbank_list=list()

        for audio in audio_file:
            (rate, sig) = wav.read(typeFilePath+audio)
            fbank_feat,energy = fbank(sig, rate)
            a=np.concatenate(fbank_feat[0:12,:], axis=None)
            # print(a)
            fbank_list.append(a)

        fbank_list=np.array(fbank_list)
        return fbank_list