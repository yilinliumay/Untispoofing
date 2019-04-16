from Classifier import Classifier
from sklearn import mixture

class GMM(Classifier):

    def buildClassifier(self, audio_feature, audio_label):
        print("Build GMM model")
        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        #print(audio_feature)
        clf.fit(audio_feature)
        return clf

    def predict(self,audio_feature,clf):
        print("Predict labels using GMM model")
        label_predict = clf.predict(audio_feature)
        final_label_predict=['genuine'if label==0 else 'spoof' for label in label_predict]
        label_predict_proba=clf.predict_proba(audio_feature)
        return final_label_predict,label_predict_proba[:,0]






