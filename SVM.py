from Classifier import Classifier
from sklearn import svm

class SVM (Classifier):
    # The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
    def buildClassifier(self, X_features, Y_train):
        print("Build SVM model")
        clf = svm.SVC(gamma='scale',probability=True)
        clf.fit(X_features, Y_train)
        return clf

    def predict(self,audio_feature,clf):
        print("Predict labels using SVM model")
        label_predict = clf.predict(audio_feature)
        label_predict_proba = clf.predict_proba(audio_feature)
        return label_predict,label_predict_proba[:,0]
