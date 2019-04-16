from sklearn.metrics import roc_curve, auc
from Evaluation import Evaluation
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np


class EER(Evaluation):

    def compute(self,y,label_predict,y_pred):
        print("Evaluation by EER")
        fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label="genuine")
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
        return eer






