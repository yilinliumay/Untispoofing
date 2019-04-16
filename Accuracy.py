from Evaluation import Evaluation


class Accuracy(Evaluation):

    def compute(self, train_label,label_predict,label_predict_proba):
        print("Evaluation by Accuracy")
        count = 0
        for y, y_predict in zip(train_label, label_predict):
            if y != y_predict:
                count += 1
        return (1 - float(count) / len(train_label))
