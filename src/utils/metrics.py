from sklearn.metrics import confusion_matrix
from numpy import round as np_round

def classwise_accuracy_score(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    l_acc = np_round(cm.diagonal()/cm.sum(axis=1), 2).tolist()

    if class_names is not None:
        return dict(zip(class_names, l_acc))
    else:
        return l_acc

# unit test
if __name__ == "__main__":
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    acc = classwise_accuracy_score(y_true, y_pred, ["c0","c1","c2"])
    print(acc)