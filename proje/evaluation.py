# evaluation.py
import numpy as np

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    return matrix.tolist()

def calculate_precision_recall(conf_matrix):
    precisions = []
    recalls = []
    for i in range(len(conf_matrix)):
        tp = conf_matrix[i][i]
        fp = sum(conf_matrix[j][i] for j in range(len(conf_matrix)) if j != i)
        fn = sum(conf_matrix[i][j] for j in range(len(conf_matrix)) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(round(precision, 4))
        recalls.append(round(recall, 4))
    return precisions, recalls

def calculate_accuracy(conf_matrix):
    correct = sum(conf_matrix[i][i] for i in range(len(conf_matrix)))
    total = sum(sum(row) for row in conf_matrix)
    return round(correct / total, 4) if total > 0 else 0
