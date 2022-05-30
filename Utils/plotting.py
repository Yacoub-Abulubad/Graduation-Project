from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np


def Decoding(y_pred, y_true):
    y_pred_dec = []
    y_true_dec = []
    maximum = 0.0
    index = 0
    for i in range(len(y_pred)):
        maximum = max(y_pred[i])
        index = np.where(np.asarray(y_pred[i]) == maximum)
        y_pred_dec.append(index[0][0])
        index = 0

    index = 0
    for i in range(len(y_true)):
        maximum = y_true[i][0]
        for j in range(0, len(y_true[0])):
            if maximum < y_true[i][j]:
                index = j
        y_true_dec.append(index)
        index = 0

    return y_pred_dec, y_true_dec, len(y_true[0])


def scoring_system(y_pred_dec, y_true_dec, length, classes_list):

    confusion = confusion_matrix(y_true_dec, y_pred_dec)
    tp = sum(confusion[i][i] for i in range(length))
    fp = 0
    for j in range(length):
        for w in range(length):
            if j == w:
                pass
            else:
                fp = fp + confusion[w][j]

    accuracy = tp / (tp + fp)

    print("Accuracy:")
    print(accuracy)

    df = pd.DataFrame(confusion, index=classes_list, columns=classes_list)

    fig, ax = plt.subplots(figsize=(length + 3, length + 2))
    plt.xlabel("Predicted")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    plt.Axes.set_title(ax, "prediction", fontsize=15)
    plt.ylabel("Actual", fontsize=15)
    plt.ylim(0, length)
    plt.xlim(length, 0)
    sn.heatmap(df,
               annot=True,
               fmt="d",
               cmap='Blues',
               linecolor='black',
               linewidths=1)
    plt.ylabel('Actual', rotation=0, va='center')
    plt.yticks(rotation=0)


def Conf_Mat_Plot(y_pred, y_true, classes_list):
    y_pred_dec, y_true_dec, length = Decoding(y_pred, y_true.tolist())

    scoring_system(y_pred_dec, y_true_dec, length, classes_list)


def EFF_graphing(history):
    plt.plot(history.history[list(history.history)[1]])
    plt.plot(history.history[list(history.history)[4]])
    plt.title('model precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history[list(history.history)[2]])
    plt.plot(history.history[list(history.history)[5]])
    plt.title('model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def EFFC_graphing(history):
    plt.plot(history.history[list(history.history)[1]])
    plt.plot(history.history[list(history.history)[4]])
    plt.title('model precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history[list(history.history)[2]])
    plt.plot(history.history[list(history.history)[5]])
    plt.title('model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_performance(history, Classifier=True):
    if Classifier == True:
        EFFC_graphing(history)
    else:
        EFF_graphing(history)
