import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
import time
import _pickle as pickle
import scikitplot as skplt
import os
import itertools

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_iter = [0]

def tick():
    _iter[0] += 1

def plot(d, name, value):
    _since_last_flush[d+name][_iter[0]] = value

def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
        _since_beginning[name].update(vals)

        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(name.replace(' ', '_')+'.jpg')

    # print ("iter {}\t{}".format(_iter[0], "\t".join(prints)))
    _since_last_flush.clear()

    # with open('log.pkl', 'wb') as f:
    #     pickle.dump(dict(_since_beginning), f, 3)

_since_beginning_vt = collections.defaultdict(lambda: {})
_since_last_flush_vt = collections.defaultdict(lambda: {})
# _since_beginning_val = collections.defaultdict(lambda: {})
# _since_last_flush_val = collections.defaultdict(lambda: {})
_iter_vt = [0]

def tick_vt():
    _iter_vt[0] += 1

def plot_vt(d, name, value_training, value_validation):
    _since_last_flush_vt[d+name][_iter_vt[0]] = [value_training, value_validation]

def flush_vt(adjust=False):

    for name, vals in _since_last_flush_vt.items():
        _since_beginning_vt[name].update(vals)

        x_vals = np.sort(list(_since_beginning_vt[name].keys()))
        y_vals_train = [_since_beginning_vt[name][x][0] for x in x_vals]
        y_vals_validation = [_since_beginning_vt[name][x][1] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals_train, label='Training')
        plt.plot(x_vals, y_vals_validation, label='Validation')
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(name.replace(' ', '_')+'.jpg')

        # plt.title("Validation Accuracy vs. Number of Training Epochs")
        # plt.xlabel("Training Epochs")
        # plt.ylabel("Validation Accuracy")
        # plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
        # plt.plot(range(1,num_epochs+1),shist,label="Scratch")
        # plt.ylim((0,1.))
        # plt.xticks(np.arange(1, num_epochs+1, 1.0))
        # plt.legend()

    _since_last_flush_vt.clear()

    # with open('logvt.pkl', 'wb') as f:
    #     pickle.dump(dict(_since_beginning_vt), f, 3)

def add_prefix(prefix, path):
    return os.path.join(prefix, path)

def plt_roc(test_y, probas_y, prefix, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro, plot_macro=plot_macro)
    plt.savefig(add_prefix(prefix, 'roc_auc_curve.png'))
    plt.close()


def plot_confusion_matrix(cm, classes, prefix, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    refence:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(add_prefix(prefix, 'confusion_matrix.png'))
    plt.close()