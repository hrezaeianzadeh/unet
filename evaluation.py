import os
import numpy
from matplotlib import pyplot, pylab

def plot_results(X, y_true, y_pred, num_samples=5, save_dir=None):
    pyplot.figure()
    for k in range(num_samples):
        x, t, p = numpy.squeeze(X[k]), numpy.squeeze(y_true[k]), numpy.squeeze(y_pred[k])
        # p[p > 0.3] = 1
        # p[p < 0.3] = 0
        # print(numpy.unique(p))
        pylab.subplot(5, 3, k*3 + 1)
        pylab.axis('off')
        pylab.imshow(x)
        if k == 0:
            pylab.title('RGB')
        pylab.subplot(5, 3, k*3 + 2)
        pylab.axis('off')
        pylab.imshow(t)
        if k == 0:
            pylab.title('y_true')
        pylab.subplot(5, 3, k*3 + 3)
        pylab.axis('off')
        pylab.imshow(p)
        if k == 0:
            pylab.title('y_pred')
    if save_dir is not None:
        pylab.savefig('report_unet.png', dpi=250)
    return pylab.show()

def pixel_wise_eval(y_true, y_pred):
    y_true, y_pred = numpy.squeeze(y_true), numpy.squeeze(y_pred)
    tp, tn, fp, fn = 0, 0, 0, 0
    for k in range(y_pred.shape[0]):
        t, p = y_true[k], y_pred[k]
        # p[p > 0.3] = 1
        # p[p < 0.3] = 0
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[2]):
                if p[i, j] > 0 and t[i, j] > 0:
                    tp += 1
                elif p[i, j] == 0 and t[i, j] == 0:
                    tn += 1
                elif p[i, j] > 0 and t[i, j] == 0:
                    fp += 1
                elif p[i, j] == 0 and t[i, j] > 0:
                    fn += 1

    precision = 0
    recall = 0
    f_score = 0

    if (tp + fp) != 0: precision = tp / (tp + fp)
    if (tp + fn) != 0: recall = tp / (tp + fn)
    if (precision + recall) != 0: f_score = (2 * precision * recall) / (precision + recall)

    accuracy = (tp + tn) / (2048 * 2048)

    return {'Accuracy': accuracy, 'True Positive': tp, 'True Negative': tn, 'False Positive': fp, 'False Negative': fn, 'Presision': precision, 'Recall': recall, 'F_Score': f_score}


def evaluate_model(repo):
    y_true = numpy.load(os.path.join(repo, 'y_test.npy'))
    y_pred = numpy.load(os.path.join(repo, 'y_pred.npy'))
    X_test = numpy.load(os.path.join(repo, 'X_test.npy'))

    plot_results(X_test, y_true, y_pred)

    return print(pixel_wise_eval(y_true=y_true, y_pred=y_pred))