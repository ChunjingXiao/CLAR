import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE

from tensorflow.keras.applications.vgg16 import preprocess_input

import re
import scipy.io as sio
csi = sio.loadmat('data/csi_label/csi_sifi1250.mat')
csi_labels = sio.loadmat('data/csi_label/csi_label_sifi1250.mat')
csi=csi['csi'][:].astype(np.uint8)
csi_label = csi_label['csi_label'][0]
print('evaluate_shape(csi):', csi.shape)
def get_features(base_model, df, class_labels, VGG=True):
    """Computes base_model features encoding from dataframe df
    Args:
        base_model: encoder
        df: Dataframe
        class_labels: labels (strings)
        VGG: Boolean to confirm if model is VGG (important for preprocessing)
    Returns:
        features: base_model features encodings
        y: labels (int)
        feats: dictionary with features per class
    Notes:
        Not strictly necessary, but features are calculated seperately for each
        class in feats variable.
    """
    feats = {}
    class_instances = {}
    for i, label in enumerate(class_labels):
        feats[label] = []
        class_instances[label] = (df.class_label == label).sum()


    features = []
    for label in class_labels:
        for index in range(class_instances[label]):

            filename = df.loc[df["class_label"] == label]["filename"].iloc[
                index
            ]
            # # 采用正则化的方法
            csi_name = re.findall('\d+', filename)  # list,'\d+'提取整数
            csi_index = list(map(int, csi_name))  # int
            csi_one = np.squeeze(csi[csi_index, :, :, :], axis=0)  # axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
            img = cv.cvtColor(csi_one, cv.COLOR_BGR2RGB)


            if VGG:

                img_pr = preprocess_input(img)
            else:

                img_pr = img

            # Feature prediction for each img + Normalization
            feat_un_norm = base_model.predict(np.array([img_pr])).flatten()
            feat_norm = feat_un_norm / np.max(np.abs(feat_un_norm), axis=0)
            if len(feats[label]) == 0:
                feats[label] = np.array(feat_norm)
            else:
                feats[label] = np.vstack((feats[label], feat_norm))
        if len(features) == 0:
            features = feats[label]
        else:
            if len(feats[label]) == 0:
                print("{0} is empty".format(label))
            else:
                features = np.vstack((features, feats[label]))


    y = []
    for i, label in enumerate(class_labels):
        y = np.concatenate((y, np.array([i] * class_instances[label])))

    return features, y, feats

def linear_classifier(
    features_train,
    y_train,
    features_test,
    y_test,
    class_labels,
    fraction=1.0,
    test_size=0.2,
):
    """Evaluates the feature quality by the means of a logistic regression classifier
    Args:
        features: Training instances
        y: labels (ints)
        class_labels: labels (strings)
        fraction: fraction of features used for training clf
        test_size: fraction of features used for evaluation clf
    Prints:
        Top-1 accuracy and classification reports
    Notes:
        10-fold cross-validation is used to tune regularization hyperparameters of clf
    """
    if fraction != 1.0:
        features_train, feat_not_used, y_train, y_not_used = train_test_split(
            features_train,
            y_train,
            test_size=1 - fraction/(1-test_size),
            random_state=42,
            shuffle=True,
            stratify=y_train
        )

    clf = LogisticRegressionCV(cv=3, max_iter=1000, verbose=0, n_jobs=1).fit(
        features_train, y_train,
    )


    print(f"Accuracy on test: {round(clf.score(features_test, y_test),2)} \n")
    y_pred_test = clf.predict(features_test)
    classification_report_test = classification_report(
        y_test,
        y_pred_test,
        labels=list(range(0, len(class_labels))),
        target_names=class_labels,
    )



def random_indexes(a, b, feats_in_plot):
    """Support function for tSNE_vis
    Args:
        a: start index
        b: end index
        feats_in_plot: # of featuers to be plotted per class
    Returns:
        Random list of feats_in_plot indexes between a and b
    """
    randomList = []

    for i in range(feats_in_plot):

        randomList.append(random.randint(a, b - 1))

    return randomList


def tSNE_vis(
    df,
    features,
    class_labels,
    save_tag="J119",
    save_figure=True,
    feats_in_plot=150,
):
    """Plots the feature quality by the means of t-SNE
    Args:
        df: Dataframe
        features: Training instances
        class_labels: labels (strings)
        save_tag: title of plot to save
    Prints & Saves:
        t-SNE plot of 250 instances of each class
    """

    class_colours = ["1", "2", "3", "4", "8"]
    class_markers = [str(i) for i in range(10)]
    class_instances = {}
    for i, label in enumerate(class_labels):
        class_instances[label] = (df.class_label == label).sum()

    tsne_m = TSNE(n_jobs=8, random_state=42)
    X_embedded = tsne_m.fit_transform(features)

    fig = plt.figure(figsize=(10, 10))
    lr = 150
    p = 50
    index = 0

    for (label, colour, c_i) in zip(
        class_labels, class_colours, class_instances

    ):
        indexes = random_indexes(
            index, index + class_instances[label], feats_in_plot
        )
        plt.scatter(X_embedded[indexes, 0], X_embedded[indexes, 1], marker=colour)
        index += class_instances[label]

    fig.legend(
        bbox_to_anchor=(0.075, 0.061),
        loc="lower left",
        ncol=1,
        labels=class_labels,
    )
    if save_figure:
        plt.savefig(
            "figures/" + save_tag + ".svg", bbox_inches="tight",
        )
