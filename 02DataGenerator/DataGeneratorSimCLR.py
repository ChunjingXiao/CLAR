import cv2 as cv

import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.utils import data_utils

from SimCLR_data_util import *
import random
import data_process
import re
import scipy.io as sio

csi = sio.loadmat('data/csi_label/csi_sifi1250.mat')
csi_labels = sio.loadmat('data/csi_label/csi_label_sifi1250.mat')
csi = csi['csi'][:].astype(np.uint8)
csi_labels = csi_labels['csi_label'][0]
print('DateGenerator_shape(csi):', csi.shape)

class DataGeneratorSimCLR(data_utils.Sequence):
    def __init__(
            self,
            df,
            batch_size=50,
            subset="train",
            shuffle=True,
            info={},
            width=30,
            height=200,
            VGG=True,
    ):
        super().__init__()
        self.df = df
        self.indexes = np.asarray(self.df.index)
        self.indexesla = np.asarray(self.df[:300].index)
        self.indexesun = np.asarray(self.df[300:].index)
        self.batch_size = batch_size
        self.subset = subset
        self.shuffle = shuffle
        self.info = info
        self.width = width
        self.height = height
        self.VGG = VGG
        self.on_epoch_end()

    def __len__(self):
         return int(np.ceil(len(self.df[:300]) / float(self.batch_size)) + np.ceil(len(self.df[300:]) / float(self.batch_size)))#加注释

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.indexesla)
            np.random.shuffle(self.indexesun)
        # pass

    def __getitem__(self, index):

        X = np.empty(
            (2 * self.batch_size, 1, self.height, self.width, 3),
            dtype=np.float32,
        )

        indexes = self.indexes[
            index*self.batch_size: (index + 1)*self.batch_size
        ]
        batch_data = self.df.iloc[indexes]

        if index < int(np.ceil(len(self.indexesla)/float(self.batch_size))):
           index_la = index
           indexla = self.indexesla[index_la*self.batch_size:(index_la+1)*self.batch_size]
           batch_data = self.df.iloc[indexla]
        else:
           index_un = index - int(np.ceil(len(self.indexesla)/float(self.batch_size)))
           indexun = self.indexesun[index_un*self.batch_size:(index_un+1)*self.batch_size]
           batch_data = self.df.iloc[indexun]


        shuffle_a = np.arange(self.batch_size)
        shuffle_b = np.arange(self.batch_size)

        if self.subset == "train":
            random.shuffle(shuffle_a)
            random.shuffle(shuffle_b)
            # pass
        if self.subset == "val":
            pass
        labels_ab_aa = np.zeros((self.batch_size, 2 * self.batch_size))
        labels_ba_bb = np.zeros((self.batch_size, 2 * self.batch_size))
        global min_covered, crop_num, record_num
        if index < int(np.ceil(len(self.indexesla)/float(self.batch_size))):
             for l, row in enumerate(batch_data.iterrows()):
                 filename = row[1]["filename"]
                 csi_name = re.findall('\d+', filename)
                 csi_index = list(map(int, csi_name))
                 csi_one = np.squeeze(csi[csi_index, :, :, :], axis=0)


                 y_class = row[1]["class_label"]
                 df_lab = self.df.iloc[self.indexesla]
                 imgs_lab = df_lab[df_lab['class_label'] == y_class]

                 imgs_lab_1 = imgs_lab.drop(imgs_lab[imgs_lab['filename'] == filename].index)
                 imgs = imgs_lab_1.sample(n=2, axis=0).reset_index(drop=True)

                 img1_name = imgs['filename'][0]
                 csi1_name = re.findall('\d+', img1_name)
                 csi1_index = list(map(int, csi1_name))
                 csi1_one = np.squeeze(csi[csi1_index, :, :, :], axis=0)

                 img_dtw, _, path = data_process.dtw_distance(csi_one[:, 0, 0].astype(np.int), csi1_one[:, 0, 0].astype(np.int))
                 img1 = np.empty((200, 30, 3), dtype=np.float32)
                 for ii, jj in path:
                     img1[ii] = 0.6*csi_one[ii].astype(np.int) + 0.4 * (csi1_one[jj].astype(np.int))
                 img1 = img1.astype(np.uint8)
                 img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
                 img_1 = tf.convert_to_tensor(
                     np.asarray((img1 / 255)).astype("float32")
                 )

                 img2_name = imgs['filename'][1]
                 csi2_name = re.findall('\d+', img2_name)
                 csi2_index = list(map(int, csi2_name))
                 csi2_one = np.squeeze(csi[csi2_index, :, :, :], axis=0)
                 img_dtw, _, path = data_process.dtw_distance(csi_one[:, 0, 0].astype(np.int), csi2_one[:, 0, 0].astype(np.int))
                 img2 = np.empty((200, 30, 3), dtype=np.float32, )
                 for ii, jj in path:
                    img2[ii] = (0.6*csi_one[ii].astype(np.int) + 0.4 * (csi2_one[jj].astype(np.int))).astype(np.uint8)

                 img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
                 img_2 = tf.convert_to_tensor(
                     np.asarray((img2 / 255)).astype("float32")
                 )

                 num_chooses = list(enumerate([0, 20, 40, 60, 80, 100]))
                 random.shuffle(num_chooses)
                 record_num = num_chooses[0][1]
                 min_covered = 0.2

                 crop_num = 0
                 img_T1 = preprocess_for_train(
                     img_1,
                     self.height,
                     self.width,
                     crop=False,
                     random_slide=True,
                     flip=True,
                     color_distort=True,
                     blur=True,
                 )

                 crop_num = 1
                 img_T2 = preprocess_for_train(
                     img_2,
                     self.height,
                     self.width,
                     crop=False,
                     random_slide=True,
                     flip=True,
                     color_distort=True,
                     blur=True,
                 )

                 if self.VGG:
                     img_T1 = tf.dtypes.cast(img_T1 * 255, tf.int32)
                     img_T2 = tf.dtypes.cast(img_T2 * 255, tf.int32)
                     weight = data_process.weight_cac((img_T1).numpy(), (img_T2).numpy())

                     img_T1 = preprocess_input(np.asarray(img_T1))
                     img_T2 = preprocess_input(np.asarray(img_T2))



                 X[shuffle_a[l]] = img_T1

                 X[self.batch_size + shuffle_b[l]] = img_T2


                 labels_ab_aa[shuffle_a[l], shuffle_b[l]] = weight

                 labels_ba_bb[shuffle_b[l], shuffle_a[l]] = weight
             y = tf.concat([labels_ab_aa, labels_ba_bb], 1)
        else:

            for l, row in enumerate(batch_data.iterrows()):
                filename = row[1]["filename"]

                csi_name = re.findall('\d+', filename)
                csi_index = list(map(int, csi_name))
                csi_one = np.squeeze(csi[csi_index, :, :, :], axis=0)

                img = cv.cvtColor(csi_one, cv.COLOR_BGR2RGB)
                img = tf.convert_to_tensor(
                    np.asarray((img / 255)).astype("float32")
                )

                num_chooses = list(enumerate([0, 20, 40, 60, 80, 100]))
                random.shuffle(num_chooses)
                record_num = num_chooses[0][1]
                min_covered = 0.2
                crop_num = 0


                img_T1 = preprocess_for_train(
                    img,
                    self.height,
                    self.width,
                    crop=False,
                    random_slide=True,
                    flip=True,
                    color_distort=True,
                    blur=True,
                )
                crop_num=1
                img_T2 = preprocess_for_train(
                    img,
                    self.height,
                    self.width,
                    crop=False,
                    random_slide=True,
                    flip=True,
                    color_distort=True,
                    blur=True,
                )

                if self.VGG:
                    img_T1 = tf.dtypes.cast(img_T1 * 255, tf.int32)
                    img_T2 = tf.dtypes.cast(img_T2 * 255, tf.int32)
                    weight = data_process.weight_cac((img_T1).numpy(), (img_T2).numpy())
                    img_T1 = preprocess_input(np.asarray(img_T1))
                    img_T2 = preprocess_input(np.asarray(img_T2))


                X[shuffle_a[l]] = img_T1

                X[self.batch_size + shuffle_b[l]] = img_T2


                labels_ab_aa[shuffle_a[l], shuffle_b[l]] = weight  # weig_lab[num_chooses[0][0]]

                labels_ba_bb[shuffle_b[l], shuffle_a[l]] = weight  # weig_lab[num_chooses[0][0]]

            y = tf.cast(tf.concat([labels_ab_aa, labels_ba_bb], 1), tf.float32)


        return list(X), y



