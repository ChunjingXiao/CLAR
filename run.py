import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from evaluate_features import get_features, linear_classifier, tSNE_vis

df = pd.read_pickle("data/pickle/df_mat_sifi1250.pickle")
class_labels = ['label_'+str(i) for i in range(10)]
num_classes = len(df['class_one_hot'][0])
df_train, df_test = train_test_split(df, test_size=0.2, random_state=52, shuffle=True, stratify=df['class_label'])
df_train_1 = pd.read_pickle("data/pickle/df_mat_train_sifi1250.pickle")
df_test_1 = pd.read_pickle("data/pickle/df_mat_test_sifi1250.pickle")
df_2 = pd.concat([df_train, df_test], ignore_index=True)
df_val = None
df_val_1 = None

dfs = {
    'data': df_2,
    "train": df_train,
    'val': df_val,
    "test": df_test
}
# Img size
height_img = 200
width_img = 30
input_shape = (height_img, width_img, 3)


params_vgg16 = {'weights': "imagenet",
                'include_top': False,
                'input_shape': input_shape,
                'pooling': None}


base_model = VGG16(**params_vgg16)
feat_dim = 4 * 1 * 512

from DataGeneratorSimCLR import DataGeneratorSimCLR as DataGenerator
from SimCLR import SimCLR
batch_size = 50

num_layers_ph = 2
feat_dims_ph = [2048, 128]
num_of_unfrozen_layers = 1
save_path = 'models/newtest'

SimCLR = SimCLR(
        base_model = base_model,
        input_shape = input_shape,
        batch_size = batch_size,
        feat_dim = feat_dim,
        feat_dims_ph = feat_dims_ph,
        num_of_unfrozen_layers = num_of_unfrozen_layers,
        save_path = save_path
    )

params_generator = {'batch_size': batch_size,
                    'shuffle': True,
                    'width': width_img,
                    'height': height_img,
                    'VGG': True
                   }


data_val=''
data_val_0=""
data_train = DataGenerator(df_train.reset_index(drop=True), **params_generator)
data_train_0 = DataGenerator(df_2.reset_index(drop=True), **params_generator)
data_test = DataGenerator(df_test.reset_index(drop=True), subset = "test", **params_generator)
SimCLR.unfreeze_and_train(data_train_0, data_val_0, num_of_unfrozen_layers = 19, r = 19, lr = 1e-4, epochs = 50)
y_predict_test_after = SimCLR.predict(data_test)
base_model = SimCLR.base_model
features_train, y_train, feats = get_features(base_model, df_train, class_labels)
features_train_1, y_train_1, feats = get_features(base_model, df_train_1, class_labels)
features_test, y_test, feats = get_features(base_model, df_test, class_labels)
features_test_1, y_test_1, feats = get_features(base_model, df_test_1, class_labels)


fractions = [1.0]
for fraction in fractions:
    print(f"    ==== Linear {fraction * 100}% of the training data used ==== \n")
    linear_classifier(features_train, y_train, features_test, y_test, class_labels, fraction = fraction)


batch_size_classifier = 32
fractions = [1.0]
params_generator_classifier = {'max_width':width_img,
                            'max_height': height_img,
                            'num_classes': num_classes,
                            'VGG': True
                            }

params_training_classifier = {'1.0':{
                                "reg_dense" : 0.005,
                                "reg_out" : 0.005,
                                "nums_of_unfrozen_layers" : [5, 5, 6, 7],
                                # "lrs": [1e-5, 1e-4, 1e-6, 1e-4],
                                # "epochs" : [20, 10, 20, 20]
                                "lrs": [1e-4, 1e-5, 5e-5, 1e-5],
                                "epochs" : [20, 10, 20, 20]
                                # "lrs" : [1e-3, 1e-4, 5e-5, 1e-5],
                                # "epochs" : [10, 10, 20, 20]
                                },
                              '0.6':{
                                "reg_dense": 0.005,
                                "reg_out": 0.005,
                                "nums_of_unfrozen_layers": [5, 5, 6, 7],
                                # "lrs": [1e-5, 1e-4, 1e-6, 1e-4],
                                # "epochs": [20, 10, 20, 20],
                                "lrs": [1e-4, 1e-5, 5e-5, 1e-5],
                                "epochs" : [20, 10, 20, 20]
                                # "lrs": [1e-3, 1e-4, 5e-5, 5e-5],
                                # "epochs": [10, 10, 15, 20]
                              },
                              '0.4': {
                                "reg_dense": 0.005,
                                "reg_out": 0.005,
                                "nums_of_unfrozen_layers": [5, 5, 6, 7],
                                # "lrs": [1e-5, 1e-4, 5e-5, 1e-6],
                                # "epochs": [20, 10, 20, 20],
                                "lrs": [1e-4, 1e-5, 5e-5, 1e-5],
                                "epochs" : [20, 10, 20, 20]
                                # "lrs" : [1e-3, 1e-4, 5e-5, 1e-6],
                                # "epochs" : [10, 20, 20, 15]
                              },
                              '1.0':{
                                "reg_dense" : 0.005,
                                "reg_out" : 0.001,
                                "nums_of_unfrozen_layers" : [5, 5, 6, 7],
                                "lrs" : [1e-4, 1e-5, 5e-5,1e-5],
                                "epochs" : [10, 10, 20, 15]
                                # "lrs": [1e-3, 1e-4, 5e-5, 1e-5],
                                # "epochs": [10, 20, 20, 15]
                              }
                            }

for fraction in fractions:
    print(f"    Fine-tuned==== {fraction * 100}% of the training data used ==== \n")
    SimCLR.train_NL_and_evaluate(dfs = dfs,
                                batch_size = batch_size_classifier,
                                params_generator = params_generator_classifier,
                                fraction = fraction,
                                class_labels = class_labels,
                                reg_dense = params_training_classifier[str(fraction)]["reg_dense"],
                                reg_out = params_training_classifier[str(fraction)]["reg_out"],
                                nums_of_unfrozen_layers = params_training_classifier[str(fraction)]["nums_of_unfrozen_layers"],
                                lrs = params_training_classifier[str(fraction)]["lrs"],
                                epochs = params_training_classifier[str(fraction)]["epochs"],
                                verbose_epoch = 0,
                                verbose_cycle = 1
                                )