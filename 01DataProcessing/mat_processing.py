import pandas as pd
import numpy as np
from data_process import csi_split, csi_cat
csi_train, csi_train_y, csi_test, csi_test_y = csi_split()


df_mat_train = pd.DataFrame(columns=('filename', 'class_label', 'class_one_hot'))
df_mat_test = pd.DataFrame(columns=('filename', 'class_label', 'class_one_hot'))

for i in range(len(csi_train)):
    filename = 'csi_train_' + str(i)
    class_label = 'label_'+str(csi_train_y[i])
    class_one_hot = np.zeros(10, dtype=np.float32)
    class_one_hot[csi_train_y[i]] = 1.0

    df_mat_train = df_mat_train.append(pd.DataFrame({'filename':[filename],'class_label':[class_label],'class_one_hot':[class_one_hot]}), ignore_index=True)
zzz=1
for j in range(len(csi_test)):
    filename = 'csi_test_' + str(j+1250)
    class_label = 'label_'+str(csi_test_y[j])
    class_one_hot = np.zeros(10, dtype=np.float32)
    class_one_hot[csi_test_y[j]] = 1.0

    df_mat_test = df_mat_test.append(pd.DataFrame({'filename':[filename],'class_label':[class_label],'class_one_hot':[class_one_hot]}), ignore_index=True)

df_mat = pd.concat([df_mat_train, df_mat_test], ignore_index=True)
save_dataframe = 1
if save_dataframe:
    df_mat_train.to_pickle("data/df_mat_train_ori.pickle")
    df_mat_test.to_pickle('data/df_mat_test_ori.pickle')
    df_mat.to_pickle('data/df_mat_ori.pickle')