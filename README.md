# WiFi CSI-based activity recognition using Contrastive Self-Supervised Learning


This is the PyTorch source code for the WiFi CSI-based activity recognition. 
The code runs on Python 3. 
Install the dependencies and prepare the datasets with the following commands:



## Dataset


The two public datasets used in the paper are shown below.


### DeepSeg Dataset


The data that we extract from raw CSI data for our experiments can be downloaded from Baidu Netdisk or Google Drive:


Data of CSI amplitudes: Data_CsiAmplitudeCut Baidu Netdisk: https://pan.baidu.com/s/12DwlT58PzlVAyBc-lYx1lw (Password: k8yp) 
or Google Drive: https://drive.google.com/drive/folders/1PLzV6ZWAauMQLf08NUkd5UeKrqyGMHgv


Manually marked Labels for CSI amplitude data: Label_CsiAmplitudeCut Baidu: https://pan.baidu.com/s/1nY5Og4NlLb7VH5oBQ-LH9w (Password: xnra) 
or Google: https://drive.google.com/drive/folders/1855zX-93QjmAt2wSeJk0rTJRiPaFMGBd (1 boxing; 2 hand swing; 3 picking up; 4 hand raising; 5 running; 6 pushing; 7 squatting; 8 drawing O; 9 walking; 10 drawing X)



Also the raw CSI data we collected can be downloaded via Baidu or Google: Data_RawCSIDat. Note that there is no need to download the raw CSI data for running our experiments. Downloading Data_CsiAmplitudeCut and Label_CsiAmplitudeCut is enough for our experiments. Baidu: https://pan.baidu.com/s/1FpA2u_fzFIh4FuNIcWOPdQ (Password: hhcv) or Google: https://drive.google.com/drive/folders/1vUeJYChsDgBzv7bJbiKDEfAHQje3SW9G




### SignFi Dataset

The SignFi dataset comes from the link below: https://github.com/yongsen/SignFi


## Requirement

Python 3.7

Tensorflow 2.4.1

The codes are tested under window10.


## Folder descriptions:

*01DataProcessing*: This is used to extract the data in CSI format from the original WiFi and convert it into PNG format in order to make better use of the data.


*02DataGenerator*: This is used to generate augmented samples based on the source data.


*03ActivityRecognition*: This is used to conduct activity recognition.














