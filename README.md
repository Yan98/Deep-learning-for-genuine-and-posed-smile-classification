# RealSmileNet pytorch Implementation
##Dependency 
* python 3.7
* numpy 1.18.1 
* Pillow 7.0.0 
* dlib 19.19.0
* opencv-python 4.2.0.32
* torch 1.4.0 
* torchvision 0.5.0 

Note, The current version of code only tested on MacOS.

## Dataset
* obtain the whole UVA-NEMO database from https://www.uva-nemo.org
* obtain the whole MMI database from https://mmifacedb.eu
* obtain the whole BBC database from https://www.bbc.co.uk/science/humanbody/mind/surveys/smiles/
* obtain the whole SPOS database from https://www.oulu.fi/cmvs/node/41317

The sample data is contained in processed_data folder


## Train RealSmileNet
test.py contains the test program for deep smile net. We do not provide the test program for model performance as it is time-consuming. Only the basic test are provided. run python test.py 

##########################################################################################
##########################################################################################

run python demo.py for the sample training of UVA-NEMO databases, the verbose.txt will contains the log of training. Note only one public data from UVA-NEMO are provided.

The model weight will be saved in database_training folder, for example if the model is trained on UVA-NEMO database, the folder names uvanemo_training

demo.py is also works as the test program, run python demo.py

usage: demo.py [-h] [--database DATABASE] [--batch_size BATCH_SIZE]
               [--label_path LABEL_PATH] [--frame_path FRAME_PATH]
               [--frequency FREQUENCY] [--sub SUB] [--epochs EPOCHS] [--lr LR]

DeepSmileNet training, optional arguments for demo.py :
  -h, --help            show this help message and exit
  --database DATABASE   select the database to run, please selected from
                        UVANEMO,SPOS,MMI and BBC
  --batch_size BATCH_SIZE
                        the mini batch size used for training
  --label_path LABEL_PATH
                        the path contains training labels
  --frame_path FRAME_PATH
                        the path contains processed data
  --frequency FREQUENCY
                        the frequency used to sample the data
  --sub SUB             the subsitution for the model, please selected from
                        org,LSTM,GRU,resnet,miniAlexnet,minidensenet
  --epochs EPOCHS       number of total epochs to run
  --lr LR, --learning-rate LR
                        learning rate

##########################################################################################
##########################################################################################

data_preprocessin.py contains the function used to preprocess UVA-NEMO,BBC,SPOS and MMI database. The pre trained shape_predictor_68_face_landmarks model can be downloaded from http://dlib.net/files/ 

##########################################################################################
##########################################################################################

model.py contains the implementation of different model structures.
The implementation of NonLocalBlock is from https://github.com/AlexHex7/Non-local_pytorch/tree/master/lib

The implementation of ConvLSTM is based on https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py 

The implmentation of resnet is based on  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py and https://github.com/zhunzhong07/Random-Erasing/blob/master/models/cifar/resnet.py 

The implementation of AlexNet is based on https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py    

#The implementation of DenseNet is based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py        
##########################################################################################
##########################################################################################

