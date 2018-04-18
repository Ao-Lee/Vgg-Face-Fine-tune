
# what content does this project contain
1. face detection and alignment
![Alt text](https://github.com/Ao-Lee/Vgg-Face-Fine-tune/raw/master/TestCases/imgs/detection/_01.jpg)

2. face verification predictions
![Alt text](https://github.com/Ao-Lee/Vgg-Face-Fine-tune/raw/master/TestCases/imgs/verification/example.png)

3. fine tune a pre-trained model with your customized dataset

# how it works
the network used in this project is vgg16 and it was pre-trained by Oxford to classify 2622 identities. Check [this paper](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) for more details. Since their model was trained for a classification purpose, it must be
tuned to fit verification tasks. A triplet loss is used in this project just as what the paper describes. For more details about triplet loss, check [this](https://arxiv.org/abs/1503.03832) Google facenet paper

# how to install
1. install anaconda for python 3
2. install tensorflow
3. install keras (theno backend is not tested)
4. install other python packages such as tqdm

# how to perform face detection and alignment
run TestCases\test_align_database.py

# Datasets used 
different face datasets were used in this project, they are
LFW dataset (http://vis-www.cs.umass.edu/lfw/)
AR dataset (http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html)
CAS-PEAL dataset (http://www.jdl.ac.cn/peal/)

# how to perform verification
any face dataset can be used, since lfw data set is easy to download and prepare, we use lfw dataset for illustration
1. download lfw dataset. Since we have our own align implementation. it is recommended to download data without alignment. the quick link is [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
2. align the lfw database. Open TestCases\test_align_database.py and edit the following two lines
```
source = 'imgs\\align_test\\origin'
target = 'imgs\\align_test\\aligned'
```
run test_align_database.py after running, you should get a folder of aligned images.

3. download the model:
	* todo: provide download link
4. open cfg.py and modify the '_dir_models' variable, it changes the absolute path to the model
5. open TestCases\test_verification_on_general_dataset.py and change dir_image to be the root directory of aligned images
6.run test_verification_on_general_dataset.py

# how to train the network
- we still use lfw dataset to illustrate the process, in practice, any face dataset can be used
- note that it is recommended to use a GPU to train. Training on a cpu machine could be very slow.

1. open train.py and modify this line:
	reader_LFW = LFWReader(dir_images='E:\\DM\\Faces\\Data\\LFW\\aligned')
	set the directory to be the absolute path of aligned lfw images
2. run the train.py code

