#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from model import *
from data import *

import pandas as pd
import os
from tqdm import tqdm
import shutil
import random
import matplotlib.pyplot as plt
import cv2
import pydicom

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

DATA_DIR = "/data/krf/dataset/ChinaSet_AllFiles"
ROOT_DIR = "/data/krf/model/unet"
LABEL_DIR = "/data/krf/dataset/"
TEST_DIR = "/data/krf/dataset/stage_1_test_images/"
MASK_DIR = os.path.join(LABEL_DIR,"stage_1_train_masks")
CASE_DIR = "/data/krf/dataset/stage_1_train_images/"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,DATA_DIR,'CXR_png','mask',data_gen_args,save_to_dir = None)
valGene = trainGenerator(2,DATA_DIR,'val','val_mask',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_lung_1024.hdf5',monitor='val_loss',verbose=1, save_best_only=True)

history = model.fit_generator(myGene,steps_per_epoch=320,epochs=100,validation_data = valGene,validation_steps = 160,callbacks=[model_checkpoint])

import matplotlib.pyplot as plt

print(history.history.keys())

plt.figure()
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
#plt.show()
plt.savefig(ROOT_DIR+"/history.png")

# In[ ]:
plt.figure()
plt.plot(history.history['dice'], 'orange', label='Dice')
plt.plot(history.history['iou'], 'blue', label='IOU')
plt.legend()
#plt.show()
plt.savefig(ROOT_DIR+"/dice_iou.png")

def saveResult2(test_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
    f_list = os.listdir(test_path)
    f_list.sort()
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,f_list[i].split('.')[0]+".png"),img)
        

def testGenerator2(test_path,num_image,target_size = (1024,1024),flag_multi_class = False,as_gray = True):
    f_list = os.listdir(test_path)
    f_list.sort()
    for f in f_list[:num_image]:
        #img = io.imread(os.path.join(test_path,f),as_gray = as_gray)
        img = pydicom.read_file(os.path.join(test_path,f)).pixel_array
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
def geneSeg(testGene,mskGene,num,msk_dir,result_dir):
    f_list = os.listdir(msk_dir)
    f_list.sort()
    for i in range(num):
        img = testGene.__next__()
        msk = mskGene.__next__()
        croped = cv2.bitwise_and(img[0,:,:,0],img[0,:,:,0],mask=msk[0,:,:,0].astype(np.uint8))
        #seg = labelVisualize(2,COLOR_DICT,croped)
        io.imsave(os.path.join(result_dir,f_list[i]),croped)
#         print("img")
#         for i in range(256):
#             print(img[0,i,:,0])
#         print("msk")
#         for i in range(256):
#             print(msk[0,i,:,0])
#         print ("crop")
#         for i in range(256):
#             print(croped[i,:])   


testGene2 = testGenerator2(CASE_DIR,len(os.listdir(CASE_DIR)))
model2 = unet()
model2.load_weights("unet_lung_1024.hdf5")
results2 = model2.predict_generator(testGene2,len(os.listdir(CASE_DIR)),verbose=1)
results4 = results2>0.5
saveResult2(CASE_DIR,LABEL_DIR+"/stage_1_train_masks",results4.astype(np.uint8)*255)
# In[ ]:
num_img = len(os.listdir(MASK_DIR))
#num_img = 1000
mskGene = testGenerator(MASK_DIR,num_img)
trainGene = testGenerator2(CASE_DIR,num_img)
geneSeg(trainGene,mskGene,num_img,MASK_DIR,LABEL_DIR+"/stage_1_train_segmentations") 
#saveResult(CASE_DIR,"/data/krf/dataset/stage_1_train_masks/",results2)


# In[ ]:


#TEST_DIR = "/data/krf/dataset/stage_1_test_images/"
#testGene3 = testGenerator2(TEST_DIR,1000)
#model3 = unet()
#model3.load_weights("unet_lung.hdf5")
#results3 = model3.predict_generator(testGene3,1000,verbose=1)


# In[ ]:
testGene3 = testGenerator2(TEST_DIR,1000)
model3 = unet()
model3.load_weights("unet_lung_1024.hdf5")
results3 = model3.predict_generator(testGene3,1000,verbose=1)
results5 = results3>0.5
saveResult2(TEST_DIR,LABEL_DIR+"/stage_1_test_masks",results5.astype(np.uint8)*255)
#num_img = len(os.listdir(MASK_DIR))

num_img = 1000
mskGene = testGenerator(LABEL_DIR+"/stage_1_test_masks",num_img)
trainGene = testGenerator2(TEST_DIR,num_img)
geneSeg(trainGene,mskGene,num_img,LABEL_DIR+"/stage_1_test_masks",LABEL_DIR+"/stage_1_test_segmentations") 
#saveResult(TEST_DIR,"/data/krf/dataset/stage_1_test_masks/",results3)

