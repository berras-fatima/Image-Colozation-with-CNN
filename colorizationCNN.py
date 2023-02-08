import numpy as np 
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow import set_random_seed
import tensorflow as tf

tf.random.set_seed(123)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)
tf.random.set_seed(2)
np.random.seed(1)


ImagePath="/painting/"

img = cv2.imread(ImagePath+"1179.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
plt.imshow(img)
img.shape



HEIGHT=224
WIDTH=224

def ExtractInput(path):
    '''converitr les images en LAB
       Input : ensemble d'images
       Output : un vecteur de luminance, un vecteur de valeur ab
    '''
    X_img=[]
    y_img=[]
    for imageDir in os.listdir(ImagePath):
        try:
            img = cv2.imread(ImagePath + imageDir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
            img = img.astype(np.float32)
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            
            img_lab_rs = cv2.resize(img_lab, (WIDTH, HEIGHT)) 
            img_l = img_lab_rs[:,:,0] 
            
            img_ab = img_lab_rs[:,:,1:]
            img_ab = img_ab/128
            
            X_img.append(img_l)
            y_img.append(img_ab)
        except:
            pass
    X_img = np.array(X_img)
    y_img = np.array(y_img)
    
    return X_img,y_img



X_,y_ = ExtractInput(ImagePath)



K.clear_session()
def InstantiateModel(in_):
    model_ = Conv2D(32,(3,3),padding='same',strides=1)(in_)
    model_ = LeakyReLU()(model_)
    
    model_ = Conv2D(64,(3,3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2,2),padding='same')(model_)
    
    model_ = Conv2D(128,(3,3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2,2),padding='same')(model_)
    
    model_ = Conv2D(256,(3,3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = Conv2D(512,(3,3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(256,(3,3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(128,(3,3), padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    
    concat_ = concatenate([model_, in_]) 
    
    model_ = Conv2D(128,(3,3), padding='same',strides=1)(concat_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = Conv2D(64,(3,3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    
    model_ = Conv2D(2,(3,3), activation='tanh',padding='same',strides=1)(model_)

    return model_



Input_Sample = Input(shape=(HEIGHT, WIDTH,1))
Output_ = InstantiateModel(Input_Sample)
Model_Colourization = Model(inputs=Input_Sample, outputs=Output_)



LEARNING_RATE = 0.001
Model_Colourization.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
Model_Colourization.summary()



def GenerateInputs(X_,y_):
    for i in range(len(X_)):
        X_input = X_[i].reshape(1,224,224,1)
        y_input = y_[i].reshape(1,224,224,2)
        yield (X_input,y_input)
Model_Colourization.fit_generator(GenerateInputs(X_,y_),epochs=53,verbose=1,steps_per_epoch=38,shuffle=True) #80



tf.keras.utils.plot_model(Model_Colourization, "Model_Colourization.png")



image = cv2.imread('/image_test/image05.png')
plt.imshow(image)

img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_ = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2Lab)
img_=img_.astype(np.float32)
img_lab_rs = cv2.resize(img_, (WIDTH, HEIGHT)) 
img_l = img_lab_rs[:,:,0] 
img_l_reshaped = img_l.reshape(1,224,224,1)
Prediction = Model_Colourization.predict(img_l_reshaped)
Prediction = Prediction*128
Prediction=Prediction.reshape(224,224,2)

plt.figure(figsize=(30,20))
plt.subplot(5,5,1+1)

img_1 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.resize(img, (224, 224))
plt.title("Grey Image")
plt.imshow(img)

plt.subplot(5,5,1+2)
img_ = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
img_[:,:,1:] = Prediction
img_ = cv2.cvtColor(img_, cv2.COLOR_Lab2RGB)
plt.title("Predicted Image")
plt.imshow(img_)

img_1 = cv2.resize(img_1, (WIDTH, HEIGHT))
plt.subplot(5,5,1)
plt.title("Ground truth")
plt.imshow(img_1)

