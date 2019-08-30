import tensorflow 
import keras 
import pandas as pd 
import numpy as np



xt=np.load('imerge_training.npy')
yt=np.load('ymerge_training.npy')


pt=[]
for i in range(len(yt)):
    pt.append(round(1/yt[i],3))
pt=np.array(pt)

from sklearn.model_selection import train_test_split
X_train1, X_test, y_train1, y_test = train_test_split(xt, pt, test_size=0.15)


from keras import layers
from keras import models
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv2D, MaxPooling2D

model = models.Sequential()
model.add(layers.Conv2D(128, (5,5), padding='same', activation='relu', input_shape=(39, 56,1)))
model.add(Dropout(0.5))
model.add(layers.Conv2D(64, (4, 4), activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(4, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0), loss='mse', metrics=['mae'])
model.summary()

from matplotlib import pyplot
history=model.fit(X_train1, y_train1, epochs=30, batch_size=256, validation_split=0.33)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

y_pred = model.predict(X_test)

err_arr = np.zeros((y_test.shape[0],))
for i in range(y_test.shape[0]):
    err_arr[i] = (y_pred[i] - y_test[i])/y_test[i] #changed 
    
def ind_ptrange(y, ptmin, ptmax):
    ooptmin = 1./ptmax
    ooptmax = 1./ptmin
    return np.where((y>ooptmin) & (y<ooptmax))[0]

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaus(x,a,mu,sig):
    return a*np.exp(-((x-mu)**2)/(2*sig**2))

#pt1 = 25
ind2 = ind_ptrange(y_test, ptmin =2, ptmax = 100)
y_test2 = y_test[ind2]
y_pred2 = y_pred[ind2]
err_arr2 = np.zeros((y_test2.shape[0],))
for i in range(y_test2.shape[0]):
    err_arr2[i] = (y_pred2[i] - y_test2[i])/y_test2[i] #changed
nbin = 40
bin2 = np.linspace(-0.5,0.5,nbin)
h1,h2 = np.histogram(err_arr2,bins=bin2)
#popt, pcov = curve_fit(gaus,h2[:-1],h1)
plt.hist(err_arr2, nbin,range=(-0.5,0.5),facecolor='g')
#plt.plot((h2[1:]+h2[:-1])/2.,gaus(h2[:-1],*popt), linewidth=3, color='red',label='fit: a=%5.3f, mu=%5.3f, sig=%5.3f'%tuple(popt))
#plt.legend()
#plt.text(-0.55, 1200, '$p_T$ @ {0}GeV'.format(pt1), fontsize=15)
#plt.text(0.55,1200,'Work in Progress', fontsize=15,horizontalalignment='right',verticalalignment='bottom')
plt.ylabel('# of events / 0.0125 ', fontsize=15)
plt.xlabel(r'$({p_T}_{pred} - {p_T}_{true})/{p_T}_{pred}$', fontsize=20)