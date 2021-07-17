import pickle 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.metrics
from tensorflow.keras.models import Sequential 	
from tensorflow.keras.layers import Activation, Dropout




pickle_in = open('X_train', 'rb')
X_train = pickle.load(pickle_in)

pickle_in = open('Y_train', 'rb')
Y_train = pickle.load(pickle_in)

pickle_in = open('X_test', 'rb')
X_test = pickle.load(pickle_in)

pickle_in = open('Y_test', 'rb')
Y_test = pickle.load(pickle_in)


print(X_train.shape)
Y_train_1 = Y_train.iloc[0:,0:10]
Y_train_2 = Y_train.iloc[0:,10:11]



inputs  = keras.Input(shape = 42, name = 'input')


dense1 = keras.layers.Dense(1024, input_dim=42, activation='relu')
# drop1 = keras.layers.Dropout(0.9)

dense2 = keras.layers.Dense(512, activation='relu')

dense3 = keras.layers.Dense(256, activation='relu')
# drop2 = keras.layers.Dropout(0.9)
 
dense4 = keras.layers.Dense(128, activation='relu')
# drop3 = keras.layers.Dropout(0.7)

dense5 = keras.layers.Dense(64, activation='relu')

 
dense6_1 = keras.layers.Dense(10, activation = 'softmax', name="10class")
dense6_2 = keras.layers.Dense(1, activation = 'sigmoid', name="1class")


x = inputs
x = dense1(x)
# x = drop1(x)

x = dense2(x)

x = dense3(x)
# x = drop2(x)

x = dense4(x)
# x = drop3(x)

x = dense5(x)
output_1= dense6_1(x)
output_2 = dense6_2(x)

model = keras.Model(inputs = inputs, outputs = [output_1,output_2], name = 'model')

loss1 = keras.losses.CategoricalCrossentropy(from_logits = False)
loss2 = keras.losses.BinaryCrossentropy(from_logits = False)

optimizer = keras.optimizers.Adam(learning_rate=0.000001)
model.compile(loss={"10class": loss1, 
                    "1class":loss2}, 
                   optimizer='adam', metrics=["accuracy"])

model.fit({'input': X_train},
         {'10class':Y_train_1, '1class':Y_train_2},
                 validation_split=0.2, 
                       epochs=3, 
                            batch_size=64)
print(model.summary())

model.save('Multilabel Model Version-2')
