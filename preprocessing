import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.preprocessing import OneHotEncoder

import pickle 
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Sequential 	
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

pd.set_option('display.max_columns', None)

train = pd.read_csv('/Users/sahme/Desktop/Collection/Sublime Files/kaggle datasets/IEEE comp/Train.csv')
test = pd.read_csv('/Users/sahme/Desktop/Collection/Sublime Files/kaggle datasets/IEEE comp/Test.csv')


# print("Shape of Dataset:", train.shape)
# print("Shape of Dataset:", test.shape)
# print(train.head(5))
# print(test.head(5))
# print(train)
# print(test.shape)

# print(len(train["F28"].unique()))
# for i in train:
# 	print(i, len(train[i].unique()))

# print(train.dtypes)

train = shuffle(train)

train.reset_index(inplace=True, drop=True)
# test.reset_index(inplace=True, drop=True)

X_test = test.iloc[0:, 0:42]
X_train = train.iloc[0:, 0:42]


# print(train["Label1"].value_counts()/train.shape[0])
# print(train["Label2"].value_counts()/train.shape[0])

# print(test["Label1"].value_counts()/test.shape[0])
# print(test["Label2"].value_counts()/test.shape[0])


# print(train.isnull().sum()/train.shape[0])

# sns.set(style = 'whitegrid')
# ax = sns.barplot(x = train["Label2"], y = train["F12"], data = train)
# plt.show()

pop = train.pop('Label2') #removing label2 then we will replace it to end of df

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(train[['Label1']]).toarray())
train = train.join(enc_df)

train = train.drop('Label1', 1) 
train['Label2'] = pop 


train = train.rename(columns={'Label2': 'Eleven'})
train = train.rename(columns={0: 'One'})
train = train.rename(columns={1: 'Two'})
train = train.rename(columns={2: 'Three'})
train = train.rename(columns={3: 'Four'})
train = train.rename(columns={4: 'Five'})
train = train.rename(columns={5: 'Six'})
train = train.rename(columns={6: 'Seven'})
train = train.rename(columns={7: 'Eight'})
train = train.rename(columns={8: 'Nine'})
train = train.rename(columns={9: 'Ten'})


Y_train = train.iloc[0:, 42:]
# Y_train = np.array(Y_train)

Y_test = test.iloc[0:, 42:]
# Y_test = np.array(Y_test)

pickle_out = open("X_train", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("Y_train", "wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("Y_test", "wb")
pickle.dump(Y_test, pickle_out)
pickle_out.close()


print('Ytrain shape:', Y_train.shape)
print('Xtrain data shape:', X_train.shape)

print('Ytest shape:', Y_test.shape)
print('Xtest data shape:', X_test.shape)


