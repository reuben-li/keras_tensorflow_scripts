import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("input3.csv", sep=",", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:35]
Y = dataset[:,35]

# create model
model = Sequential()
model.add(Dense(35, input_dim=35, init='normal', activation='relu'))
model.add(Dense(100, input_dim=35, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

seed = 7
numpy.random.seed(seed)
model.fit(X,Y, nb_epoch=500, batch_size=5)

dataframe = pandas.read_csv("thetest.csv", sep=",", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X2 = dataset[:,0:35]
print X2

results = model.predict(X2)
print type(results)
numpy.savetxt('theresult', results.astype(int),  fmt='%i')
