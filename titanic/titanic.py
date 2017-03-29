import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("train.csv", sep=",", header=1)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,2:11]
Y = dataset[:,1]

newX = []
for col in X.T: #transpose to work on columns
  newcol = []
  notstring = len(filter(lambda x: type(x).__name__ == 'int' or type(x).__name__ == 'float', col)) != 0
  if not notstring:
    for item in col:
      newcol.append(id(intern(item)))
  else:
    newcol = col
  print newX
  newX.append(newcol)

# create model
model = Sequential()
model.add(Dense(100, input_dim=9, init='normal', activation='relu'))
# model.add(Dense(100, input_dim=35, init='normal', activation='relu'))
# model.add(Dense(100, input_dim=35, init='normal', activation='relu'))
# model.add(Dense(200, input_dim=35, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))
# Compile model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')
# model.compile(loss='msle', optimizer='adam')

seed = 7
numpy.random.seed(seed)
model.fit(X,Y, nb_epoch=20000, batch_size=1400)
# model.save('zemodel')

dataframe = pandas.read_csv("thetest.csv", sep=",", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X2 = dataset[:,0:35]
print X2

kfold = KFold(n_splits=10, random_state=seed)
results = model.predict(X2) #, Y2)
print type(results)
numpy.savetxt('theresult', results.astype(int),  fmt='%i')
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#print results.mean()**0.5
