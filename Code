import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
# path = "/content/Healthcare_Investments_and_Hospital_Stay.csv"
path = '/content/drive/MyDrive/Healthcare_Investments_and_Hospital_Stay.csv'
data = pd.read_csv(path)
data.head(3)
data.describe()
data.info()
data.columns

x = data[['Time','MRI_Units','CT_Scanners','Hospital_Beds']]
y = data['Hospital_Stay']

#This is model zero, a simplified neural network
from sklearn.model_selection import train_test_split
import tensorflow as tf
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train = np.array(x_train)
y_train = np.array(y_train).reshape(-1,1)
x_test = np.array(x_test)
y_test = np.array(y_test).reshape(-1,1)

model_0 = tf.keras.models.Sequential([
     tf.keras.layers.Dense(30, activation=tf.nn.relu), # 30 neurons in this layer, activation is ReLU(x) = max(x, 0)
     tf.keras.layers.Dense(1) # output layer -> one neuron to get the final predicted price
  ]) 

model_0.compile(optimizer = "adam",
              loss = 'mse')

history_0 = model_0.fit(
    x_train, y_train, epochs=10,
    validation_data=(x_test, y_test)
)
mse_test = model_0.evaluate(x_test, y_test)
rmse_test = mse_test**(0.5)

print(f'The lowest root mean square error for model_0 is {rmse_test}')

import numpy as np
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error
multiple_regression_MSE = mean_squared_error(y_test,lm.predict(x_test))
print(np.sqrt(multiple_regression_MSE))
