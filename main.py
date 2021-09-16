import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import math

"""**Pregunta 1**"""

size = 1000
x = np.linspace(0, 1, num=size) 
y = [ np.sin(i*2*3.14)  + np.random.normal(0,0.1) for i  in x]
plt.plot(x,y,'*')

"""Funcion del modelo"""

"""**Pregunta 2**"""

x_train, x_rest, y_train, y_rest = train_test_split(x,y,train_size=0.7, shuffle=True)
x_valid, x_test, y_valid, y_test = train_test_split(x_rest,y_rest,test_size=0.67)

"""**Pregunta 3**"""

def h(x, w, b):
    sum_value = 0
    for i in range(len(w)):
        sum_value += (w[i] * pow(x, i))
    return sum_value + b
  #return np.dot(x, np.transpose(w)) + b

def derivada(y,x,b,w,p):
  sum1 = 0
  sum2 = 0
  dw_temp = []
  for i in range(len(x)):
    sum1 += (y[i] - h(x,w,b)) * (-1)
  
  for j in range(p):
    for i in range(len(x)):
      sum2 += (y[i] - h(x,w,b)) * pow(-x[i], j+1)
      dw_temp.append(sum2)
  
  db = sum1 / len(x)
  dw = [i/len(x) for i in dw_temp]
  return dw, db

def update(w,b,dw,db,alpha,p):
  w = [w[j] - alpha*dw[j] for j in range(p)]
  b -= alpha*db
  return w, b

def error_without_reg(y,x,b,w):
  sum1 = 0
  for i in range(len(x)):
    sum1 += pow(y[i] - h(x,w,b), 2)

  sum1 /= 2*len(x)
  return sum1

def error_with_reg(y,x,w,b,lamda_value,p):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
      sum1 += pow(y[i] - h(x,w,b), 2)

    for j in range(len(p)):
      sum2 += pow(w[j], 2)

    sum1 /= 2*len(x)
    sum2 /= len(x)
    return sum1+lamda_value*sum2

def algorithm1(alpha,p,epochs):
  error_list = []
  w = np.linspace(0, 1, num = p)
  b = np.random.rand()
  for i in range(epochs):
      dw, db = derivada(y,x_train,b,w,p)
      w, b = update(w,b,dw,db,alpha,p)
      e = error_without_reg(y,x,b,w)
      error_list.append(e)
      if i > 20000:
          break

"""**Pregunta 4**"""

def algorithm2(alpha,p,epochs,has_reg,lamda_value):
  training = []
  validation = []
  w = np.linspace(0, 1, num = p)
  b = np.random.rand()
  for i in range(epochs):
      dw, db = derivada(y,x_train,b,w,p)
      w, b = update(w,b,dw,db,alpha,p)
      if has_reg is False:
        temp_train = error_without_reg(y,x_train,b,w)
        temp_val = error_without_reg(y,x_valid,b,w)
      else:
        temp_train = error_with_reg(y,x_train,b,w,lamda_value,p)
        temp_val = error_with_reg(y,x_valid,b,w,lamda_value,p)
      training.append(temp_train)
      validation.append(temp_val)
      if i > 20000:
          break
  return training,validation

"""**Pregunta 5**"""

def algorithm3():
  training,validation = algorithm2(0.007,5,40,False,0)
  plt.plot(training, y, '*')
  plt.plot(validation, y, '+')

#Overfitting -> Pierde la capacidad de finalizacion

"""**Pregunta 6**"""

def algorithm4():
  for i in range(10):
    alpha = np.linspace(0, 1)
    training,validation = algorithm2(alpha,5,50,False,0)

    plt.plot(training, y, '*')
    plt.plot(validation, y, '+')

"""**Pregunta 7**"""

def algorithm5(alpha):
  for i in range(10):
    lamda_value = np.linspace(0, 1)
    training,validation = algorithm2(alpha,5,50,True,lamda_value)

    plt.plot(training, y, '*')
    plt.plot(validation, y, '+')

"""**Pregunta 8**"""

"""**Pregunta 9**"""

def error_test(y, x, b, w):
    sum_value = 0
    for i in range(len(x)):
        sum_value += pow(y[i] - h(x, w, b), 2)
    sum_value /= 2*len(x)
    return sum_value

def testing(alpha,grado,epochs,p):
    w = np.linspace(0, 1, num = grado)
    b = np.random.rand()
    testing = []
    for i in range(epochs):
        dw, db = derivada(y,x_train,b,w,p)
        w, b = update(w, b, dw, db, alpha,p)
        temp = error_test(y, x_test, b, w)
        testing.append(temp)
        if i > 20000:
            break

t = testing(0.006, 5, 100)
plt.plot(t, y, '*')