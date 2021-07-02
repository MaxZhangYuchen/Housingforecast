import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('bj_housing.csv')
Y_true = data[['Value']]
X = data.drop('Value', axis = 1)
minimum_price = np.min(Y_true)
maximum_price = np.max(Y_true)
# Standardized treatment
scaler = MinMaxScaler()
X=scaler.fit_transform(X)
Y_true=scaler.fit_transform(Y_true)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_true, test_size=0.20, random_state=0)

#Loss function
def Loss(w,b,x,y_true):
    N=x.shape[0]
    L=y_true-b-np.dot(x,w)
    return (1/N)*np.dot(L.T,L)
#using gradient descent to update w and b
def gradient_b(w,x,b,y_true,a=0.1):
    N=x.shape[0]
    L=np.dot(x,w)+b-y_true
    L_total=0
    for i in range(N):
        L_total=L_total+L[i]
    b_new=b-(2*a/N)*L_total
    return b_new
def gradient_w(w,x,b,y_true,a=0.1):
    N1=w.shape[0]
    N=x.shape[0]
    L=np.dot(x,w)+b-y_true
    for i in range(N1):
        w[i,0]=w[i,0]-(2*a/N)*np.dot(L.T,x[:,i])
    return w

# find the optimal w and b
def gradient_descent(x,y_true,ER=1e-12, MAX_LOOP=3e3):
    N_feature=x.shape[1]
    w=np.ones(shape=(6,1),dtype='float')
    b= np.array([0])
    i = 0
    error = 1
    train_loss = []
    test_loss = []
    while error > ER and i < MAX_LOOP:
        y_now=Loss(w,b,x,y_true)
        w=gradient_w(w,x,b,y_true,a=0.1)
        b=gradient_b(w,x,b,y_true,a=0.1)
        y_next=Loss(w,b,x,y_true)
        er = abs(y_next-y_now)
        Loss_train = Loss(w, b,  X_train, Y_train)
        Loss_test = Loss(w, b, X_test, Y_test)
        train_loss.append(Loss_train[0,0])
        test_loss.append(Loss_test[0,0])
        error=er[0,0]
        i+= 1
    return w,b,train_loss,test_loss

# use training set to get optimal w and b
a=gradient_descent(X_train,Y_train)
w=a[0]
b=a[1]
train_loss =a[2]
test_loss = a[3]
print('the optimal w is\n',w,'\n the optimal b is\n',b)
# use the test set to test model's error
Loss_train=Loss(w,b,X_train,Y_train)
Loss_test=Loss(w,b,X_test,Y_test)
print('\nthe Loss of the test set is\n ',Loss_test)
x=list(range(len(train_loss)))
plt.plot(x,train_loss,'r-',label="train")
plt.plot(x,test_loss,'b-',label="test")
plt.legend()
plt.xlabel("i")
plt.ylabel('Loss')
plt.show()