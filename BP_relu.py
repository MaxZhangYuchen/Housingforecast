import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('bj_housing.csv')
Y_true = data[['Value']]
X = data.drop('Value', axis = 1)
# Standardized treatment
scaler = MinMaxScaler()
X=scaler.fit_transform(X)
Y_true=scaler.fit_transform(Y_true)
# shuffle
np.random.seed(12)
np.random.shuffle(X)
np.random.seed(12)
np.random.shuffle(Y_true)
# Split data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_true, test_size=0.20, random_state=0)

#Z1=x*w1/Z2=A1*w2
def re_Z(w,x,b):
    Z=np.dot(x,w)+b
    return Z
def active(z):
    row=z.shape[0]
    col=z.shape[1]
    A = np.ones(shape=(row, col), dtype='float')
    for i in range(row):
        for j in range(col):
            if z[i,j]>0:
                A[i,j]=z[i,j]
            else:
                A[i,j]=0
    return A
def grad_active(z):
    row = z.shape[0]
    col = z.shape[1]
    grad_z = np.ones(shape=(row, col), dtype='float')
    for i in range(row):
        for j in range(col):
            if z[i,j]>0:
                grad_z[i,j]=z[i,j]
            else:
                grad_z[i,j]=0
    return grad_z

#Loss function
def Loss(w1,w2,b1,b2,x,y_true):
    N=x.shape[0]
    Z1=re_Z(w1,x,b1)
    A1=active(Z1)
    Z2=re_Z(w2,A1,b2)
    A2=Z2
    L=y_true-A2
    return (1/N)*np.dot(L.T,L)
#using gradient descent to update w and b
def gradient_w2(w1,w2,b1,b2,x,y_true,a=0.03):
    N=x.shape[0]
    L_total=0
    for i in range(N):
        n=x[i,:]
        Z1 = re_Z(w1, n.reshape((1,x.shape[1])),b1)
        A1 = active(Z1)
        Z2 = re_Z(w2, A1,b2)
        A2 = Z2
        L_total = L_total + (A2-y_true[i,0])*(A1.T)
    w2=w2-(2*a/N)*L_total
    return w2

def gradient_w1(w1,w2,b1,b2,x,y_true,a=0.03):
    N=x.shape[0]
    L2_total=0
    for i in range(N):
        n = x[i, :]
        Z1 = re_Z(w1, n.reshape((1,x.shape[1])),b1)
        A1 = active(Z1)
        Z2 = re_Z(w2, A1,b2)
        A2 = Z2
        L2_total = L2_total + (n.reshape((1,x.shape[1]))).\
            T*(grad_active((A2-y_true[i,0])*w2.T))
    w1=w1-(2*a/N)*L2_total
    return w1

def gradient_b1(w1,w2,b1,b2,x,y_true,a=0.03):
    N=x.shape[0]
    L_total=0
    for i in range(N):
        n=x[i,:]
        Z1 = re_Z(w1, n.reshape((1,x.shape[1])),b1)
        A1 = active(Z1)
        Z2 = re_Z(w2, A1,b2)
        A2 = Z2
        L_total = L_total + grad_active((((A2-y_true[i,0]))*w2.T))
    b1 = b1 - (2 * a / N) * L_total
    return b1

def gradient_b2(w1,w2,b1,b2,x,y_true,a=0.03):
    N=x.shape[0]
    L_total=0
    for i in range(N):
        n=x[i,:]
        Z1 = re_Z(w1, n.reshape((1,x.shape[1])),b1)
        A1 = active(Z1)
        Z2 = re_Z(w2, A1,b2)
        A2 = Z2
        L_total = L_total + ((A2-y_true[i,0]))
    b2=b2-(2*a/N)*L_total
    return b2
# find the optimal w and b
def gradient_descent(x,y_true,ER=1e-15, MAX_LOOP=40):
    N_feature=x.shape[1]
    w1=np.random.randn(N_feature,10)
    w2 = np.random.randn(10, 1)
    b1 = np.random.randn(1, 10)
    b2 = np.random.randn(1, 1)
    i = 0
    error = 1
    train_loss = []
    test_loss = []
    while error > ER and i < MAX_LOOP:
        y_now=Loss(w1,w2,b1,b2,x,y_true)
        w1_old=w1
        w2_old=w2
        b1_old=b1
        b2_old=b2
        w1=gradient_w1(w1_old,w2_old,b1_old,b2_old,x,y_true,a=0.03)
        w2=gradient_w2(w1_old,w2_old,b1_old,b2_old,x,y_true,a=0.03)
        b1=gradient_b1(w1_old,w2_old,b1_old,b2_old,x,y_true,a=0.03)
        b2=gradient_b2(w1_old,w2_old,b1_old,b2_old,x,y_true,a=0.03)
        y_next=Loss(w1,w2,b1,b2,x,y_true)
        er = abs(y_next-y_now)
        Loss_train = Loss(w1, w2, b1, b2, X_train, Y_train)
        Loss_test = Loss(w1, w2, b1, b2, X_test, Y_test)
        train_loss.append(Loss_train[0,0])
        test_loss.append(Loss_test[0,0])
        error=er[0,0]
        i+= 1
    print(i)
    return w1,w2,b1,b2,train_loss,test_loss

# use training set to get optimal w and b
a=gradient_descent(X_train,Y_train)
w1=a[0]
w2=a[1]
b1=a[2]
b2=a[3]
train_loss =a[4]
test_loss = a[5]
print('the optimal w1 is\n',w1,'\n the optimal w2 is\n',w2,'\nthe optimal b1 is\n',b1,'\n the optimal b2 is\n',b2)
# use the test set to test model's error
Loss_test=Loss(w1,w2,b1,b2,X_test,Y_test)
print('\n\nthe Loss of the test set is\n ',Loss_test)
x=list(range(len(train_loss)))
plt.plot(x,train_loss,'r-',label="train")
plt.plot(x,test_loss,'b-',label="test")
plt.legend()
plt.xlabel("i")
plt.ylabel('Loss')
plt.show()
