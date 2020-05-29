import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from time import sleep
warnings.filterwarnings("ignore")
plt.style.use("ggplot")
from IPython import display

def Predict(p, theta):
    x = np.insert(p, 0, 1, axis=0)
    return x.dot(theta)

def LinearRegression(X, y,lr = 0.00001):
    fig = plt.figure(figsize=(20,10),dpi=200)
    m = y.size
    old_error = 10
    error = 0
    counter = 0
    x = np.insert(X, 0, 1, axis=1)
    theta = np.random.randn(len(x[0])) + np.array([0,0])
    while True:
        counter+=1
        h = x @ theta
        error = ((y - h)**2).mean()
        theta = theta - lr * 1 / m * (x.T.dot(h - y))
        if abs(old_error-error)<0.01:
            break
        old_error = error
        x_min,x_max = X.min(),X.max()
        display.clear_output(wait=True)
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        sc = ax.scatter(X,y,s=100)
        pl = ax.plot([x_min,x_max], [Predict([x_min], theta),Predict([x_max], theta)],c="k")
       
        display.display(plt.gcf())
        print(f"Iteration {counter} : Error : {error} weight: {theta}")
    display.clear_output(wait=True)
    print(f"Iteration {counter} : Error : {error}")
    return theta