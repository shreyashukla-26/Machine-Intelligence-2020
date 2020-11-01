'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class NN:

	''' X and Y are dataframes '''

	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''

	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values

		yhat is a list of the predicted value for df X
		"""

		return yhat

	def CM(self,y_test,y_test_obs):
		'''
		Prints confusion matrix
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0

		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0

		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)

		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")

# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

if __name__ == '__main__':
    obj = NN()

    # Deleting rows with empty values
    df = pd.read_csv("LBW_Dataset.csv")
    df = df.dropna(how='any',axis=0)

    # Train - Test
    x = df[['Community','Age','Weight','Delivery phase','HB','IFA','BP','Education','Residence']]
    y = df[['Result']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Converting to numpy
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    #weights = np.array([[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.2]])
    #weights = 2*np.random.random((9,1)) - 1

    # Weights after 100000 iterations
    #weights = np.array([[ 144.07851781],[ -18.88111445],[ 211.55721058],[ -71.68262917],[-577.34797282],[ -64.94507701],[-152.23350256],[-358.13914267],[  35.82872154]])

    # Weigts after 200000 iterations
    #weights = np.array([[ 144.64306327], [ -11.4906438 ], [ 220.50090421], [ -71.66080466], [-576.99989013], [ -64.9232525 ], [-157.01042614], [-358.03002011], [  36.12190653]])

    # Weigts after 300000 iterations
    weights = np.array([[ 144.93201834], [ -10.04869737], [ 221.88890137], [ -71.78024112], [-577.95851786], [ -65.04268896], [-161.98620921], [-358.62720239], [  36.20666584]])

    bias = -0.5
    learning_rate = 0.2

    #prev_weights = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    iteration = 0;

    # Perceptron
    #while(not np.array_equal(weights, prev_weights)):
    while(iteration<100):

        pred = np.dot(x_train,weights) + bias
        pred = sigmoid(pred)
        #print(pred)

        # error numoy array
        error = y_train - pred

        # Backpropogation for each feature
        #for i in range(len(pred)):
        #delta_w = learning_rate * (y_train[i] - pred[i]) * pred[i] * (1 - pred[i]) *
        delta  = error * sigmoid_der(pred)

        #update weights
        #prev_weights = weights
        weights = weights + np.dot(x_train.T,delta)

        iteration+=1;

    # Test on test data
    test_pred = sigmoid(np.dot(x_test,weights)+bias)
    for i in range(len(test_pred)):
        if(test_pred[i]<=0.6):
            test_pred[i] = 0
        else:
            test_pred[i] = 1


    obj.CM(y_test,test_pred)
