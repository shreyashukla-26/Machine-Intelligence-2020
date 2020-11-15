'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

INPUT_LAYER_SIZE = 9
HIDDEN_LAYER_1_SIZE = 5
HIDDEN_LAYER_2_SIZE = 5
OUTPUT_LAYER_SIZE = 1

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

	def CM(y_test,y_test_obs):
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


''' Sigmoid Function '''
def sigmoid(x):
    return 1/(1+np.exp(-x))

''' Derivative of sigmoid function: '''
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


''' Function to initialise the weight and bias matrices '''
def init_weights():
    
    wh1 = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE) 
    wh2 = np.random.randn(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE) 
    wo = np.random.randn(HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE) 
    
    bias1 = np.full((1, HIDDEN_LAYER_1_SIZE), 0.1)
    bias2 = np.full((1, HIDDEN_LAYER_2_SIZE), 0.1)
    biaso = np.full((1, OUTPUT_LAYER_SIZE), 0.1)
    
    return wh1, wh2, wo, bias1, bias2, biaso
                


if __name__ == '__main__':
    
    obj = NN()
    
    df = pd.read_csv("C:\\Users\\hp\\Desktop\\Studies\\5th Sem\\Machine Intelligence\\Machine-Intelligence-2020-main\\Assignment3\\LBW_Dataset.csv")
    
    # Initialise the weight and bias matrices
    wh1,wh2,wo,bias1,bias2,biaso = init_weights()
    
    # Pre-processing
    df.fillna(df.mean(), inplace=True)

    # Train - Test
    x = df[['Community','Age','Weight','Delivery phase','HB','IFA','BP','Education','Residence']]
    y = df[['Result']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Converting to numpy
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    
    # Learning rate value
    lr = 0.5
    
    for epoch in range(200000):

        #---------------------------------------------------------------------
        
        ''' Forward Propogation '''
        
        z1 = np.dot(x_train, wh1) + bias1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, wh2) + bias2
        a2 = sigmoid(z2)
        
        zo = np.dot(a2, wo) + biaso
        ao = sigmoid(zo)

        #---------------------------------------------------------------------
        
        ''' Backpropogation 1 '''
        
        error_out = ((1 / 2) * (np.power((ao - x_train), 2)))

        dcost_dao = ao - y_train
        dao_dzo = sigmoid_der(zo)
        dzo_dwo = a2

        dcost_dwo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)    
        
        #---------------------------------------------------------------------

        ''' Backpropogation 2 '''
 
        dzo_da2 = wo 
        da2_dz2 = sigmoid_der(z2)
        dz2_dwh2 = a1
        
        '''
        print(wh2.shape)
        print("dcost_dao : ",dcost_dao.shape,"\n")
        print("dao_dzo : ",dao_dzo.shape,"\n")
        print("dzo_da2 : ",dzo_da2.shape,"\n")
        print("da2_dz2 : ",da2_dz2.shape,"\n")
        print("dz2_dwh2 : ",dz2_dwh2.shape,"\n")
        '''

        pt1 = (dcost_dao * dao_dzo)
        #print(pt1.shape)
        pt2 = np.dot(pt1,dzo_da2.T)
        #print(pt2.shape)
        pt3 = pt2 * da2_dz2
        #print(pt3.shape)
        pt4 = np.dot(dz2_dwh2.T,pt3)
        #print(pt4.shape)
        
        dcost_dwh2 =  pt4


        #---------------------------------------------------------------------
        
        ''' Backpropogation 3 '''

        dz2_da1 = wh2
        da1_dz1 = sigmoid_der(z1)
        dz1_dwh1 = x_train

        '''
        print(wh1.shape)
        print("dcost_dao :  ",dcost_dao.shape,"\n")
        print("dao_dzo :  ",dao_dzo.shape,"\n")
        print("dzo_da2 :  ",dzo_da2.shape,"\n")
        print("da2_dz2 :  ",da2_dz2.shape,"\n")
        print("dz2_da1 :  ",dz2_da1.shape,"\n")
        print("da1_dz1 :  ",da1_dz1.shape,"\n")
        print("dz1_dwh1 :  ",dz1_dwh1.shape,"\n")
        '''
        
        pt1 = (dcost_dao * dao_dzo)
        #print(pt1.shape)
        pt2 = np.dot(pt1,dzo_da2.T)
        #print(pt2.shape)
        pt3 = pt2 * da2_dz2
        #print(pt3.shape)
        pt4 = np.dot(pt3,dz2_da1)
        #print(pt4.shape)
        pt5 = pt4 * da1_dz1
        #print(pt5.shape)
        pt6 = np.dot(dz1_dwh1.T,pt5)
        #print(pt6.shape)
        
        dcost_dwh1 = pt6
        
        #---------------------------------------------------------------------
        
        ''' Update Weights '''
        
        wh1 -= lr * dcost_dwh1
        wh2 -= lr * dcost_dwh2
        wo -= lr * dcost_dwo

        #---------------------------------------------------------------------
        
        print(epoch)

         
        
    # Test on test data
    z1 = np.dot(x_train, wh1)
    a1 = sigmoid(z1)

    z2 = np.dot(x_train, wh1)
    a2 = sigmoid(z2)

    zo = np.dot(a2, wo)
    ao = sigmoid(zo)

    obj.CM(y_test,ao)