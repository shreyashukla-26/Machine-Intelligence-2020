'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space - carries 1 mark

Input Layer - 9 neurons
1 Hidden Layer - 4 neurons
Output Layer - 1 neuron
Weight - (9 cross 4 matrix) and (4 cross 1 matrix)
Bias - haven't included bias but it's supposd to be there (supposed to be a matrix as mentioned in the pdf)
Activation fucntion - Sigmoid for all neurons
Loss function - using RMSE for error (If that's what they mean by Loss function)

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

    df = pd.read_csv("LBW_Dataset.csv")
 
    # Deleting rows with empty values
    df = df.dropna(how='any',axis=0)
    
    #Replacing missing values with mean
    #df.fillna(df.mean(), inplace=True)

    # Train - Test
    x = df[['Community','Age','Weight','Delivery phase','HB','IFA','BP','Education','Residence']]
    y = df[['Result']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Converting to numpy
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    wh = np.random.rand(len(x_train[0]),4)
    wo = np.random.rand(4, 1)

	# After 200000 iterations
    wh = np.array([[0.21523141, 0.2757868, 0.1715146, 0.01635213], [0.92260834, 0.88752657, 0.48935254, 0.11335845], [0.27867197, 0.44425992, 0.86082173, 0.8986357 ], [0.64365148, 0.78640503, 0.75495827, 0.48983266], [0.8046612,  0.85268786, 0.26778352, 0.8142019 ], [0.28657463, 0.78866965, 0.92386554, 0.57133837], [0.90910202, 0.86338865, 0.09446842, 0.87762915], [0.23636098, 0.95466745, 0.50246623, 0.55865984], [0.63155968, 0.01170663, 0.52540515, 0.13832194]])
    wo = np.array([[0.94104904], [1.063898  ], [0.37349239], [0.23652035]])

	# After 400000 iterations
    wh = np.array([[0.21523141, 0.2757868,  0.1715146,  0.01635213],
 [0.92260834, 0.88752657, 0.48935254, 0.11335845],
 [0.27867197, 0.44425992, 0.86082173, 0.8986357 ],
 [0.64365148, 0.78640503, 0.75495827, 0.48983266],
 [0.8046612, 0.85268786, 0.26778352, 0.8142019 ],
 [0.28657463, 0.78866965, 0.92386554, 0.57133837],
 [0.90910202, 0.86338865, 0.09446842, 0.87762915],
 [0.23636098, 0.95466745, 0.50246623, 0.55865984],
 [0.63155968, 0.01170663, 0.52540515, 0.13832194]])
    wo = np.array([[0.94104904],
                   [1.063898  ],
                   [0.37349239],
                   [0.23652035]])

    # After 600000 iterations
    wh = np.array([[0.21523141, 0.2757868,  0.1715146,  0.01635213],
 [0.92260834, 0.88752657, 0.48935254, 0.11335845],
 [0.27867197, 0.44425992, 0.86082173, 0.8986357 ],
 [0.64365148, 0.78640503, 0.75495827, 0.48983266],
 [0.8046612, 0.85268786, 0.26778352, 0.8142019 ],
 [0.28657463, 0.78866965, 0.92386554, 0.57133837],
 [0.90910202, 0.86338865, 0.09446842, 0.87762915],
 [0.23636098, 0.95466745, 0.50246623, 0.55865984],
 [0.63155968, 0.01170663, 0.52540515, 0.13832194]])
    wo = np.array([[0.94104904],
 				   [1.063898  ],
 			   	   [0.37349239],
 			   	   [0.23652035]])

    lr = 0.5

    #print(wh)
    #print(wo)

    for epoch in range(200000):

        # Forward Propogation
        zh = np.dot(x_train, wh)
        ah = sigmoid(zh)
        #print(zh,ah)

        zo = np.dot(ah, wo)
        ao = sigmoid(zo)
        #print(zo,ao)


        # Backpropogation 1
        error_out = ((1 / 2) * (np.power((ao - x_train), 2)))
        #print(error_out.sum())

        dcost_dao = ao - y_train
        dao_dzo = sigmoid_der(zo)
        dzo_dwo = ah

        dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)


        # Backpropogation 2

        # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
        # dcost_dah = dcost_dzo * dzo_dah
        dcost_dzo = dcost_dao * dao_dzo
        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        dzh_dwh = x_train
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)


        # Update Weights ================
        wh -= lr * dcost_wh
        wo -= lr * dcost_wo

        print(epoch)

    print(wh)
    print(wo)

	# Test on test data
    zh = np.dot(x_train, wh)
    ah = sigmoid(zh)

    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    obj.CM(y_test,ao)

'''
Output : 
Confusion Matrix :
[[0, 0], [3, 17]]


Precision : 1.0
Recall : 0.85
F1 SCORE : 0.9189189189189189
'''
