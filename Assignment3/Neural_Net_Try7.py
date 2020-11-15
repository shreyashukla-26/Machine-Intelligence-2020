import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.decomposition import PCA

df = pd.read_csv("LBW_Dataset.csv")

for i in range(len(df)):
    try:
        if(df['Weight'][i] > 33):
            df['Weight'][i] = 1

        else:
            df['Weight'][i] = 0

    except:
        continue

df.fillna(df.mean(), inplace=True)
#df = df.dropna(how='any',axis=0)

#for i in range(len(df)):
#    print(df['Weight'][i],df['Result'][i])

# Train - Test
x = df[['Community','Weight','Delivery phase','IFA','Residence']]
#x = df[['Weight','Delivery phase','IFA','Education']]
#x = df[['Community','Age','Weight','Delivery phase','HB','IFA','BP','Education','Residence']]
#x = df[['Weight']]
y = df[['Result']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

'''
# PRINCIPAL COMPONENT ANALYSIS
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
#print(explained_variance)
'''

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

#x_train = (x/255).astype('float32')
#y_train = (x/255).astype('float32')
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

class NN():
    def __init__(self, sizes, epochs=150, l_rate=0.5):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        # we save all parameters in the neural network in this dictionary
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.rand(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.rand(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.rand(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.softmax(params['Z1'])
        #print(params['A1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.softmax(params['Z2'])
        #print(params['A2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        #print(params['Z3'])
        params['A3'] = self.sigmoid(params['Z3'])
        #print(params['A3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is
                  caused  by the dot and multiply operations on the huge arrays.

                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.sigmoid(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.softmax(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.softmax(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y),
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []
        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            #print(output)
            pred = np.argmax(output)
            #print(pred)
            #predictions.append(pred == np.argmax(y))
            predictions.append(output)
        #print(predictions)
        #print(y_val)
        self.CM(y_val,predictions)
        return np.mean(predictions)

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
        acc=(tp+tn)/(tp+tn+fp+fn)

        print("Confusion Matrix : ")
        print(cm)
        #print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print(f"Accuracy : {acc}")


    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            print("Test Data")
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s'.format(
                iteration+1, time.time() - start_time
            ))
            print("\n")

            print("Train Data")
            accuracy = self.compute_accuracy(x_train, y_train)
            print('Epoch: {0}, Time Spent: {1:.2f}s'.format(
            iteration+1, time.time() - start_time
            ))
            print("\n")

            #print(self.params)
            #print("\n")

obj = NN(sizes=[5, 3, 3, 1])
obj.train(x_train, y_train, x_test, y_test)
