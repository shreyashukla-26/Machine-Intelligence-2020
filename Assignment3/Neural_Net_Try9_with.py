import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import math

class NN():
    def __init__(self, input_layer, hidden_1, hidden_2, output_layer, iterations=150, learning_rate=0.5):
        self.iterations = iterations
        self.learning_rate = learning_rate

        np.random.seed(57)

        self.weights = {
            'W1':np.random.rand(hidden_1, input_layer), # * np.sqrt(2. / hidden_1),
            'W2':np.random.rand(hidden_2, hidden_1), # * np.sqrt(2. / hidden_2),
            'W3':np.random.rand(output_layer, hidden_2) # * np.sqrt(2. / output_layer)
        }

        self.bias = {
            'BO': np.random.rand(),
            'B1': np.random.rand(),
            'B2': np.random.rand()
        }

    def datapreprocessing(self, df):

        '''
        df['Age'] = df['Age'].replace(np.NaN, df['Age'].mode()[0])
        df['Weight'] = df['Weight'].replace(np.NaN, df['Weight'].median())
        df['HB'] = df['HB'].replace(np.NaN, df['HB'].median())
        df['BP'] = df['BP'].replace(np.NaN, df['BP'].median())
        df.dropna(inplace = True)
        '''

        temp_df = pd.read_csv("C:\\Users\\hp\\Desktop\\Studies\\5th Sem\\Machine Intelligence\\Machine-Intelligence-2020-main\\Assignment3\\LBW_Dataset.csv")
        scaler = StandardScaler().fit(temp_df[['Weight']])
        temp_df[['Weight']] = scaler.transform(temp_df[['Weight']])

        for i in range(len(df)):
            if(pd.isnull(df['Weight'][i])):
                df['Weight'][i] = math.ceil(temp_df['Weight'].median())

            elif(df['Weight'][i] > 33):
                df['Weight'][i] = 1

            else:
                df['Weight'][i] = 0

        df.fillna(df.median(), inplace=True)

        df_majority = df[df.Result==1]
        df_minority = df[df.Result==0]

        df_minority_upsampled = resample(df_minority,
                                        replace=True,
                                        n_samples=72,
                                        random_state=42)
        global df_upsampled
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        return df_upsampled



    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def fit(self, x_train, y_train, x_test, y_test):
        for iteration in range(self.iterations):
            for x,y in zip(x_train, y_train):
                weights = self.weights
                bias = self.bias

                # input layer activations becomes sample
                weights['A0'] = x

                # input layer to hidden layer 1
                weights['Z1'] = np.dot(weights["W1"], weights['A0']) + bias['BO']
                weights['A1'] = self.sigmoid(weights['Z1'])

                # hidden layer 1 to hidden layer 2
                weights['Z2'] = np.dot(weights["W2"], weights['A1']) + bias['B1']
                weights['A2'] = self.sigmoid(weights['Z2'])

                # hidden layer 2 to output layer
                weights['Z3'] = np.dot(weights["W3"], weights['A2']) + bias['B2']
                weights['A3'] = self.sigmoid(weights['Z3'])

                predicted = weights['A3']

                delta_w = {}

                # Calculate W3 update
                error = 2 * (predicted - y) / predicted.shape[0] * self.sigmoid_derivative(weights['Z3'])
                delta_w['W3'] = np.outer(error, weights['A2'])

                # Calculate W2 update
                error = np.dot(weights['W3'].T, error) * self.sigmoid_derivative(weights['Z2'])
                delta_w['W2'] = np.outer(error, weights['A1'])

                # Calculate W1 update
                error = np.dot(weights['W2'].T, error) * self.sigmoid_derivative(weights['Z1'])
                delta_w['W1'] = np.outer(error, weights['A0'])

                #Update weights
                for key, value in delta_w.items():
                    self.weights[key] -= self.learning_rate * value

            print("Test Data")
            self.predict(x_test, y_test)
            print('Iteration: {0}'.format(iteration+1))
            print("\n")

            print("Train Data")
            self.predict(x_train, y_train)
            print('Iteration: {0}'.format(iteration+1))
            print("\n")

    def predict(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []
        weights = self.weights
        bias = self.bias
        
        for x, y in zip(x_val, y_val):

            # input layer activations becomes sample
            weights['A0'] = x

            # input layer to hidden layer 1
            weights['Z1'] = np.dot(weights["W1"], weights['A0']) + bias['BO']
            weights['A1'] = self.sigmoid(weights['Z1'])

            # hidden layer 1 to hidden layer 2
            weights['Z2'] = np.dot(weights["W2"], weights['A1']) + bias['B1']
            weights['A2'] = self.sigmoid(weights['Z2'])

            # hidden layer 2 to output layer
            weights['Z3'] = np.dot(weights["W3"], weights['A2']) + bias['B2']
            weights['A3'] = self.sigmoid(weights['Z3'])

            predicted = weights['A3']
            predictions.append(predicted)

        self.CM(y_val,predictions)

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
        try:
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
        except:
            print("Division by Zero\n")

obj = NN(6, 3, 3, 1)

df = pd.read_csv("C:\\Users\\hp\\Desktop\\Studies\\5th Sem\\Machine Intelligence\\Machine-Intelligence-2020-main\\Assignment3\\LBW_Dataset.csv")

obj.datapreprocessing(df)
df = df_upsampled

# Train - Test
x = df[['Community','Weight','Delivery phase','IFA','Residence','Delivery phase']]
#x = df[['Community','Weight','HB','IFA','BP','Education','Residence']]
#x = df[['Weight']]
y = df[['Result']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

obj.fit(x_train, y_train, x_test, y_test)
