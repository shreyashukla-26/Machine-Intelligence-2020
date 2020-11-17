import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import math

class NN():
    #initialise hyperparameters
    def __init__(self, input_layer, hidden_1, hidden_2, output_layer, iterations=100, learning_rate=0.5):

        self.iterations = iterations
        self.learning_rate = learning_rate

        np.random.seed(57)

        # initialise weights - He initialisation
        self.weights = {
            'W1':np.random.rand(hidden_1, input_layer) * np.sqrt(2. / hidden_1) * 2,
            'W2':np.random.rand(hidden_2, hidden_1) * np.sqrt(2. / hidden_2) * 2,
            'W3':np.random.rand(output_layer, hidden_2)  * np.sqrt(2. / output_layer) * 2
        }

        # initialise bias
        self.bias = {
            'BO': np.random.rand(),
            'B1': np.random.rand(),
            'B2': np.random.rand()
        }


    #Function to pre-process the data
    def datapreprocessing(self, df):

        '''Binary threshold on Weight column'''
        # Normalising the 'Weight' column to obtain it's median
        temp_df = pd.read_csv("LBW_Dataset.csv")
        scaler = StandardScaler().fit(temp_df[['Weight']])
        temp_df[['Weight']] = scaler.transform(temp_df[['Weight']])

        # Applying a binary threshold of 33 on the 'Weight' column
        for i in range(len(df)):
            # Replacing NULL values with the ceil of median of the original normalized 'Weight' column
            if(pd.isnull(df['Weight'][i])):
                df['Weight'][i] = math.ceil(temp_df['Weight'].median())
            # Assigning 1 to values greater than 33
            elif(df['Weight'][i] > 33):
                df['Weight'][i] = 1
            # Assigning 1 to values lesser than or equal to 33
            else:
                df['Weight'][i] = 0


        # Replace all Nan values with the median values of the columns
        df.fillna(df.median(), inplace=True)

        '''Upsampling to handle the Imbalanced Classes'''

        # Split the dataframe based on the classes
        df_majority = df[df.Result==1]
        df_minority = df[df.Result==0]

        # Use resampling on the minority class to create more records for the same.
        df_minority_upsampled = resample(df_minority,
                                        replace=True,
                                        n_samples=72,
                                        random_state=42)

        # Concatenate the dataframes of the majority class and upsampled minority class. The two classes are now equally represented
        global df_upsampled
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])


    # Sigmoid Function
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Derivative of Sigmoid Function
    def sigmoid_derivative(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    # Train the model
    def fit(self, x_train, y_train):
        #iterating for the specified number of epochs
        for iteration in range(self.iterations):
            for x,y in zip(x_train, y_train):

                weights = self.weights
                bias = self.bias

                '''Forward Propagation'''

                # input layer to hidden layer 1
                # z1 = x * weight1 + bias1
                # a1 = sigmoid(z1)
                weights['Z1'] = np.dot(weights["W1"], x) + bias['BO']
                weights['A1'] = self.sigmoid(weights['Z1'])

                # hidden layer 1 to hidden layer 2
                # z2 = a1 * weight2 + bias2
                # a2 = sigmoid(z2)
                weights['Z2'] = np.dot(weights["W2"], weights['A1']) + bias['B1']
                weights['A2'] = self.sigmoid(weights['Z2'])

                # hidden layer 2 to output layer
                # z3 = a2 * weight3 + bias3
                # a3 = sigmoid(z3)
                weights['Z3'] = np.dot(weights["W3"], weights['A2']) + bias['B2']
                weights['A3'] = self.sigmoid(weights['Z3'])

                # Final prediction for current epoch
                predicted = weights['A3']

                '''Back Propagation'''

                #Dictionary to store differential of weights wrt error
                delta_w = {}

                # Calculate W3 update
                # dcost_dw3 = (predicted - y) * sigmoid_der(Z3) * A2
                error = 2 * (predicted - y) / predicted.shape[0] * self.sigmoid_derivative(weights['Z3'])
                delta_w['W3'] = np.outer(error, weights['A2'])

                # Calculate W2 update
                # dcost_dw2 = (predicted - y) * sigmoid_der(Z3) * W3 * sigmoid_der(Z2) * A1
                error = np.dot(weights['W3'].T, error) * self.sigmoid_derivative(weights['Z2'])
                delta_w['W2'] = np.outer(error, weights['A1'])

                # Calculate W1 update
                # dcost_dw1 = (predicted - y) * sigmoid_der(Z3) * W3 * sigmoid_der(Z2) * W2 * sigmoid_der(Z1) * x
                error = np.dot(weights['W2'].T, error) * self.sigmoid_derivative(weights['Z1'])
                delta_w['W1'] = np.outer(error, x)

                #Update weights
                for key, value in delta_w.items():
                    self.weights[key] -= self.learning_rate * value


            print("Epoch :",iteration+1)


    # Predict using the fitted model
    def predict(self, x_val, y_val):

        predictions = []
        weights = self.weights
        bias = self.bias

        for x, y in zip(x_val, y_val):

            # input layer to hidden layer 1
            # z1 = x * weight1 + bias1
            # a1 = sigmoid(z1)
            weights['Z1'] = np.dot(weights["W1"], x) + bias['BO']
            weights['A1'] = self.sigmoid(weights['Z1'])

            # hidden layer 1 to hidden layer 2
            # z2 = a1 * weight2 + bias2
            # a2 = sigmoid(z2)
            weights['Z2'] = np.dot(weights["W2"], weights['A1']) + bias['B1']
            weights['A2'] = self.sigmoid(weights['Z2'])

            # hidden layer 2 to output layer
            # z3 = a2 * weight3 + bias3
            # a3 = sigmoid(z3)
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

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        acc=(tp+tn)/(tp+tn+fp+fn)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        print(f"Accuracy : {acc}")


# Create an instance of the NN class
obj = NN(6, 3, 3, 1)

# Read the dataset
df = pd.read_csv("LBW_Dataset.csv")

# Preprocess the data
obj.datapreprocessing(df)
df = df_upsampled

# Select the most relevant attributes
x = df[['Community','Weight','Delivery phase','IFA','Residence','Delivery phase']]
y = df[['Result']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Convert the data to numpy arrays
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

#Train the model
obj.fit(x_train, y_train)

#Final results on the train data
print("\n-----------------------------")
print("\nTrain Data")
obj.predict(x_train, y_train)

#Final results on the test data
print("\n-----------------------------")
print("\nTest Data")
obj.predict(x_test, y_test)
print("\n-----------------------------")
