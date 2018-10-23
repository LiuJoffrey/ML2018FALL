import numpy as np

class NeuralNetwork(object):
    


    def __init__(self, learning_rate=0.01, epochs=15000, batch_size=50, feature_size = 33):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        np.random.seed(0)

        #in this case I have used random init for weights
        #Smarter way to initialize weights is Xavier initialization
        self.weigts_one = np.random.randn(feature_size, 50)
        self.bias_one = np.zeros((1, 50))
        #self.weigts_two = np.random.randn(50, 25)
        #self.bias_two = np.zeros((1, 25))
        self.weighs_three = np.random.randn(50, 1)
        self.bias_three = np.zeros((1, 1))
        
    
    def sigmoid(self, x):
        
        return 1 / (1 + np.exp(-x))
    
    def der_sigmoid(self, x):
        
        return (1 - x) * x
    
    def train(self, X ,y):
        
        #Because of the cost function you will need to reshape y array to : [len(y), 1]
        y_train = np.reshape(y, (len(y), 1))
        
        time_per_epoch = len(X) // self.batch_size

        #Training loop
        for i in range(self.epochs):
            
            for j in range(10):
                idx = np.random.choice(len(X), self.batch_size, replace=True)

                X_batch = X[idx, :]
                y_batch = y_train[idx, :]
                #first = j*self.batch_size
                #last = (j+1)*self.batch_size
                #if last>len(X):
                #    last = len(X)
                #X_batch = X[first:last, :]
                #y_batch = y_train[first:last, :]
                
                l1, scores = self.forward(X_batch)
                
                cost = y_batch - scores
                
                if i % 1000 == 0:
                    print(np.mean(np.square(cost)))
                    
                #backprop
                dscores = cost * self.der_sigmoid(scores)
                dW3 = np.dot(l1.T, dscores)
                db3 = np.sum(dscores, axis=0, keepdims=True)
                #dl2 = np.dot(dscores, self.weighs_three.T) * self.der_sigmoid(l2)
                #db2 = np.sum(dl2, axis=0, keepdims=True)
                #dW2 = np.dot(l1.T, dl2)
                dl1 = np.dot(dscores, self.weighs_three.T) * self.der_sigmoid(l1)
                dW1 = np.dot(X_batch.T, dl1)
                db1 = np.sum(dl1, axis=0, keepdims=True)
                    
                #NOTE: We are not using Regularizaiton term in this network
                #Stochastic Gradient Descent
                self.weigts_one += self.learning_rate * dW1
                self.bias_one += self.learning_rate * db1
                #self.weigts_two += self.learning_rate * dW2
                #self.bias_two += self.learning_rate * db2
                self.weighs_three += self.learning_rate * dW3
                self.bias_three += self.learning_rate * db3
            
                
    def forward(self, X):
        '''
        This function is used for forward propagation through our network
        
        Input(s): X - features from dataset
        '''

        l1 = self.sigmoid((np.dot(X, self.weigts_one) + self.bias_one))
        #l2 = self.sigmoid((np.dot(l1, self.weigts_two) + self.bias_two))
        scores = self.sigmoid((np.dot(l1, self.weighs_three) + self.bias_three))
        return l1, scores
    
    def predict(self, X):
        '''
        This function is used to threshold values from our network to 1 or 0 depending how big it is
        because of the Sigmoid activation function we will get result between 0 and 1 for every sample
        
        Input(s): X - features from dataset
        '''
        l1, scores = self.forward(X)
        pred = []
        for i in range(len(scores)):
            if scores[i] >= 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred
    
    def accuracy(self, pred, y_test):
        '''
        Simple function to check how accuarte is neural network.

        Input(s): pred - predicted values from the network
                 y_test - currect values for our TEST set
        '''
        assert len(pred) == len(y_test)
        true_pred = 0
        for i in range(len(pred)):
            if pred[i] == y_test[i]:
                true_pred += 1
        print((true_pred/len(pred))*100, "%")