# ==================================INCLUDES==================================

import numpy as np


# Non supervised Hebbian model using Sanger/Oja model
# ===================================MODEL====================================

class Model():
    def __init__(self, S = [], learningRate = 0.2, sanger = True, maxIter = 2500):
        # =============================VARS=============================
        self.S = S                                                  # Array with number of neurons per layer
        self.layers = len(S)                                        # Number of layers
        self.W = self.createRandomWeights()                         # Array of weight matrices
        self.learningRate = learningRate                            # Learning rate
        self.sanger = sanger                                        # If sanger if false, the model is an Oja model
        self.maxIter = maxIter                                      # Maximum training iterations


    # ===============================FUNCTIONS===============================

    def activate(self, X, W):
        M = np.shape(W)[1]                                          # Layer exit size
        diag = np.triu(np.ones((M, M)))                             # Diagonal matrix of ones
        Y = np.dot(np.reshape(X, (1, len(X))), W)                   # Determine layer exit
        
        if (self.sanger):
            Z = np.dot(W, np.transpose(Y*diag))                     # Predict X using Sanger
            dW = (np.reshape(X, (len(X), 1)) - Z) * Y               # Weight corrections using Sanger
        else:
            Z = np.dot(Y, W.transpose())                            # Predict X using Oja
            dW = np.outer(X - Z, Y)                                 # Weight corrections using Oja
        return Y, dW



    def learn(self, X):
        ans = []                                                    # Array of final layer outputs
        for i in range(0, len(X)):
            dW = []                                                 # Array of weight corrections
            ans_i = [X[i]]                                          # Array of answers
            for j in range(0, self.layers - 1):
                y_j, dw_j = self.activate(ans_i[j], self.W[j])      # Determine weight corrections for each layer
                ans_i.append(y_j)                                   # Append current layer answer
                dW.append(dw_j)                                     # Save weight corrections
                
            ans.append(ans_i[len(ans_i) - 1])                       # Save final output
            
            self.adaptation(dW)                                     # Correct weights

        return ans
    
    def train(self, X):
        o = self.orthogonalWeights()
        iters = 0
        while(o > 0.1 and iters < self.maxIter):
            self.learn(X)
            o = self.orthogonalWeights()
            iters += 1
            print(iters, o)
        return iters, o


    # ===============================AUXILIARY===============================

    # Generate randomized matrix of dimensions IxJ
    def generateWeights(self, n, m, i, j):
        return np.matrix(np.random.normal(i, j, (n, m)))

    # Generate random weights for inner layers
    # Returns array of matrices
    def createRandomWeights(self):
        w = []                                                                # Array of weight matrices
        for i in range(0, self.layers - 1):
            layerW = self.generateWeights(self.S[i], self.S[i+1], 0, 0.1)     # Create weight matrix for each connection
            w.append(layerW)
        return w
    
    def adaptation(self, dW):
        for k in range(0, self.layers - 1):
            self.W[k] = self.W[k] + dW[k]*self.learningRate
    
    def orthogonalWeights(self):
        o = np.sum(np.abs(np.dot(self.W[len(self.W) - 1].transpose(), self.W[len(self.W) - 1]) - np.identity(self.W[len(self.W) - 1].shape[1])))/2
        return o