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
        Y = np.dot(X, W)                   # Determine layer exit
        
        if (self.sanger):
            ydiag = np.multiply(Y, diag)                             # MxM matrix
            Z = np.dot(ydiag, np.transpose(W))                       # Predict X using Sanger
            dW = np.transpose((np.multiply(Z, -1) + X))               # Weight corrections using Sanger
            dW = np.multiply(dW, Y)
        else:
            Z = np.dot(Y, W.transpose())                            # Predict X using Oja
            dW = np.outer(X - Z, Y)                                 # Weight corrections using Oja
        return Y, dW



    def learn(self, X):
        ans = []                                                    # Array of final layer outputs
        for i in range(0, len(X)):
            dW = []                                                 # Array of weight corrections
            ans_i = [np.reshape(X[i], (1, len(X[i])))]              # Array of answers, first element has shape (n,), so must be reshaped
            for j in range(0, self.layers - 1):
                y_j, dw_j = self.activate(ans_i[j], self.W[j])      # Determine weight corrections for each layer
                ans_i.append(y_j)                                   # Append current layer answer
                dW.append(dw_j)                                     # Save weight corrections
                
            ans.append(ans_i[len(ans_i) - 1])                       # Save final output
            
            self.adaptation(dW)                                     # Correct weights

        return ans
    
    def train(self, X):
        learning = []
        o = self.orthogonalWeights()
        iters = 0
        while(o > 0.1 and iters < self.maxIter):
            self.learn(X)
            o = self.orthogonalWeights()
            iters += 1
            learning.append(o)
            print(iters, o)
        return iters, o, learning


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
    
    # Apply weight corrections given array of weight correction matrices
    def adaptation(self, dW):
        for k in range(0, self.layers - 1):
            self.W[k] = self.W[k] + dW[k]*self.learningRate
    
    # Last layer orthogonal weights
    def orthogonalWeights(self):
        w = self.W[len(self.W) - 1]
        M = np.shape(w)[1]
        o = np.sum(np.abs(np.dot(np.transpose(w), w) - np.identity(M)))/2
        return o
