from .layer import *

class Network(object):
    def __init__(self):

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        self.cnn1 = Convolution(3, 1, 28)
        self.act1 = Relu()
        self.fc1 = FullyConnected(26*26, 10) ## Just an example.You can alter sample code anywhere. 
        self.loss = SoftmaxWithloss()



    def forward(self, input, target):
        ## by yourself .Finish your own NN framework
        h1 = self.cnn1.forward(input.reshape(input.shape[0], 28, 28))
        n1 = self.act1.forward(h1)
        h2 = self.fc1.forward(n1)
        pred, loss = self.loss.forward(h2, target)
        
        loss_total = loss.sum()
        
        return pred, loss_total

    def backward(self):
        ## by yourself .Finish your own NN framework
        #h1_grad = 
        #_ = self.fc1.backward(h1_grad)
        loss_grad = self.loss.backward()
        h2_grad = self.fc1.backward(loss_grad)
        n1_grad = self.act1.backward(h2_grad)
        self.cnn1.backward(n1_grad)


    def update(self, lr):
        ## by yourself .Finish your own NN framework
        
        self.fc1.weight -= lr * np.sum(self.fc1.weight_grad, axis = 0)
        self.fc1.bias -= lr * np.sum(self.fc1.bias_grad, axis = 0)
        self.cnn1.weight -= lr * np.sum(self.cnn1.weight_grad, axis = 0)
        self.cnn1.bias -= lr * np.sum(self.cnn1.bias_grad, axis = 0)
        
        
