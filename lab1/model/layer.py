import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
class Convolution(_Layer):
    def __init__(self, kernal_size, stride, in_size):
        self.weight = np.random.randn(kernal_size, kernal_size) * 0.01
        
        self.kernal_size = kernal_size
        self.stride = stride
        self.in_size = in_size
        self.out_size = in_size - kernal_size + 1

        self.bias = np.zeros([self.out_size, self.out_size])
    
    def forward(self, input):
        self.input = input
        image_num = input.shape[0]
        output = np.empty([image_num, self.out_size, self.out_size])
        for index in range(image_num):
            cmatrix = np.lib.stride_tricks.as_strided(
                input[index],
                shape=(self.out_size, self.out_size, self.kernal_size, self.kernal_size),
                strides=(self.stride*self.in_size, self.stride, self.stride*self.in_size, self.stride)
            )
            cmatrix = cmatrix.reshape(-1, self.kernal_size*self.kernal_size)
            output[index] = self.weight.ravel().dot(cmatrix.T).reshape(self.out_size, self.out_size) + self.bias
            
        return output
    
    def backward(self, output_grad):
        image_num = output_grad.shape[0]
        output_grad = output_grad.reshape([image_num, self.out_size, self.out_size])
        #input_grad = np.empty([image_num, self.in_size, self.in_size])
        self.weight_grad = np.empty([image_num, self.kernal_size, self.kernal_size])
        self.bias_grad = np.empty([image_num, self.out_size, self.out_size])
        for batch_idx in range(image_num):

            cmatrix = np.lib.stride_tricks.as_strided(
                self.input[batch_idx],
                shape=(self.out_size, self.out_size, self.kernal_size, self.kernal_size),
                strides=(self.stride*self.in_size, self.stride, self.stride*self.in_size, self.stride)
            )
            cmatrix = cmatrix.reshape(-1, self.kernal_size*self.kernal_size)

            weight_grad_tmp = np.zeros([self.kernal_size, self.kernal_size])
            for out_idx in range(output_grad[batch_idx].size):
                weight_grad_tmp += output_grad[batch_idx].ravel()[out_idx] * cmatrix[out_idx].reshape([self.kernal_size, self.kernal_size])

            self.weight_grad[batch_idx] = weight_grad_tmp
            self.bias_grad[batch_idx] = output_grad[batch_idx]

        #return input_grad
    
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros([1, out_features])


    def forward(self, input):
        self.forward_pass = input
        image_num = input.shape[0]
        output = np.empty([image_num, 1, self.bias.shape[1]])
        for index in range(image_num):
            output[index] = input[index].ravel().dot(self.weight) + self.bias

        return output

    def backward(self, output_grad):
        image_num = output_grad.shape[0]
        input_grad = np.empty([image_num, self.weight.shape[0]])
        self.weight_grad = np.empty([image_num, self.weight.shape[0], self.weight.shape[1]])
        self.bias_grad = np.empty([image_num, self.bias.shape[0], self.bias.shape[1]])
        for batch_idx in range(image_num):
            input_grad[batch_idx] = output_grad[batch_idx].dot(self.weight.T)
            self.weight_grad[batch_idx] = np.outer(self.forward_pass[batch_idx].ravel(), output_grad[batch_idx])
            self.bias_grad[batch_idx] = output_grad[batch_idx]

        return input_grad

## by yourself .Finish your own NN framework
class Relu(_Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        output = np.maximum(0, input)
        
        return output
        
    def backward(self, output_grad):
        input_grad = np.where(output_grad < 0, 0, 1)
        
        return input_grad

class Sigmoid(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        output = np.where(input < 0, np.exp(input)/(1 + np.exp(input)), 1/(1 + np.exp(-input)))

        return output

    def backward(self, output_grad):
        input_grad = self.forward(output_grad) * (1 - self.forward(output_grad))
        
        return input_grad

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass

    def forward(self, input, target):
        self.target = target

        image_num = input.shape[0]
        predict = np.empty([image_num, 10])
        your_loss = np.empty([image_num, 1])
        for index in range(image_num):

            '''Softmax'''
            x =input[index] - input[index].max()
            predict[index] = np.exp(x) / np.exp(x).sum()



            '''Cross entropy'''
            your_loss[index] = - (target[index].dot(np.log(predict[index] + 1e-15)))
        self.predict = predict

        return predict, your_loss

    def backward(self):

        input_grad = self.predict - self.target

        return input_grad
    
    