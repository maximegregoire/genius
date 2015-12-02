import os.path

from XOR_Operator import xor_network

class Result:
    def __init__(self, network, inputs, outputs):
        self.network = network
        self.inputs = inputs
        self.outputs = outputs;
#endclass

class Network:
    
    def __init__(self, name, model_path, training_function, load_model_function, output_function, number_of_inputs):
        self.name = name
        self.model_path = model_path
        self.number_of_inputs = number_of_inputs
        self.training_function = training_function
        self.output_function = output_function
        self.in_training = False
        self.trained = os.path.isfile(self.model_path)
        if self.trained:
            print("in loading function\n")
            load_model_function()
            
    def __repr__(self):
        return self.name
        
    def __self__(self):
        return self.name
        
    def train(self):
        if self.trained or self.in_training:
            print("Error : Network already trained!")
            return
        if self.training_function == None:
            print("The training function is undefined")
            return
        self.in_training = True
        self.training_function()
        self.in_training = False
        self.trained = True;
        
    def get_output(self, inputs):
        return Result(self, inputs, self.output_function(inputs))
        
def networks_available():
    networks = []
    folder_path = 'networks/'
    networks.append(Network("XOR_Operator", folder_path + "XOR_Operator/XOR_Operator.obj", xor_network.train_xor_network, xor_network.load_network, xor_network.get_output, 2))
    networks.append(Network("MNIST", folder_path + "mnist.obj", None, None, None, 700))
    networks.append(Network("Iris", folder_path + "iris.obj", None, None, None, 10))
    return networks
    
def get_network(networks, name):
    for n in networks:
        if n.name == name:
            return n
    