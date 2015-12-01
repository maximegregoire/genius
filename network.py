import os.path

import xor_network

class Network:
    
    def __init__(self, name, model_path, training_function):
        self.name = name
        self.model_path = model_path
        self.training_function = training_function
        self.trained = os.path.isfile(self.model_path)
        
    def __repr__(self):
        return self.name
        
    def __self__(self):
        return self.name
        
    def train(self):
        if self.trained:
            print("Error : Network already trained!")
            return
        if self.training_function == None:
            print("The training function is undefined")
            return
        self.training_function()
        self.trained = True;
        
def networks_available():
    networks = []
    folder_path = 'models/'
    networks.append(Network("XOR Operator", folder_path + "xor_model.obj", xor_network.train_xor_network))
    networks.append(Network("MNIST", folder_path + "mnist.obj", None))
    networks.append(Network("Iris", folder_path + "iris.obj", None))
    return networks
    
def get_network(networks, name):
    for n in networks:
        if n.name == name:
            return n
    