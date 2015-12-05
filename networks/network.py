import os.path

from XOR_Operator import xor_network
from numpy import genfromtxt

class Result:
    def __init__(self, network, inputs, outputs):
        self.network = network
        self.inputs = inputs
        self.outputs = outputs;
#endclass

class Network:
    
    def __init__(self, name, training_path, testing_path):#model_path, training_function, load_model_function, output_function, number_of_inputs):
        self.name = name
        self.training_path = training_path
        self.testing_path = testing_path

        self.training_function = None
        self.output_function = None
        self.trained = False
        self.in_training = False
        self.number_of_inputs = 0
        self.input_start_column = 0
        self.input_end_column = 0
        self.output_column = 0
        self.number_of_columns = 0
        self.qualitative_outputs = False
        
        # CSV, separated by a coma
        self.delimitation = ','
            
    def __repr__(self):
        return self.name
        
    def __self__(self):
        return self.name
        
    def getSampleTraining(self, number_of_lines):
        if not os.path.isfile(self.training_path):
            raise ValueError("Cannot find the training file : " + self.training_path)
        self.training_data = genfromtxt(self.training_path, delimiter=self.delimitation)
        self.number_of_columns = len(self.training_data[0])
        return self.training_data[0:number_of_lines]
        
    def getInputs(self, data):
        if data == None:
            raise ValueError("The data has not been loaded")
        self.inputs = np.array([i[input_start_column:input_end_column + 1] for i in data])
        
    def getOutputs(self, data):
        if data == None:
            raise ValueError("The data has not been loaded")
        if self.qualitative_outputs:
            self.outputs, self.outputs_onehot = convertOneHot(data, output_column)
        else:
            raise ValueError("NOT IMPLEMENTED YET")
            self.outputs = np.array([i[self.output_column] for i in data])
        
    def train(self):
        self.training_data = genfromtxt(self.training_path, delimiter=self.delimitation)        
        self.testing_data = genfromtxt(self.testing_path, delimiter=self.delimitation)
        training_outputs = getOutputs(self.training_data)
        testing_outputs = getOutputs(self.testing_data)
        
        
    def get_output(self, inputs):
    	if output_function == None:
    		raise ValueError("Output function undefined")
        return Result(self, inputs, self.output_function(inputs))
        
def networks_available():
    networks = []
    folder_path = 'networks/'
    networks.append(Network("XOR_Operator", None, None))
    networks.append(Network("MNIST", None, None))
    networks.append(Network("Iris", None, None))
    return networks
    
def get_network(networks, name):
    for n in networks:
        if n.name == name:
            return n
    
