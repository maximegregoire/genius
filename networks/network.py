import parsing
from XOR_Operator import xor_network

import os.path
import numpy as np
import tensorflow as tf
import dill
import random

class Result:
    def __init__(self, network, inputs, outputs):
        self.network = network
        self.inputs = inputs
        self.outputs = outputs;
#endclass

class Network:
    
    def __init__(self, name, training_path, testing_path):
        self.name = name
        self.training_path = training_path
        self.testing_path = testing_path
        self.training_function = None
        self.output_function = None
        self.trained = False
        self.in_training = False
        self.input_start_column = 0
        self.input_end_column = 0
        self.output_column = 0
        self.number_of_columns = 0
        
        self.qualitative_outputs = False
        self.max_qualitative_output = 0
        
        self.accuracy = 0
        self.best_accuracy = 0
        self.epochs_for_best_accuracy = 0
        self.epochs_for_accuracy = 0
        self.number_of_tests = 0
        
        self.sess = None
        self.initialized = False
        self.tf_train_step = None
        self.tf_accuracy = None
        self.tf_in = None
        self.tf_softmax_correct = None
        
        # CSV, separated by a coma
        self.delimitation = ','
            
    def __repr__(self):
        return self.name
        
    def __self__(self):
        return self.name
        
    def getSampleTraining(self, number_of_lines):
        if not os.path.isfile(self.training_path):
            raise ValueError("Cannot find the training file : " + self.training_path)
        self.training_data = np.genfromtxt(self.training_path, delimiter=self.delimitation)
        self.number_of_columns = len(self.training_data[0])
        return self.training_data[0:number_of_lines]
        
    def getMaxOutput(self, data):
        return np.array([int(i[self.output_column]) for i in data]).max()        
        
    def getInputs(self, data):
        if data == None:
            raise ValueError("The data has not been loaded")
        self.inputs = np.array([i[self.input_start_column:self.input_end_column + 1] for i in data])
        
    def getQualitativeOutputs(self, data, max_output):
        if data == None:
            raise ValueError("The data has not been loaded")
        outputs, outputs_onehot = parsing.convertOneHot(data, self.output_column, max_output)
        return (outputs, outputs_onehot)
        
    def extractData(self):
        if self.qualitative_outputs:
            if self.testing_path == None:
                full_training = np.genfromtxt(self.training_path, delimiter=self.delimitation)
                full_size = len(full_training)
                random.shuffle(full_training)
                
                self.training_data = full_training[:full_size-self.number_of_tests]
                self.testing_data = full_training[full_size-self.number_of_tests:]
            else:
                self.training_data = np.genfromtxt(self.training_path, delimiter=self.delimitation)
                self.testing_data = np.genfromtxt(self.testing_path, delimiter=self.delimitation)
                
            max_output = max(self.getMaxOutput(self.training_data), self.getMaxOutput(self.testing_data))
            print "max output = ", max_output
            self.training_outputs, self.training_outputs_onehot = self.getQualitativeOutputs(self.training_data, max_output)
            self.testing_outputs, self.testing_outputs_onehot = self.getQualitativeOutputs(self.testing_data, max_output)
        
    def getRelativePath(self):
        return 'networks/models/' + self.name + '.pik'
        
    def saveModel(self):
        with open(self.getRelativePath(), "wb") as f:
            dill.dump(self, f)
            
    def deleteModel(self):
        if not os.path.isfile(self.getRelativePath()):
            raise ValueError("The model file is not present")
        os.remove(self.getRelativePath())
            
    def initialize(self, method):
        self.accuracy = 0
        self.best_accuracy = 0
        self.epochs_for_accuracy = 0
        self.epochs_for_best_accuracy = 0
        if method == "Gradient descent":
            self.initializeGradient()
            self.method_initialized = method
            self.initialized = True
        elif method == "K-nearest neighbors":
            self.initializeKnn()
            self.method_initialized = method
            self.initialized = True
        else:
            #todo: error in training
            raise ValueError("Training method not recognized")
    
    def initializeKnn(self):        
        if self.qualitative_outputs:            
            self.tf_in = tf.placeholder("float", [None, self.number_of_columns])
            self.tf_testing = tf.placeholder("float", [self.number_of_columns])
            
            # Calculate L1 Distance
            self.distance = tf.reduce_sum(tf.abs(tf.add(self.tf_in, tf.neg(self.tf_testing))), reduction_indices=1)
            # Predict: Get min distance index (Nearest neighbor)
            self.prediction = tf.arg_min(self.distance, 0)
            
            init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init)
            accuracy = 0
            #output part
            for i in range(len(self.testing_data)):
                # Get nearest neighbor
                nn_index = self.sess.run(self.prediction, feed_dict={self.tf_in: self.training_data, self.tf_testing: self.testing_data[i,:]})
                # Calculate accuracy
                if np.argmax(self.training_outputs_onehot[nn_index]) == np.argmax(self.testing_outputs_onehot[i]):
                    accuracy += 1./len(self.testing_data)
            self.accuracy = accuracy
            self.epochs_for_accuracy = "N/A"
            self.best_accuracy = "N/A"
            self.epochs_for_best_accuracy = "N/A"
            self.trained = True
        else:
            raise ValueError("NOT IMPLEMENTED")
    
    def initializeGradient(self):        
        if self.qualitative_outputs:
            number_of_outputs = max(len(self.training_outputs_onehot[0]),len(self.testing_outputs_onehot[0])) 
            # the placeholder for the inputs
            self.tf_in = tf.placeholder("float", [None, self.number_of_columns], name="The_raw_data")

            # the placeholder for the outputs
            self.tf_softmax_correct = tf.placeholder("float", [None, number_of_outputs], name="The_correct_data")

            self.tf_weight = tf.Variable(tf.zeros([self.number_of_columns, number_of_outputs]))
            self.tf_bias = tf.Variable(tf.zeros([number_of_outputs]))
            self.tf_softmax = tf.nn.softmax(tf.matmul(self.tf_in,self.tf_weight) + self.tf_bias)

            # Training via backpropagation
            self.tf_cross_entropy = -tf.reduce_sum(self.tf_softmax_correct*tf.log(self.tf_softmax))

            # Train using tf.train.GradientDescentOptimizer
            self.tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.tf_cross_entropy)

            # Add accuracy checking nodes
            self.tf_correct_prediction = tf.equal(tf.argmax(self.tf_softmax,1), tf.argmax(self.tf_softmax_correct,1))
            self.tf_accuracy = tf.reduce_mean(tf.cast(self.tf_correct_prediction, "float"))
            
            # Initialize and run
            tf.Session().close()
            self.sess = tf.Session()
            #sess = tf.InteractiveSession()
            init = tf.initialize_all_variables()
            self.sess.run(init)

            # This is for saving all our work
            saver = tf.train.Saver([self.tf_weight,self.tf_bias])
        else:
            raise ValueError("NOT IMPLEMENTED")
    
    def train(self, method, number_of_epochs, stop_at_100_accuracy):
        if method != self.method_initialized:
            raise ValueError("The current training method does not match the initialization method")
        print("Method = "+ method)
        if method == "Gradient descent":
            self.trainGradient(number_of_epochs, stop_at_100_accuracy)
        elif method == "K-nearest neighbors":
            # No training to do, as KNN is not training-based
            return
        else:
            #todo: error in training
            raise ValueError("Training method not recognized")
        
    def trainGradient(self, number_of_epochs, stop_at_100_accuracy):
        if self.qualitative_outputs:
            # Run the training

            k=[]
            saved=0
            result=0
            for i in range(number_of_epochs):
                self.sess.run(self.tf_train_step, feed_dict={self.tf_in: self.training_data, self.tf_softmax_correct: self.training_outputs_onehot})
                # Print accuracy
                result = self.sess.run(self.tf_accuracy, feed_dict={self.tf_in: self.testing_data, self.tf_softmax_correct: self.testing_outputs_onehot})
                print "Run {},{}".format(i,result)
                k.append(result)
                self.epochs_for_accuracy += 1
                if result >= self.best_accuracy:
                    self.best_accuracy = result
                    self.epochs_for_best_accuracy = self.epochs_for_accuracy
                
                if stop_at_100_accuracy and result == 1 and saved == 0:
                    break
                    saved=1
                    # Saving
                    #saver.save(self.sess,"./tenIrisSave/saveOne")
            
            self.accuracy = result
            self.trained = True
            print("Network successfuly trained")
            

        else:
            raise ValueError("NOT IMPLEMENTED YET")
        
        
        
    def get_output(self, inputs):
    	if output_function == None:
    		raise ValueError("Output function undefined")
        return Result(self, inputs, self.output_function(inputs))
        
def networks_available():
    networks = []
    folder_path = 'networks/models/'
    
    print("Loading the networks available:")
    
    for f in os.listdir(folder_path):
        if f.endswith(".pik"):
            with open(folder_path + f, "rb") as networkFile:
                n = dill.load(networkFile)
                networks.append(n)
    
    print len(networks), " networks available"
    return networks
    
def get_network(networks, name):
    for n in networks:
        if n.name == name:
            return n
