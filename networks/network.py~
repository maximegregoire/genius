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
        self.trained = False
        self.in_training = False
        self.input_start_column = 0
        self.input_end_column = 0
        self.output_column = 0
        self.number_of_columns = 0
        self.reshaped = False
        
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
        
        self._index_in_epoch = 0
        
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
        outputs, outputs = parsing.convertOneHot(data, self.output_column, max_output)
        return (outputs, outputs)
        
    def extractData(self):
        self.training_data, self.training_outputs = parsing.parse(self.training_path, qualitative = self.qualitative_outputs, output_column = self.output_column)
        self.testing_data, self.testing_outputs = parsing.parse(self.testing_path, qualitative = self.qualitative_outputs, output_column = self.output_column)
        print "len(training_outputs) = ", len(self.training_outputs)
        print "len(training_outputs[0]) = ", len(self.training_outputs[0])
        return 1
    
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
            self.training_outputs, self.training_outputs = self.getQualitativeOutputs(self.training_data, max_output)
            self.testing_outputs, self.testing_outputs = self.getQualitativeOutputs(self.testing_data, max_output)
            self.training_data = self.training_data[:,[x for x in xrange(self.input_start_column, self.input_end_column+1)]]
            self.testing_data = self.testing_data[:,[x for x in xrange(self.input_start_column,self.input_end_column+1)]]
        else:
            self.training_data = np.genfromtxt(self.training_path, delimiter=self.delimitation)
            self.testing_data = np.genfromtxt(self.testing_path, delimiter=self.delimitation)
            self.training_outputs = self.training_data[:,[self.output_column]]
            self.testing_outputs = self.testing_data[:,[self.output_column]]
            self.training_data = self.training_data[:,[x for x in xrange(self.input_start_column, self.input_end_column+1)]]
            self.testing_data = self.testing_data[:,[x for x in xrange(self.input_start_column,self.input_end_column+1)]]
        
    def getRelativePath(self):
        return 'networks/models/' + self.name + '.pik'
        
    def saveModel(self):
        with open(self.getRelativePath(), "wb") as f:
            dill.dump(self, f)
            
    def deleteModel(self):
        if not os.path.isfile(self.getRelativePath()):
            raise ValueError("The model file is not present")
        os.remove(self.getRelativePath())
            
    def initialize(self, method, learning_rate_multilayer, layers):
        self.accuracy = 0
        self.best_accuracy = 0
        self.epochs_for_accuracy = 0
        self.epochs_for_best_accuracy = 0
        if method == "Logistic regression":
            self.initializeLogistic()
            self.method_initialized = method
            self.initialized = True
        elif method == "K-nearest neighbors":
            self.initializeKnn()
            self.method_initialized = method
            self.initialized = True
        elif method == "Multilayer perceptron":
            #self.initialize2layer(learning_rate_multilayer)
            #self.initializeOldMultilayer(learning_rate_multilayer, layers)
            self.initializeDeep(learning_rate_multilayer, layers)
            self.method_initialized = method
            self.initialized = True
        else:
            #todo: error in training
            raise ValueError("Training method not recognized")
            
    def initializeDeep(self, learning_rate, layers):
        if layers <= 0:
            raise ValueError("The number of layers must be greater than zero")
    
        if self.qualitative_outputs:
            # Network Parameters
            hidden_features = []
            for i in range(layers):
                hidden_features.append(100)
                
            n_input = self.input_end_column - self.input_start_column + 1
            n_classes = max(len(self.training_outputs[0]),len(self.testing_outputs[0]))
            print "n_classes = ", n_classes

            # tf Graph input
            self.x = tf.placeholder("float", [None, n_input])
            self.y = tf.placeholder("float", [None, n_classes])

            # Create model
            def multilayer_perceptron(_X, _weights, _biases, _number_of_layers):
                layers = []
                layers.append(tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0])))
                
                for i in range(_number_of_layers-1):
                    layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i], _weights[i+1]), _biases[i+1])))
          
                return tf.matmul(layers[_number_of_layers-1], _weights[_number_of_layers]) + _biases[_number_of_layers]

            # Store layers weight & bias
            weights = []
            weights.append(tf.Variable(tf.random_normal([n_input, hidden_features[0]])))
            
            for i in range(layers - 1):
                weights.append(tf.Variable(tf.random_normal([hidden_features[i], hidden_features[i+1]])))
                
            weights.append(tf.Variable(tf.random_normal([hidden_features[layers-1], n_classes])))
            
            biases = []
            
            for i in range(layers):
                biases.append(tf.Variable(tf.random_normal([hidden_features[i]])))
                
            biases.append(tf.Variable(tf.random_normal([n_classes])))
            
            # Construct model
            self.pred = multilayer_perceptron(self.x, weights, biases, layers)

            # Define loss and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y)) # Softmax loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost) # Adam Optimizer

            # Initializing the variables
            init = tf.initialize_all_variables()

            # Launch the graph
            self.sess = tf.Session()
            self.sess.run(init)
            
    def initialize2layer(self, learning_rate):
        if self.qualitative_outputs:
            # Network Parameters
            n_hidden_1 = 256 # 1st layer num features
            n_hidden_2 = 256 # 2nd layer num features
                                    
            n_input = self.input_end_column - self.input_start_column + 1
            n_classes = max(len(self.training_outputs[0]),len(self.testing_outputs[0]))
            
            # tf Graph input
            self.x = tf.placeholder("float", [None, n_input], name="x")
            self.y = tf.placeholder("float", [None, n_classes], name="y")

            # Create model
            def multilayer_perceptron(_X, _weights, _biases):
                layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
                layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
                return tf.matmul(layer_2, _weights['out']) + _biases['out']

            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

            # Construct model
            self.pred = multilayer_perceptron(self.x, weights, biases)

            # Define loss and optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y)) # Softmax loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost) # Adam Optimizer

            # Initializing the variables
            init = tf.initialize_all_variables()

            # Launch the graph
            self.sess = tf.Session()
            self.sess.run(init)
            
    def initializeKnn(self):        
        if self.qualitative_outputs:            
            n_input = self.input_end_column - self.input_start_column + 1            
            self.tf_in = tf.placeholder("float", [None, n_input])
            self.tf_testing = tf.placeholder("float", [n_input])
            
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
                if np.argmax(self.training_outputs[nn_index]) == np.argmax(self.testing_outputs[i]):
                    accuracy += 1./len(self.testing_data)
            self.accuracy = accuracy
            self.epochs_for_accuracy = "N/A"
            self.best_accuracy = "N/A"
            self.epochs_for_best_accuracy = "N/A"
            self.trained = True
        else:
            raise ValueError("NOT IMPLEMENTED")
    
    def initializeLogistic(self):        
        if self.qualitative_outputs:
            number_of_outputs = max(len(self.training_outputs[0]),len(self.testing_outputs[0])) 
            n_input = self.input_end_column - self.input_start_column + 1
            
            # the placeholder for the inputs
            self.x = tf.placeholder("float", [None, n_input], name="The_raw_data")

            # the placeholder for the outputs
            self.tf_softmax_correct = tf.placeholder("float", [None, number_of_outputs], name="The_correct_data")

            self.tf_weight = tf.Variable(tf.zeros([n_input, number_of_outputs]))
            self.tf_bias = tf.Variable(tf.zeros([number_of_outputs]))
            self.pred = tf.nn.softmax(tf.matmul(self.x,self.tf_weight) + self.tf_bias)

            # Training via backpropagation
            self.tf_cross_entropy = -tf.reduce_sum(self.tf_softmax_correct*tf.log(self.pred))

            # Train using tf.train.GradientDescentOptimizer
            self.tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.tf_cross_entropy)

            # Add accuracy checking nodes
            self.correct_prediction = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.tf_softmax_correct,1))
            self.tf_accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
            
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
    
    def train(self, method, number_of_epochs_logistic, number_of_epochs_multilayer, stop_at_100_accuracy):
        if method != self.method_initialized:
            raise ValueError("The current training method does not match the initialization method")
        if method == "Logistic regression":
            self.trainLogistic(number_of_epochs_logistic, stop_at_100_accuracy)
        elif method == "K-nearest neighbors":
            # No training to do, as KNN is not training-based
            return
        elif method == "Multilayer perceptron":
            self.trainMultilayer(number_of_epochs_multilayer, stop_at_100_accuracy)
        else:
            #todo: error in training
            raise ValueError("Training method not recognized")
            
    
    def trainMultilayer(self, number_of_epochs, stop_at_100_accuracy):
        display_step = 1
        accuracy = 0
        if (self.reshaped == False):
            #self.training_data = self.training_data.reshape(len(self.training_data), len(self.training_data[0][0]))
            #self.testing_data = self.testing_data.reshape(len(self.testing_data), len(self.testing_data[0][0]))
            self.reshaped = True
        for epoch in range(number_of_epochs):
            avg_cost = 0.
            # Fit training using batch data
            self.sess.run(self.optimizer, feed_dict={self.x: self.training_data, self.y: self.training_outputs})
            # Compute average loss
            avg_cost += self.sess.run(self.cost, feed_dict={self.x: self.training_data, self.y: self.training_outputs})/len(self.training_data)                
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            tf_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy = tf_accuracy.eval({self.x: self.testing_data, self.y: self.testing_outputs}, session=self.sess)
            print "Epoch {}, Accuracy = {}, \t Average cost = {}".format(epoch, accuracy, avg_cost)
            self.epochs_for_accuracy += 1
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.epochs_for_best_accuracy = self.epochs_for_accuracy
            if stop_at_100_accuracy and accuracy == 1:
                break
                
        #self.getOutputMultilayer()
        self.accuracy = accuracy    
        self.trained = True
           
        
    def trainLogistic(self, number_of_epochs, stop_at_100_accuracy):
        if self.qualitative_outputs:
            # Run the training

            k=[]
            saved=0
            result=0
            for i in range(number_of_epochs):
                self.sess.run(self.tf_train_step, feed_dict={self.x: self.training_data, self.tf_softmax_correct: self.training_outputs})
                # Print accuracy
                result = self.sess.run(self.tf_accuracy, feed_dict={self.x: self.testing_data, self.tf_softmax_correct: self.testing_outputs})
                print "Epoch {}\t Accuracy = {}".format(i, result)
                k.append(result)
                self.epochs_for_accuracy += 1
                if result > self.best_accuracy:
                    self.best_accuracy = result
                    self.epochs_for_best_accuracy = self.epochs_for_accuracy
                
                if stop_at_100_accuracy and result == 1 and saved == 0:
                    break
                    saved=1
                    # Saving
                    #saver.save(self.sess,"./tenIrisSave/saveOne")
            
            #self.getOutputMultilayer()
            self.accuracy = result
            self.trained = True            

        else:
            raise ValueError("NOT IMPLEMENTED YET")
        
    def getOutputMultilayer(self):
        inputs = np.array([0.993907341,0.726324016,0.99736115,0.677222966,0.654576652,0.48444006,0.855064044,0.995660934,0.991843484,0.677317723,0.71134155,0.66282646,0.497109384,0.922318371,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        inputs = inputs.reshape(1, 38)
        print "predictions ", self.pred.eval(feed_dict={self.x: inputs}, session=self.sess)
        
    def get_output(self, inputs):
    	#if output_function == None:
    	#	raise ValueError("Output function undefined")
        #todo: fix
        return Result(self, inputs, getOutputMultilayer(inputs))
    
def networks_available():
    networks = []
    folder_path = 'networks/models/'
        
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
