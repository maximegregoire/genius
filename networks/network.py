import parsing
from XOR_Operator import xor_network

import os.path
import numpy as np
import tensorflow as tf
import dill

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
        self.accuracy = 0
        self.best_accuracy = 0
        self.epochs_for_best_accuracy = 0
        self.epochs_for_accuracy = 0
        
        self.sess = None
        self.initialized = False
        self.summary_writer = None
        self.tf_train_step = None
        self.tf_accuracy = None
        self.summary_op = None
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
        training_data = np.genfromtxt(self.training_path, delimiter=self.delimitation)
        self.number_of_columns = len(training_data[0])
        return training_data[0:number_of_lines]
        
    def getInputs(self, data):
        if data == None:
            raise ValueError("The data has not been loaded")
        self.inputs = np.array([i[input_start_column:input_end_column + 1] for i in data])
        
    def getQualitativeOutputs(self, data):
        if data == None:
            raise ValueError("The data has not been loaded")
        outputs, outputs_onehot = parsing.convertOneHot(data, self.output_column)
        return (outputs, outputs_onehot)
        
    def getRelativePath(self):
        return 'networks/models/' + self.name + '.pik'
        
    def saveModel(self):
        with open(self.getRelativePath(), "wb") as f:
            dill.dump(self, f)
            
    def deleteModel(self):
        if not os.path.isfile(self.getRelativePath()):
            raise ValueError("The model file is not present")
        os.remove(self.getRelativePath())
            
    def initialize(self):
        training_data = np.genfromtxt(self.training_path, delimiter=self.delimitation)        
        testing_data = np.genfromtxt(self.testing_path, delimiter=self.delimitation)
        
        if self.qualitative_outputs:
            training_outputs, training_outputs_onehot = self.getQualitativeOutputs(training_data)
            testing_outputs, testing_outputs_onehot = self.getQualitativeOutputs(testing_data)
            number_of_outputs = len(testing_outputs_onehot[0])
            # the placeholder for the inputs
            self.tf_in = tf.placeholder("float", [None, self.number_of_columns], name="The_raw_data")

            # the placeholder for the outputs
            self.tf_softmax_correct = tf.placeholder("float", [None, number_of_outputs], name="The_correct_data")

            tf_weight = tf.Variable(tf.zeros([self.number_of_columns, number_of_outputs]))
            tf_bias = tf.Variable(tf.zeros([number_of_outputs]))
            tf_softmax = tf.nn.softmax(tf.matmul(self.tf_in,tf_weight) + tf_bias)

            # Training via backpropagation
            tf_cross_entropy = -tf.reduce_sum(self.tf_softmax_correct*tf.log(tf_softmax))

            # Train using tf.train.GradientDescentOptimizer
            self.tf_train_step = tf.train.GradientDescentOptimizer(0.01).minimize(tf_cross_entropy)

            # Add accuracy checking nodes
            tf_correct_prediction = tf.equal(tf.argmax(tf_softmax,1), tf.argmax(self.tf_softmax_correct,1))
            self.tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))
            
            # Recreate logging dir
            import shutil, os, sys
            TMPDir='./tenIrisSave'
            try:
             shutil.rmtree(TMPDir)
            except:
             print "Tmp Dir did not exist"
            os.mkdir(TMPDir, 0755 )
            
            # Initialize and run
            tf.Session().close()
            self.sess = tf.Session()
            #sess = tf.InteractiveSession()
            init = tf.initialize_all_variables()
            self.sess.run(init)

            # Build the summary operation based on the TF collection of Summaries.
            tf.train.write_graph(self.sess.graph_def, TMPDir + '/logsd','graph.pbtxt')
            
            tf.scalar_summary("Accuracy:", self.tf_accuracy)
            tf.histogram_summary('weights', tf_weight)
            tf.histogram_summary('bias', tf_bias)
            tf.histogram_summary('softmax', tf_softmax)
            tf.histogram_summary('accuracy', self.tf_accuracy)
            
            self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.train.SummaryWriter(TMPDir + '/logs',self.sess.graph_def)

            # This is for saving all our work
            saver = tf.train.Saver([tf_weight,tf_bias])
        else:
            raise ValueError("NOT IMPLEMENTED")
        
    def train(self, number_of_epochs, stop_at_100_accuracy):
        training_data = np.genfromtxt(self.training_path, delimiter=self.delimitation)        
        testing_data = np.genfromtxt(self.testing_path, delimiter=self.delimitation)
        
        if self.qualitative_outputs:
            training_outputs, training_outputs_onehot = self.getQualitativeOutputs(training_data)
            testing_outputs, testing_outputs_onehot = self.getQualitativeOutputs(testing_data)
            number_of_outputs = len(testing_outputs_onehot[0])

            # Run the training

            k=[]
            saved=0
            result=0
            for i in range(number_of_epochs):
                self.sess.run(self.tf_train_step, feed_dict={self.tf_in: training_data, self.tf_softmax_correct: training_outputs_onehot})
                # Print accuracy
                result = self.sess.run(self.tf_accuracy, feed_dict={self.tf_in: testing_data, self.tf_softmax_correct: testing_outputs_onehot})
                print "Run {},{}".format(i,result)
                k.append(result)
                #summary_str = self.sess.run(self.summary_op,feed_dict={self.tf_in: testing_data, self.tf_softmax_correct: testing_outputs_onehot})
                #self.summary_writer.add_summary(summary_str, i)
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
