from flask import Flask, render_template, request, redirect, session
from copy import deepcopy
import threading
import os
from networks import network
from werkzeug import secure_filename

DEBUG = True
ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_FOLDER = 'upload'

networks = network.networks_available()
new_model = None
result = None
app = Flask(__name__)
app.secret_key = "f29jfd9fj903-0ld"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = DEBUG

def debug(s):
    if DEBUG:
        print('DEBUG:\t' + s)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def train_and_redirect(network):
    debug("Training network")
    network.train()
    return redirect('/')
        
@app.route('/', methods=['GET','POST'])
def main_page():
    debug('In main_page method')
    return render_template('index.html', networks=networks, result=result, model=new_model)
    
@app.route('/train', methods=['POST'])
def modelSubmission():
    debug('In model submission method')
    if request.method == 'POST':
    
        n = network.get_network(networks, request.form['network'])
        if n == None:
            debug("Error, can't find the network")
            return
            
        if request.form['btn'] == 'Train':
            debug("Training network")
            method = request.form['method']
            number_of_epochs = int(request.form['epochs'])
            stop_at_100_accuracy = request.form.getlist('100')
            if not n.initialized:
                n.initialize()
                n.initialized = True
            n.train(number_of_epochs=number_of_epochs, stop_at_100_accuracy=stop_at_100_accuracy)
             
            #todo : update progress
            #t = threading.Thread(target=train_and_redirect, args=(n,))
            #t.start()
        elif request.form['btn'] == 'Reset':
            n.initialized = False
            n.best_accuracy = 0
            n.accuracy = 0
            n.epochs_for_best_accuracy = 0
            n.epochs_for_accuracy = 0
            n.trained = False
        
        
    return redirect('/')
    
@app.route('/newmodel', methods=['POST'])
def upload():
    debug('Uploading files')
    if request.method == 'POST':
        trainingFile = request.files['training']
        testingFile = request.files['testing']
        name = request.form['name']
        if not name:
            #todo: throw error
            debug("error, no name")
            return
        if not trainingFile or not testingFile:
            #todo : throw error
            debug("error, no file")
            return
        if not allowed_file(trainingFile.filename) or not allowed_file(testingFile.filename):
        	#todo : throw error
        	debug("Wrong file format")
        	return
        trainingFilename = secure_filename(trainingFile.filename)
        testingFilename = secure_filename(testingFile.filename)
        trainingFile.save(os.path.join(app.config['UPLOAD_FOLDER'], trainingFilename))
        testingFile.save(os.path.join(app.config['UPLOAD_FOLDER'], testingFilename))
        global new_model
        new_model = network.Network(name, UPLOAD_FOLDER + '/' + trainingFilename, UPLOAD_FOLDER + '/' + testingFilename)
        debug("Model created : " + str(new_model))
        return redirect('/')
    return redirect('/')


@app.route('/finishmodel', methods=['POST'])
def finishModel():
    debug('Finishing the model')
    if request.method == 'POST':
        startInput = int(request.form['startInput'])
        endInput = int(request.form['endInput'])
        output = int(request.form['output'])
        qualitative = checked = 'qualitative' in request.form
        global new_model
        if startInput < 0 or endInput <= startInput:
            #todo : display error
            debug("Error, The input columns are not valid")
            return
        if output < 0 or (output >= startInput and output <= endInput):
            #todo : display error
            debug("Error, The output column is not valid")
            return
        if endInput >= new_model.number_of_columns or output >= new_model.number_of_columns:
            #todo : display error
            debug("Error, The columns are in an invalid range")
            return
        new_model.input_start_column = startInput
        new_model.input_end_column = endInput
        new_model.output_column = output
        new_model.qualitative_outputs = qualitative
        new_model.saveModel()
        debug("Model completed")
        networks.append(deepcopy(new_model))
        new_model = None
    return redirect('/')
    
@app.route('/output/<path:path>', methods=['POST'])
def output(path):
    debug('In output method')
    if request.method == 'POST':
        n = network.get_network(networks, path)
        debug('network = ' + str(n))
        inputs = []
        for i in range(n.number_of_inputs):
            inputs.append(request.form[str(i)])
            debug('input ' + str(i) + ' = ' + request.form[str(i)])
        global result
        result = n.get_output(map(int, inputs))
        
    return redirect('/')

if __name__ == "__main__":
    app.run(use_reloader=False)
    #session.clear()
