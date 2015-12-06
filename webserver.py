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
error_initialization = None
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
           
def allowedName(name):
    for n in networks:
        if str(n) == name:
            return False
    return True
        
@app.route('/', methods=['GET','POST'])
def main_page():
    debug('In main_page method')
    return render_template('index.html', networks=networks, result=result, model=new_model, error_initialization=error_initialization)
    
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
        elif request.form['btn'] == 'Delete':
            n.deleteModel()
            networks.remove(n)
        
    return redirect('/')
    
@app.route('/newmodel', methods=['POST'])
def upload():
    debug('Uploading files')
    if request.method == 'POST':
        trainingFile = request.files['training']
        testingFile = request.files['testing']
        name = request.form['name']
        global error_initialization
        if not name or not allowedName(name):
            message = "Error: the name of the network must be different than the others"
            debug(message)
            error_initialization = message
            return redirect('/')
        if not trainingFile or not testingFile:
            message = "Error: the training and testing files must be attached"
            debug(message)
            error_initialization = message
            return redirect('/')
        if not allowed_file(trainingFile.filename) or not allowed_file(testingFile.filename):
            message = "Error: the training and testing files must have the " + str(ALLOWED_EXTENSIONS) + " format"
            debug(message)
            error_initialization = message
            return redirect('/')
        trainingFilename = secure_filename(trainingFile.filename)
        testingFilename = secure_filename(testingFile.filename)
        trainingFile.save(os.path.join(app.config['UPLOAD_FOLDER'], trainingFilename))
        testingFile.save(os.path.join(app.config['UPLOAD_FOLDER'], testingFilename))
        global new_model
        new_model = network.Network(name, UPLOAD_FOLDER + '/' + trainingFilename, UPLOAD_FOLDER + '/' + testingFilename)
        debug("Model created : " + str(new_model))
        error_initialization = None
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
        global error_initialization
        global new_model
        if startInput < 0 or endInput <= startInput:
            message = "Error: the input columns are not valid"
            debug(message)
            error_initialization = message
            return redirect('/')
        if output < 0 or (output >= startInput and output <= endInput):
            message = "Error: the output column is not valid"
            debug(message)
            error_initialization = message
            return redirect('/')
        if endInput >= new_model.number_of_columns or output >= new_model.number_of_columns:
            message = "Error: the columns are in an invalid range"
            debug(message)
            error_initialization = message
            return redirect('/')
        new_model.input_start_column = startInput
        new_model.input_end_column = endInput
        new_model.output_column = output
        new_model.qualitative_outputs = qualitative
        new_model.saveModel()
        debug("Model completed")
        networks.append(deepcopy(new_model))
        new_model = None
        error_initialization = None
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
