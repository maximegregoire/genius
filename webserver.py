from flask import Flask, render_template, request, redirect
import threading
from networks import network

DEBUG = True

networks = network.networks_available()
result = None
app = Flask(__name__)
app.debug = DEBUG

def debug(s):
    if DEBUG:
        print('DEBUG:\t' + s)

def train_and_redirect(network):
    network.train()
    redirect('/')
        
@app.route('/', methods=['GET','POST'])
def main_page():
    debug('In main_page method')
    return render_template('index.html', networks=networks, result=result)
    
@app.route('/train', methods=['POST'])
def train():
    debug('In train method')
    if request.method == 'POST':
        n = network.get_network(networks, request.form['network'])
        if not n.trained:
            debug('Training' + n.name + 'network')
            train_and_redirect(n)
            t = threading.Thread(target=train_and_redirect, args=(n,))
            t.start()
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
    app.run()