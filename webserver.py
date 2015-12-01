from flask import Flask, render_template, request, redirect
import threading
import xor_network
import network

DEBUG = True

networks = network.networks_available()
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
    return render_template('index2.html', networks=networks)
    
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

if __name__ == "__main__":
    app.run()