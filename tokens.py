from flask import session
import random
from string import ascii_letters, digits

def generate_form_token():
    """Sets a token to prevent double posts."""
    if '_form_token' not in session:
        form_token = \
            ''.join([random.choice(ascii_letters+digits) for i in range(32)])
        session['_form_token'] = form_token
    return session['_form_token']
