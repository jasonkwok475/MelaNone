from flask import Flask, render_template
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='', template_folder='../frontend')

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/api/hello')
def hello():
    return {'message': 'Hello from Flask!'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
