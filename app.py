from datetime import timedelta
from functools import update_wrapper

from flask import Flask, request, jsonify, make_response, current_app, render_template, g
from flask_bootstrap import Bootstrap
import os
import scripts.ner as ner


# create the app
app = Flask(__name__)
Bootstrap(app)


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response



def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    """See: http://flask.pocoo.org/snippets/56/"""
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


def get_model_files():
    ner_model_files = [name for name in os.listdir("models") if name.endswith('h5')]
    return(ner_model_files)



@app.route('/', methods=['GET'])
def list_models():
    g.model_files = get_model_files()
    return render_template('index.html')



@app.route('/api', methods=['POST'])
@crossdomain(origin='*', headers=['Content-Type'])
def annotate():

    if request.method == 'POST':

        try:
            json_data = request.get_json()
            #metadata = json_data["meta"]
            #model = metadata["model"]
            #tokenize = metadata["tokenize"]  # type/value, default
            #sentences = json_data["data"]

            sample_data = ['Ich bin Sabine Müllers erster CDU-Generalsekretär.',
                           'Tom F. Manteufel wohnt auf der Insel Borkum.']
            sentences = [ner.tokenize(s) for s in sample_data]
            model = 'germeval.h5'

        except TypeError:
            return jsonify({'error': 'Invalid request format.'})

        if not 'model_files' in g:
            g.model_files = get_model_files()

        if model not in g.model_files:
            return jsonify({'error' : 'Unknown model: ' + model})

        app.logger.info('Loading model ...')

        # load keras model
        if not 'ner_model' in g:
            g.ner_model = ner.load_ner_model(model)

        app.logger.info('Model loaded.')

        # if tokenize == true tokenize

        # predict

        result = ner.predict(g.ner_model, sentences)

        app.logger.info('Result:' + str(result))

        json_response = {"data": jsonify(result)}

        return jsonify(json_response)


if __name__ == "__main__":
    app.run(threaded = True)


