from datetime import timedelta
from functools import update_wrapper

from jsonschema import validate, ValidationError

from flask import Flask, request, jsonify, make_response, current_app, render_template, g
from flask_bootstrap import Bootstrap
import os
import scripts.ner as ner
import scripts.models as models
import logging
import tensorflow as tf
import fastText

# create the app
app = Flask(__name__)
Bootstrap(app)
# app.logger.setLevel(logging.INFO)




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
    return ner_model_files


# get models
ner_models = {}
ner_graphs = {}
# for model_file in ["germeval.h5"]:
for model_file in get_model_files():
    app.logger.info('Loading NER model ' + model_file)
    if model_file.startswith("conll"):
        max_sequence = 100
    else:
        max_sequence = 56
    ner_models[model_file] = ner.NerModel(model_file, max_sequence)
    ner_graphs[model_file] = tf.get_default_graph()


app.logger.info('Loading fastText model.')
models.ft = fastText.load_model("embeddings/wiki.de.bin")
app.logger.info('Done.')


@app.route('/', methods=['GET'])
def list_models():
    g.model_files = get_model_files()
    return render_template('index.html')



@app.route('/api', methods=['POST'])
@crossdomain(origin='*', headers=['Content-Type'])
def annotate():

    if request.method == 'POST':

        global ner_graphs

        try:
            json_data = request.get_json()
            schema = {
                "type": "object",
                "properties": {
                    "meta": {"type": "object", "properties": {
                        "model": {"type": "string"}
                    }},
                    "data": {"type": "object", "properties": {
                        "sentences": {"type": "array", "items": {"type": "string"}, "maxItems": 5000},
                        "tokens": {"type": "array",
                                   "items": {"type": "array", "items": {"type": "string"}, "maxItems": 100},
                                   "maxItems": 5000}
                    }}
                }
            }
            validate(json_data, schema)
        except ValidationError:
            return jsonify({'errors': 'Invalid request format.'})

        metadata = json_data["meta"]
        model = metadata["model"]

        if model not in ner_models:
            return jsonify({"errors" : "Unknown model: " + model})

        json_response = {}

        if json_data["data"]["sentences"]:
            sentences = json_data["data"]["sentences"]
            sentences = [ner.tokenize(s) for s in sentences]
            sentences = [s[:ner_models[model].max_sequence_length] for s in sentences]
            # predict
            with ner_graphs[model].as_default():
                result = ner.predict(ner_models[model], sentences)
            app.logger.debug('sentences:' + str(result))
            json_response["sentences"] = result

        if json_data["data"]["tokens"]:
            sentences = json_data["data"]["tokens"]
            sentences = [s[:ner_models[model].max_sequence_length] for s in sentences]
            # predict
            with ner_graphs[model].as_default():
                result = ner.predict(ner_models[model], sentences)
            app.logger.debug('tokens:' + str(result))
            json_response["tokens"] = result

        return jsonify(json_response)


if __name__ == "__main__":
    app.run(threaded = True)


