from keras.models import load_model
from keras_contrib.layers import CRF
import re
import scripts.models as models
import scripts.utils as utils
import json
import fastText

def create_custom_objects():
    instanceHolder = {"instance": None}
    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)
    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)
    return {"ClassWrapper": ClassWrapper ,"CRF": ClassWrapper, "loss": loss, "accuracy":accuracy}

def load_model_indexes(indexes_file, max_sequence_length = 56):
    indexMappings = json.load(open(indexes_file, 'r', encoding='UTF-8'))
    models.idx2Label = {int(k):v for k,v in indexMappings[0].items()}
    models.label2Idx = indexMappings[1]
    models.char2Idx = indexMappings[2]
    models.case2Idx = indexMappings[3]
    models.max_sequence_length = max_sequence_length


def load_ner_model(model_file):
    if not models.ft:
        models.ft = fastText.load_model("embeddings/wiki.de.bin")
    indexes_file = re.sub('\.h5$', '.indexes', model_file)
    load_model_indexes('models/' + indexes_file)
    ner_model = load_model('models/' + model_file, custom_objects=create_custom_objects())
    return(ner_model)


def tokenize(sentence):
    words = re.compile('https?://\\S+|[\\w-]+|[^\\w-]+', re.UNICODE).findall(sentence.strip())
    words = [w.strip() for w in words if w.strip() != '']
    return(words)

def predict(ner_model, document):

    predict_sentences = []
    for sentence in document:
        tokens = []
        for t in sentence:
            tokens.append([t, 'O'])
        predict_sentences.append(tokens)

    _, predicted_labels = utils.predict_sequences(ner_model, predict_sentences)

    predicted_sentences = []
    for i_s, sentence in enumerate(document):
        tokens = []
        for i_t, t in enumerate(sentence):
            tokens.append([t, models.idx2Label[predicted_labels[i_s][i_t]]])
        predicted_sentences.append(tokens)

    return(predicted_sentences)