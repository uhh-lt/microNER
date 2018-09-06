from keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras_contrib.layers import CRF
import re
import scripts.models as models
import scripts.utils as utils
import json

import numpy as np
import math
import copy


class NerModel(object):
    def __init__(self, model_file, max_sequence_length = 56):
        self.model_file = model_file
        self.model = None
        self.load_ner_model()
        self.max_sequence_length = max_sequence_length
        self.nb_embedding_dims = 300
        self.nb_char_embeddings = 52

    def create_custom_objects(self):
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
        return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "loss": loss, "accuracy": accuracy}

    def load_ner_model(self):
        # keras model
        self.model = load_model('models/' + self.model_file, custom_objects=self.create_custom_objects())
        # indexes
        indexes_file = re.sub('\.h5$', '.indexes', self.model_file)
        indexMappings = json.load(open('models/' + indexes_file, 'r', encoding='UTF-8'))
        self.idx2Label = {int(k):v for k,v in indexMappings[0].items()}
        self.label2Idx = indexMappings[1]
        self.char2Idx = indexMappings[2]
        self.case2Idx = indexMappings[3]




def tokenize(sentence):
    words = re.compile('https?://\\S+|[\\w-]+|[^\\w-]+', re.UNICODE).findall(sentence.strip())
    words = [w.strip() for w in words if w.strip() != '']
    return(words)


class NerPredictGenerator(Sequence):
    def __init__(self, sentence_data, ner_model, batch_size=32):
        self.ner_model = ner_model
        self.sentence_data = sentence_data
        self.indices = np.arange(len(self.sentence_data))
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.sentence_data) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.get_processed_data(inds)
        return batch_x, batch_y

    def get_processed_data(self, inds):

        word_embeddings = []
        case_embeddings = []
        char_embeddings = []

        output_labels = []

        for index in inds:
            sentence = copy.copy(self.sentence_data[index])

            temp_word = []
            temp_casing = []
            temp_char = []

            temp_output = []

            # padding
            sequence_length = len(sentence)
            words_to_pad = self.ner_model.max_sequence_length - sequence_length
            for i in range(words_to_pad):
                sentence.append(['PADDING_TOKEN', 'PADDING_TOKEN'])

            # create data input for words
            for w_i, word in enumerate(sentence):

                word, label = word

                temp_output.append(self.ner_model.label2Idx[label])

                casing = utils.getCasing(word, self.ner_model.case2Idx)
                temp_casing.append(casing)

                if word == 'PADDING_TOKEN':
                    temp_char2 = np.array([self.ner_model.char2Idx['PADDING_TOKEN']])
                    temp_char.append(temp_char2)
                    word_vector = [0] * self.ner_model.nb_embedding_dims
                    temp_word.append(word_vector)
                else:
                    # char
                    temp_char2 = []
                    all_chars = list(word)

                    for char in all_chars:
                        if char in self.ner_model.char2Idx.keys():
                            temp_char2.append(self.ner_model.char2Idx[char])
                        else:
                            temp_char2.append(self.ner_model.char2Idx['UNKNOWN'])
                    temp_char2 = np.array(temp_char2)
                    temp_char.append(temp_char2)

                    # word
                    word_vector = models.ft.get_word_vector(word.lower())
                    temp_word.append(word_vector)

            temp_char = pad_sequences(temp_char, self.ner_model.nb_char_embeddings)
            word_embeddings.append(temp_word)
            case_embeddings.append(temp_casing)
            char_embeddings.append(temp_char)

            temp_output = to_categorical(temp_output, len(self.ner_model.label2Idx))
            output_labels.append(temp_output)

        return ([np.array(word_embeddings),
                     np.array(case_embeddings),
                     np.array(char_embeddings)],
                    np.array(output_labels))




def predict(ner_model, document):

    predict_sentences = []
    for sentence in document:
        tokens = []
        for t in sentence:
            tokens.append([t, 'O'])
        predict_sentences.append(tokens)

    _, predicted_labels = predict_sequences(ner_model, predict_sentences, ner_model.label2Idx)

    predicted_sentences = []
    for i_s, sentence in enumerate(document):
        tokens = []
        for i_t, t in enumerate(sentence):
            tokens.append([t, ner_model.idx2Label[predicted_labels[i_s][i_t]]])
        predicted_sentences.append(tokens)

    return(predicted_sentences)


def predict_sequences(ner_model, sentences, label2Idx, level2=False):
    steps = 0
    true_labels = []
    pred_labels = []
    all_true_labels = []
    for s in sentences:
        if level2:
            all_true_labels.append([w[2] for w in s])
        else:
            all_true_labels.append([w[1] for w in s])
    all_pred_labels = ner_model.model.predict_generator(NerPredictGenerator(sentences, ner_model))

    for s_id, s in enumerate(all_true_labels):
        not_padded_true = []
        not_padded_pred = []
        predicted_labels = utils.get_label_from_categorical(all_pred_labels[s_id])
        for t_id, t in enumerate(s):
            if t != 'PADDING_TOKEN':  # skip PADDING_TOKEN
                not_padded_true.append(label2Idx[t])
                not_padded_pred.append(predicted_labels[t_id])
        true_labels.append(not_padded_true)
        pred_labels.append(not_padded_pred)

    return (true_labels, pred_labels)