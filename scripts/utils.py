from scripts.validation import compute_f1
from keras.utils import Sequence, to_categorical
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import math
import scripts.models as models
import copy
import json

def getCasing(word, caseLookup):
    
    if word == 'PADDING_TOKEN':
        return(caseLookup['PADDING_TOKEN'])
    
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
   
    return caseLookup[casing]

# changing all -deriv to MISC. Removing all -part 
def modify_labels(dataset):
    bad_labels = ['I-PERderiv','I-OTHpart','B-ORGderiv', 'I-OTH','B-OTHpart','B-LOCderiv','I-LOCderiv','I-OTHderiv','B-PERderiv','B-OTHderiv','B-PERpart','I-PERpart','I-LOCpart','B-LOCpart','I-ORGpart','I-ORGderiv','B-ORGpart','B-OTH']
    for sentence in dataset:
        for word in sentence:
            label = word[1]
            if label.endswith('part'):
                word[1] = 'O'
                continue
            if label in bad_labels:
                first_char = label[0]
                if first_char == 'B' :
                    word[1] = 'B-MISC'
                else:
                    word[1] = 'I-MISC'
    return dataset
   
    
def get_sentences_germeval(path, level2 = False):
    sentences = []
    with open(path, 'r', encoding = 'UTF-8') as f:
        sentence = []
        for line in f:
            
            line = line.strip()
            
            # append sentence
            if len(line) == 0:
                if len(sentence):
                    sentences.append(sentence)
                sentence = []
                continue
            
            # get sentence tokens
            splits = line.split()
            if splits[0] == '#':
                continue
            if level2:
                # word, label 1st level, label 2nd level
                # temp = [splits[1],splits[2],splits[3]]
                temp = [splits[1],splits[3]]
            else:
                # word, label 1st level
                temp = [splits[1],splits[2]]
            sentence.append(temp)
        
        # append last
        if len(sentence):
            sentences.append(sentence)    
    return sentences


# preproecessing data from Conll
def get_sentences_conll(filename, cut_len = 100):
    '''
        -DOCSTART- -X- -X- O

    EU NNP B-NP B-ORG
    rejects VBZ B-VP O
    German JJ B-NP B-MISC
    call NN I-NP O
    to TO B-VP O
    boycott VB I-VP O
    British JJ B-NP B-MISC
    lamb NN I-NP O
    . . O O
    
    '''
    
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename,'r', encoding='UTF-8')
    sentences = []
    sentence = []
    for line in f:
        line = line.strip()
        
        if len(line) == 0:
            if len(sentence):
                sentences.append(sentence[:cut_len])
            sentence=[]
            continue
        
        splits = line.split()
        
        word=splits[0]
        if word=='-DOCSTART-':
            continue
        label=splits[-1]
        temp=[word,label]
        sentence.append(temp)
    f.close()
    return sentences



def get_label_from_categorical(a):
    labels = []
    for label in a:
        label = np.ndarray.tolist(label)
        label = np.argmax(label)
        labels.append(label)
    return(labels)

def predict_sequences(model, sentences, level2 = False):
    steps = 0
    true_labels = []
    pred_labels = []
    all_true_labels = []
    for s in sentences:
        if level2:
            all_true_labels.append([w[2] for w in s])
        else:
            all_true_labels.append([w[1] for w in s])
    all_pred_labels = model.predict_generator(NerSequence(sentences, level2 = level2))
    
    for s_id, s in enumerate(all_true_labels):
        not_padded_true = []
        not_padded_pred = []
        predicted_labels = get_label_from_categorical(all_pred_labels[s_id])
        for t_id, t in enumerate(s):
            if t != 'PADDING_TOKEN': # skip PADDING_TOKEN 
                not_padded_true.append(models.label2Idx[t])
                not_padded_pred.append(predicted_labels[t_id])
        true_labels.append(not_padded_true)
        pred_labels.append(not_padded_pred)
    
    return(true_labels, pred_labels)


class NerSequence(Sequence):
    def __init__(self, sentence_data, shuffle_data = False, batch_size=32, level2 = False, add_marks = False):
        self.sentence_data = sentence_data
        self.shuffle_data = shuffle_data
        self.batch_size = batch_size
        self.level2 = level2
        self.add_marks = add_marks
        self.indices = np.arange(len(self.sentence_data))
        if self.shuffle_data:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.sentence_data) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = self.get_processed_data(inds)
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle_data:
            np.random.shuffle(self.indices)
        
    def get_processed_data(self, inds):
        word_embeddings = []
        case_embeddings = []
        char_embeddings = []
        nerlevel1_embeddings = []
        
        output_labels = []

        for index in inds: 
            sentence = copy.copy(self.sentence_data[index])

            temp_word= []
            temp_casing = []
            temp_char= []
            
            # for 2nd level prediction: take 1st level (outer) as additional input
            temp_nerlevel1=[]

            temp_output=[]

            # padding
            sequence_length = len(sentence)
            words_to_pad = models.max_sequence_length - sequence_length
            for i in range(words_to_pad):
                if self.level2:
                    sentence.append(['PADDING_TOKEN', 'PADDING_TOKEN', 'PADDING_TOKEN'])
                else:
                    sentence.append(['PADDING_TOKEN', 'PADDING_TOKEN'])

            # create data input for words
            for w_i, word in enumerate(sentence):
                
                if self.level2:
                    word, label_outer, label = word
                    temp_nerlevel1.append(models.labelOuter2Idx[label_outer])
                else:
                    word, label = word
                
                temp_output.append(models.label2Idx[label])

                casing = getCasing(word, models.case2Idx)
                temp_casing.append(casing)

                if word == 'PADDING_TOKEN':
                    temp_char2=np.array([models.char2Idx['PADDING_TOKEN']])
                    temp_char.append(temp_char2)
                    word_vector = [0] * models.nb_embedding_dims
                    temp_word.append(word_vector)
                else:
                    # char
                    temp_char2=[]
                    all_chars = list(word)
                    
                    if self.add_marks:
                        all_chars.insert(0, "<W>")
                        all_chars.append("</W>")

                        if w_i == 0:
                            all_chars.insert(0, "<S>")

                        if w_i == (sequence_length - 1):
                            all_chars.append("</S>")
                    # print(all_chars)
                    
                    for char in all_chars:
                        if char in models.char2Idx.keys():
                            temp_char2.append(models.char2Idx[char])
                        else:
                            temp_char2.append(models.char2Idx['UNKNOWN'])
                    temp_char2 = np.array(temp_char2)
                    temp_char.append(temp_char2)

                    # word
                    word_vector = models.ft.get_word_vector(word.lower())
                    # word_vector = models.ft.get_word_vector(word)
                    temp_word.append(word_vector)

            temp_char = pad_sequences(temp_char, models.nb_char_embeddings)
            word_embeddings.append(temp_word)
            case_embeddings.append(temp_casing)
            char_embeddings.append(temp_char)
            
            nerlevel1_embeddings.append(temp_nerlevel1)
            
            temp_output = to_categorical(temp_output, len(models.label2Idx))
            output_labels.append(temp_output)

        if self.level2:
            return([np.array(word_embeddings), 
                    np.array(nerlevel1_embeddings),
                    np.array(case_embeddings), 
                    np.array(char_embeddings)], 
                   np.array(output_labels))
        else:
            return([np.array(word_embeddings), 
                    np.array(case_embeddings), 
                    np.array(char_embeddings)], 
                   np.array(output_labels))


class F1History(Callback):
    def __init__(self, model_file, devSet, level2 = False):
        self.model_file = model_file
        self.devSet = devSet
        self.level2 = level2
        self.max_f1 = 0
        
    def save_model(self):
        self.model.save(self.model_file)
        with open(self.model_file + ".indexes", "w") as f:
            json.dump([models.idx2Label, models.label2Idx, models.char2Idx, models.case2Idx], f)
                 
    def on_train_begin(self, logs={}):
        self.acc = []
        self.f1_scores = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('val_acc'))
        true_labels, pred_labels = predict_sequences(self.model, self.devSet, level2 = self.level2)
        pre, rec, f1 = compute_f1(pred_labels, true_labels, models.idx2Label)
        self.f1_scores.append(f1)
        if epoch > -1 and f1 > self.max_f1:
            print("\nNew maximum F1 score: " + str(f1) + " (before: " + str(self.max_f1) + ") Saving to " + self.model_file)
            self.max_f1 = f1
            self.save_model()

