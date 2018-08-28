from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,GlobalMaxPooling1D,Flatten,concatenate
from keras.initializers import RandomUniform
from keras_contrib.layers import CRF
import numpy as np

# global variables
label2Idx = None
idx2Label = None
case2Idx = None
char2Idx = None
idx2Char = None
max_sequence_length = None
# fasttext model
ft = None
ft_char_embeddings = None
# token embedding dims
nb_embedding_dims = 300
# char embedding dims
nb_char_embeddings = 52
# 2nd level prediction
labelOuter2Idx = None
idx2LabelOuter = None
# model


def get_model_lstm():
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
    words_input = Input(shape=(None, nb_embedding_dims), dtype='float32', name='words_input')
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False, name = 'case_embed')(casing_input)
    character_input=Input(shape=(None,nb_char_embeddings,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),32,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    char_lstm = TimeDistributed(Bidirectional(LSTM(50, name = 'char_lstm')))(embed_char_out)
    output = concatenate([words_input, casing, char_lstm])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.5, name = 'token_lstm'))(output)
    output = TimeDistributed(Dense(len(label2Idx), name = 'token_dense'))(output)
    crf = CRF(len(label2Idx), name = 'crf')
    output = crf(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss=crf.loss_function, optimizer='nadam', metrics=[crf.accuracy])
    model.summary()
    return(model)

def get_model_3cnn():
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
    words_input = Input(shape=(None, nb_embedding_dims), dtype='float32', name='words_input')
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False, name = 'case_embed')(casing_input)
    character_input=Input(shape=(None,nb_char_embeddings,),name='char_input')
    
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),32,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    # embed_char_out=TimeDistributed(Embedding(len(char2Idx),32,weights=[ft_char_embeddings], trainable=False, name='char_embedding'))(character_input)
    
    kernel_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in kernel_sizes:
        conv = TimeDistributed(Conv1D(
                             kernel_size=sz,
                             filters=32,
                             padding="same",
                             activation="relu",
                             strides=1,
                             name='charcnn_' + str(sz)))(embed_char_out)
        # conv = TimeDistributed(MaxPooling1D(3, name = 'charcnn_maxpool'))(conv)
        conv = TimeDistributed(GlobalMaxPooling1D(name = 'charcnn_maxpool'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        conv_blocks.append(conv)
    output = concatenate([words_input, casing, conv_blocks[0], conv_blocks[1], conv_blocks[2]])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.5, name='token_lstm'))(output)
    # output = TimeDistributed(Dense(len(label2Idx), activation="relu", name = 'token_dense'))(output)
    crf = CRF(len(label2Idx), name = 'crf')
    output = crf(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss=crf.loss_function, optimizer='nadam', metrics=[crf.accuracy])
    model.summary()
    return(model)

def get_model_lstm3cnn():
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
    words_input = Input(shape=(None, nb_embedding_dims), dtype='float32', name='words_input')
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False, name = 'case_embed')(casing_input)
    
    # lstm
    character_input=Input(shape=(None,nb_char_embeddings,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),32,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    char_lstm = TimeDistributed(Bidirectional(LSTM(50, name = 'char_lstm', return_sequences = True)))(embed_char_out)
    
    # 3-cnn
    kernel_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in kernel_sizes:
        conv = TimeDistributed(Conv1D(
                             kernel_size=sz,
                             filters=32,
                             padding="same",
                             activation="relu",
                             strides=1,
                             name='charcnn_' + str(sz)))(char_lstm)
        conv = TimeDistributed(MaxPooling1D(52, name = 'charcnn_maxpool'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        conv_blocks.append(conv)
    
    # model
    output = concatenate([words_input, casing, conv_blocks[0], conv_blocks[1], conv_blocks[2]])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.5, name = 'token_lstm'))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation="relu", name = 'token_dense'))(output)
    crf = CRF(len(label2Idx), name = 'crf')
    output = crf(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss=crf.loss_function, optimizer='nadam', metrics=[crf.accuracy])
    model.summary()
    return(model)



def get_model_3cnnlstm():
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
    words_input = Input(shape=(None, nb_embedding_dims), dtype='float32', name='words_input')
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False, name = 'case_embed')(casing_input)
    
    character_input=Input(shape=(None,nb_char_embeddings,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),32,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    
    # 3-cnn
    kernel_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in kernel_sizes:
        conv = TimeDistributed(Conv1D(
                             kernel_size=sz,
                             filters=32,
                             padding="same",
                             activation="relu",
                             strides=1,
                             name='charcnn_' + str(sz)))(embed_char_out)
        conv = TimeDistributed(MaxPooling1D(52, name = 'charcnn_maxpool'))(conv)
        # conv = TimeDistributed(Flatten())(conv)
        conv_blocks.append(conv)
        
    # lstm
    conv_conc = concatenate([conv_blocks[0], conv_blocks[1], conv_blocks[2]])
    char_lstm = TimeDistributed(Bidirectional(LSTM(50, name = 'char_lstm')))(conv_conc)
    
    # model
    output = concatenate([words_input, casing, char_lstm])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.5, name = 'token_lstm'))(output)
    output = TimeDistributed(Dense(len(label2Idx), name = 'token_dense'))(output)
    crf = CRF(len(label2Idx), name = 'crf')
    output = crf(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss=crf.loss_function, optimizer='nadam', metrics=[crf.accuracy])
    model.summary()
    return(model)


def get_model_3cnn_2ndlevel():
    
    L1embeddings = np.identity(len(labelOuter2Idx), dtype='float32')
    L1_input = Input(shape=(None,), dtype='int32', name='L1_input')
    L1_embed = Embedding(output_dim=L1embeddings.shape[1], input_dim=L1embeddings.shape[0], weights=[L1embeddings], trainable=False, name = 'L1_embed')(L1_input)
    
    words_input = Input(shape=(None, nb_embedding_dims), dtype='float32', name='words_input')
    
    caseEmbeddings = np.identity(len(case2Idx), dtype='float32')
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings], trainable=False, name = 'case_embed')(casing_input)
    
    character_input=Input(shape=(None,nb_char_embeddings,),name='char_input')
    embed_char_out=TimeDistributed(Embedding(len(char2Idx),32,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
    
    kernel_sizes = (3, 4, 5)
    conv_blocks = []
    for sz in kernel_sizes:
        conv = TimeDistributed(Conv1D(
                             kernel_size=sz,
                             filters=32,
                             padding="same",
                             activation="relu",
                             strides=1,
                             name='charcnn_' + str(sz)))(embed_char_out)
        conv = TimeDistributed(MaxPooling1D(52, name = 'charcnn_maxpool'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        conv_blocks.append(conv)
    
    output = concatenate([words_input, L1_embed, casing, conv_blocks[0], conv_blocks[1], conv_blocks[2]])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.5, name='token_lstm'))(output)
    output = TimeDistributed(Dense(len(label2Idx), name = 'token_dense'))(output)
    crf = CRF(len(label2Idx), name = 'crf')
    output = crf(output)
    model = Model(inputs=[words_input, L1_input, casing_input, character_input], outputs=[output])
    model.compile(loss=crf.loss_function, optimizer='nadam', metrics=[crf.accuracy])
    model.summary()
    return(model)
