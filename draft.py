import numpy as np
import tensorflow as tf
import os
import argparse
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input, GRU, Concatenate, Activation, concatenate
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from encoder import Encoder
from bahdanauAttention import BahdanauAttention
from decoder import Decoder
from feature_extraction import FeatureExtraction


# parser = argparse.ArgumentParser()
# p = parser.add_argument('')

BATCH_SIZE = 8
units = 40
max_len = 40
embedding_dim = 256
train_path = '/home/akzharkyn/Documents/Thesis/train/'
val_path = '/home/akzharkyn/Documents/Thesis/val/'
train_frame ='/home/akzharkyn/Documents/Thesis/train/frames'
val_frame ='/home/akzharkyn/Documents/Thesis/val/frames'
target_token_index = np.load('new_dictionary.npy', allow_pickle=True).item()
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())


def load_desc_dict(doc, dataset):
    caption = {}
    captions = []
    for line in doc.split('\n'):
        # split line by white space
        if len(line) < 2:
            continue
        # take the first token as the video id, the rest as the description
        text_line = line.split(' ')
        video_id, video_desc = text_line[0], ' '.join(text_line[1:])
        captions.append((video_id, video_desc))
        # store descriptions in a dictionary, where key is a video_id, value is a list of descriptions for that video
        if video_id in dataset:
            if video_id not in caption.keys():
                caption[video_id] = [video_desc]
            else:
                caption[video_id].append(video_desc)
    return caption, captions


def evaluate(inputs, enc, dec, max_length):
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = enc(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token_index['startseq']], 0)
    result = ''
    for t in range(max_length):
        predictions, dec_hidden, attention_weights = dec(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += reverse_target_char_index[predicted_id] + ' '

        if reverse_target_char_index[predicted_id] == 'endseq':
            return result[:-1]

        dec_input = tf.expand_dims([predicted_id], 0)

    return result[:-1]


# Video descriptions for train and validation sets
filename = 'captions28365.txt'
file = open(filename, 'r')
text = file.read()
file.close()
train_videos = os.listdir(train_path)
train_ids = list(map(lambda x: x.split(".")[0], train_videos))
train_dictionary, train_list = load_desc_dict(text, train_ids)
print(train_list[:5])
