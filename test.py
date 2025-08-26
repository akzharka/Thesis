import numpy as np
import tensorflow as tf
import os
import argparse
from encoder import Encoder
from bahdanauAttention import BahdanauAttention
from decoder import Decoder
from feature_extraction import FeatureExtraction


parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str)
parser.add_argument('--src', type=str, help="Folder where videos are located")
parser.add_argument('--backbone', type=str, default='vgg16')
parser.add_argument('--model', type=str, help="checkpoint folder")
args = parser.parse_args()


BATCH_SIZE = 8
units = 40  # number of units in GRU for encoder and decoder structure
max_len = 40  # maximum number of words in the generated sentence
embedding_dim = 256  # dimension of the embedding layer

video_name = args.video
backbone = args.backbone
video_path = args.src + '/' + video_name
frame_path = args.src + '/frames'
target_token_index = np.load('new_dictionary.npy', allow_pickle=True).item()
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())


def load_desc_dict(doc):
    caption = {}
    for line in doc.split('\n'):
    # split line by white space
        if len(line) < 2:
            continue
    # take the first token as the video id, the rest as the description
        l = line.split(' ')
        video_id, video_desc = l[0], ' '.join(l[1:])
    # store descriptions in a dictionary, where key is a video_id, value is a list of descriptions for that video
        if video_id not in caption.keys():
            caption[video_id] = [video_desc]
        else:
            caption[video_id].append(video_desc)
    return caption


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


#Video descriptions
filename = 'captions28365.txt'
file = open(filename, 'r')
text = file.read()
file.close()
video_dictionary = load_desc_dict(text)

#Vocabulary
all_words = set()
for descriptions in video_dictionary.values():
    for description in descriptions:
        [all_words.update(description.split())]
vocab_size = len(all_words) + 1


encoder = Encoder(units, BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = args.model
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Extract features
feature_extraction = FeatureExtraction(backbone, 40, video_path, frame_path)
feature_extraction.extract_frames()
frames = os.listdir(frame_path)
feature_sequence = []
for frame in frames:
    features = feature_extraction.extract_features(frame_path + "/" + frame)
    feature_sequence.append(features)


actual_desc = video_dictionary[video_name.split(".")[0]]
model_input = np.expand_dims(feature_sequence, 0)
print('Predicted description:', evaluate(model_input, encoder, decoder, max_len))
print('Actual description:', actual_desc)
