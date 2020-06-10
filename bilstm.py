import csv
import numpy as np
from matplotlib import pyplot as plt

# store all.trde line by line & tokenized
dataset = []
with open('all.trde') as tsv:
	for line in csv.reader(tsv, dialect="excel-tab"):
		if line != []:
			if len(line) == 2:
				dataset.append(['"', '"', 'OTHER'])
			else:	
				dataset.append(line)

# extract the set of characters in data and do an integer-mapping for numeric representation
raw_text = open('all.trde', 'r').read()
chars = sorted(list(set(raw_text)))
chars.insert(0,'PAD')
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i,c) for i, c in enumerate(chars))

# integer-mapping for tags
tags = ['DE', 'TR', 'DE TR', 'LANG3', 'AMBIG', 'OTHER', 'NE.DE', 'NE.TR', 'NE.LANG3', 'NE.OTHER', 'NE.AMBIG']
tags_to_int = dict((t, i) for i, t in enumerate(tags))
int_to_tag = dict((i, t) for i, t in enumerate(tags))

# map a given word to its numeric representation, e.g. "ein" -> [52, 26, 101] 
def encode_token(token):
	encoded_token = [char_to_int[char] for char in token]
	return encoded_token

# map all dataset to numeric represetations
# return is sth like [[[encoded word], encoded tag], ..., [[encoded word], encoded tag]]
# e.g. [[[52, 26, 101], 0], [[150, 23], 0], ...]
def encode_dataset(dataset):
	encoded_dataset = []
	for line in dataset:
		token = line[0]
		label = line[2]
		encoded_token = encode_token(token)
		encoded_label = tags_to_int[label]
		encoded_dataset.append([encoded_token, encoded_label])
	return encoded_dataset

encoded_dataset = encode_dataset(dataset)

# do post padding for encoded tokens
from keras.preprocessing.sequence import pad_sequences
all_tokens = [i[0] for i in dataset]
max_len = len(max(all_tokens, key=len))
all_encoded_tokens = [i[0] for i in encoded_dataset]
all_encoded_labels = [i[1] for i in encoded_dataset]
padded_encoded_tokens = pad_sequences(all_encoded_tokens, maxlen=max_len, padding='post', value=char_to_int['PAD'])

# convert the encoded tags (of the dataset) into one-hot (vector) representation: 
# for a tag t whose integer representation is i, the index i is 1, and the rest is 0
# DE == 0 ==> [1 0 0 ... 0], TR == 1 ==> [0 1 0 0 ... 0]
def one_hot_encode(encoded_tags, dict_size):
	cat_sequences = []
	for encoding in encoded_tags:
		#print(encoding)
		cats = np.zeros(dict_size)
		cats[encoding] = 1.0
		cat_sequences.append(cats)
	return np.array(cat_sequences)

# split the data as 4:1 for train & test
from sklearn.model_selection import train_test_split
train_tokens, test_tokens, train_tags, test_tags = train_test_split(padded_encoded_tokens, all_encoded_labels, test_size=0.2)

print(padded_encoded_tokens[:20])
print(all_encoded_labels[:20])
print(len(train_tokens))
print(one_hot_encode(train_tags, len(tags_to_int)))
print(len(one_hot_encode(train_tags, len(tags_to_int))))

from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

EMBEDDING_DIM = 100
model = Sequential()
model.add(Embedding(len(char_to_int), EMBEDDING_DIM, input_length=train_tokens.shape[1])) #model.add(SpatialDropout1D(0.2))
#model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(len(tags_to_int), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
model.summary()

# change the batch size to a large number like 200 for a faster run, but it decreases the accuracy
model.fit(train_tokens, one_hot_encode(train_tags, len(tags_to_int)), batch_size=100, epochs=3)
scores = model.evaluate(test_tokens, one_hot_encode(test_tags, len(tags_to_int)))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(scores[0],scores[1]))


# test the model on a hand-written file called 'test_sample.txt'
print("SAMPLE:")
sample = []
def create_sample():
	with open('test_sample.txt') as tsv:
		for line in csv.reader(tsv, dialect="excel-tab"):
			if line != []:
				sample.append(line)
	# print(len(sample))
	encoded_sample = encode_dataset(sample)
	all_encoded_tokens = [i[0] for i in encoded_sample]
	all_encoded_labels = [i[1] for i in encoded_sample]
	padded_encoded_tokens = pad_sequences(all_encoded_tokens, maxlen=max_len, padding='post', value=char_to_int['PAD'])
	return [padded_encoded_tokens, all_encoded_labels]

sample_padded = create_sample()
sample_tokens = sample_padded[0]
sample_tags = sample_padded[1]
#print(sample_tags)

sample_predictions = model.predict(sample_tokens)
#print(sample_predictions, sample_predictions.shape)
predictions = np.argmax(sample_predictions, axis = 1)
for i in range(len(predictions)):
	print("token: ", sample[i][0], " truth: ", int_to_tag[sample_tags[i]], " hyps:", int_to_tag[predictions[i]])

# calculate accuracy
count = 0.0
correct = 0.0
for i in range(len(predictions)):
  count+=1
  if sample_tags[i]==predictions[i]:
    correct+=1
print(correct/count)
print((correct/count)*100)
