from simplereader import readPoems
from sentence_transformers import SentenceTransformer, util
#from simpletransformers.classification import MultiLabelClassificationModel
import csv
import numpy as np
import lime
import sklearn
import sklearn.ensemble
import sklearn.metrics
from lime.lime_text import LimeTextExplainer
from tensorflow import keras
from tensorflow.keras import layers
import random
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

poems_english = readPoems('tsv/english.tsv')

poems_german = readPoems('tsv/emotion.german.tsv')

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

"""
#Our sentences we like to encode
sentences = ['Hallo Welt', 'Hello World']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")


print(np.linalg.norm(embeddings[0] - embeddings[1]))
print(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
print(len(embeddings[0]))
"""

data = []
for poem in poems_english:
    strophen = poem[1]
    for strophe in strophen:
        tmp = [strophe[0]]
        labels = strophe[1].split(" --- ")
        label = labels[0]
        tmp.append(label)
        data.append(tmp)
for poem in poems_german:
    strophen = poem[1]
    for strophe in strophen:
        tmp = [strophe[0]]
        labels = strophe[1].split(" --- ")
        label = labels[0]
        tmp.append(label)
        data.append(tmp)
        
#print(data)
data = np.array(data)
sentences = data[:,0]
labels = data[:,1]

#print(sentences)
#print(set(labels))

label_dict = {
    'Sadness': 0, 'Humor': 1, 'Suspense': 2, 'Nostalgia': 3, 'Uneasiness': 4, 'Annoyance': 5, 'Awe / Sublime': 6, 'Vitality': 7, 'Beauty / Joy' : 8
}

labels_num = []
for lab in labels:
    labels_num.append(label_dict[lab])
    

one_hot_labels = to_categorical(labels_num)

embeddings = model.encode(sentences)
"""
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(embeddings, labels_num)

#pred = rf.predict(test_sents)
#sklearn.metrics.f1_score(test_labels, pred, average='binary')

def pipeline(sentence):
    embedded = model.encode(sentence)
    return rf.predict_proba(embedded)
    
    

idx = 78
print("True Label: ", labels[idx])
emb = model.encode(sentences[idx])
print("Predicted Probs: ", rf.predict_proba([emb]))

print("Predicted Label: ", rf.predict([emb]))

class_names = ['Sadness', 'Humor', 'Suspense', 'Nostalgia', 'Uneasiness', 'Annoyance', 'Awe / Sublime', 'Vitality', 'Beauty / Joy']
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(sentences[idx], pipeline, num_features=6, top_labels=2)
top_labs = exp.available_labels()

print("Explanation for class {}".format(top_labs[0]))
#print(exp.as_list(label=top_labs[0]))
print ('\n'.join(map(str, exp.as_list(label=top_labs[0]))))

print("Explanation for class {}".format(top_labs[1]))
#print(exp.as_list(label=top_labs[1]))
print ('\n'.join(map(str, exp.as_list(label=top_labs[1]))))
"""
random.seed(8)
all_data = [(embeddings[i],one_hot_labels[i]) for i in range(len(embeddings))]
print(all_data[0])
random.shuffle(all_data)
print(all_data[0])
embeddings = [emb for emb,_ in all_data]
labels = [lab for _, lab in all_data]
train_data = np.array(embeddings[:int(0.8*len(embeddings))])
test_data = np.array(embeddings[int(0.8*len(embeddings)):])
train_labels = np.array(labels[:int(0.8*len(embeddings))])
test_labels = np.array(labels[int(0.8*len(embeddings)):])
print(train_data.shape)
print(train_labels.shape)
print(train_data[:10])
print(train_labels[:10])


"""
x = layers.Input(shape=(512), name="InputLayer")
h1 = layers.Dense(units=150, activation="relu")(x)
    #a1 = layers.Activation(activation="relu")(h1)
h2 = layers.Dense(units=150, activation="relu")(h1)
    #a1 = layers.Activation(activation="relu")(h1)
y = layers.Dense(units=9, activation="softmax")(h2)
mdl = keras.Model(inputs=x, outputs=y, name="logreg2_model")
mdl.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
"""

mdl = Sequential()
mdl.add(Dense(100, input_dim=512, kernel_initializer="uniform",
    activation="relu"))
mdl.add(Dense(9, activation="softmax", kernel_initializer="uniform"))
mdl.compile(loss="categorical_crossentropy", optimizer="adam",
    metrics=["accuracy"])

mdl.fit(train_data, train_labels, epochs=20,
    verbose=1)
print("[INFO] evaluating on testing set...")
(loss, accuracy) = mdl.evaluate(test_data, test_labels,
    batch_size=50, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    accuracy * 100))


