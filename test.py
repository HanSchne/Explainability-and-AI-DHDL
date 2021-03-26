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
from numpy.random import seed
import tensorflow
import random
seed(8)
tensorflow.random.set_seed(13)
random.seed(17)
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd


# read poems using simplereader
poems_english = readPoems('tsv/english.tsv')
poems_german = readPoems('tsv/emotion.german.tsv')

# extract sentences with one label
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
        
data = np.array(data)

# split data into sentences and labels
sentences = data[:,0]
labels = data[:,1]

# transform labels into numerical values and one hot encodings
label_dict = {
    'Sadness': 0, 'Humor': 1, 'Suspense': 2, 'Nostalgia': 3, 'Uneasiness': 4, 'Annoyance': 5, 'Awe / Sublime': 6, 'Vitality': 7, 'Beauty / Joy' : 8
}

labels_num = []
for lab in labels:
    labels_num.append(label_dict[lab])
one_hot_labels = to_categorical(labels_num)

# analyze distribution of labels in dataset
df = pd.DataFrame({"labels": labels_num})
print(df['labels'].value_counts())

# use pretrained multilingual model to encode sentences
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
embeddings = model.encode(sentences)

# shuffle data and split into train and test set
all_data = [(embeddings[i],one_hot_labels[i]) for i in range(len(embeddings))]
random.shuffle(all_data)
embeddings = [emb for emb,_ in all_data]
labels = [lab for _, lab in all_data]

train_data = np.array(embeddings[:int(0.8*len(embeddings))])
test_data = np.array(embeddings[int(0.8*len(embeddings)):])
train_labels = np.array(labels[:int(0.8*len(embeddings))])
test_labels = np.array(labels[int(0.8*len(embeddings)):])


# Hyperparameter Tuning
"""
learning_rates = [0.001, 0.01, 0.1]
epochs = [1,2,3,4,5,6,7,8,9,10]
middle_nodes = [20,50,100,150,200]
losses = []
accuracies = []
max_loss = 100000
min_acc = 0
max_config = None
for lr in learning_rates:
    for epoch in epochs:
        for middle_node in middle_nodes:

            adam = Adam(learning_rate=lr)
            mdl = Sequential()
            mdl.add(Dense(middle_node, input_dim=512, kernel_initializer="uniform",
                activation="relu"))
            mdl.add(Dense(9, activation="softmax", kernel_initializer="uniform"))
            mdl.compile(loss="categorical_crossentropy", optimizer=adam,
                metrics=["accuracy"])

            mdl.fit(train_data, train_labels, epochs=epoch,
                verbose=1)
            print("[INFO] evaluating on testing set...")
            (loss, accuracy) = mdl.evaluate(test_data, test_labels, verbose=1)
            print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                accuracy * 100))
            losses.append(loss)
            accuracies.append(accuracy)
            if accuracy > min_acc:
                min_acc = accuracy
                max_config = (lr,epoch,middle_node)
print(max_config)
"""

# use final model
adam = Adam(learning_rate=0.001)
mdl = Sequential()
mdl.add(Dense(100, input_dim=512, kernel_initializer="uniform", activation="relu"))
mdl.add(Dense(9, activation="softmax", kernel_initializer="uniform"))
mdl.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()])

mdl.fit(train_data, train_labels, epochs=5, verbose=1)
print("[INFO] evaluating on testing set...")
(loss, accuracy, precision, recall) = mdl.evaluate(test_data, test_labels, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
print(precision)
print(recall)

# apply LIME to obtain explanations for a specific instance
def pipeline(sentence, mdl=mdl, model=model):
    n = len(sentence)
    embedded = model.encode(sentence)
    print(embedded.shape)
    return mdl.predict(embedded, batch_size=embedded.shape[0])

idx = 78
print("True Label: ", labels[idx])
emb = np.array(model.encode(sentences[idx]))
emb = emb.reshape((512,1))
emb = emb.T
print("Predicted Probs: ", mdl.predict(emb, batch_size=1))

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
