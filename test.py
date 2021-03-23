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

embeddings = model.encode(sentences)

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
