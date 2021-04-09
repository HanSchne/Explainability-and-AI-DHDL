from simplereader import readPoems
from sentence_transformers import SentenceTransformer, util
#from simpletransformers.classification import MultiLabelClassificationModel
import csv
import itertools
import numpy as np
import lime
import spacy
from anchor import anchor_text
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
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def main(TRAIN=False, TUNING=False, ANCHOR=False, LIME=True, STATISTICS=False):
    # read poems using simplereader
    poems_english = readPoems('tsv/english.tsv')
    poems_german = readPoems('tsv/emotion.german.tsv')
    poems_chinese = readPoems('tsv/chinese.tsv')
    print(len(poems_english))
    print(len(poems_german))
    print(len(poems_chinese))
    #set up label dictionary
    label_dict = {
        'Sadness': 0, 'Humor': 1, 'Suspense': 2, 'Nostalgia': 3, 'Uneasiness': 4, 'Annoyance': 5, 'Awe / Sublime': 6, 'Vitality': 7, 'Beauty / Joy' : 8
    }

    #array of stanzas
    stanzas = []

    #array of most prominent label for each stanza
    labels = []
    
    #list of languages
    lang = []

    # extract sentences with one label
    for poem in itertools.chain(poems_english, poems_german, poems_chinese):
        for stanza in poem[1:]:
            if poem in poems_english:
                lang.append(0)
            elif poem in poems_german:
                lang.append(1)
            else:
                lang.append(2)
            labelsPerStanza = []
            currentStanzaIndex = len(stanzas)
            newStanza = 1
            for line in stanza:
                if newStanza:
                    stanzas.append(line[0])
                    newStanza = 0
                else:
                    stanzas[currentStanzaIndex] += " " + line[0]
                labelsPerStanza.extend(line[1].split(" --- "))
                if len(line) > 2:
                    labelsPerStanza.extend(line[2].split(" --- "))
            counter = [0,0,0,0,0,0,0,0,0]
            for label in labelsPerStanza:
                counter[label_dict[label]] += 1
            labels.append(np.argmax(counter))
    
    # plot dataset statistics
    if STATISTICS is True:
        df = pd.DataFrame({"stanzas": stanzas, "labels": labels, "languages": lang})
        
        bar_labels = [lab.replace(" ", "") for lab in label_dict.keys()]
        ger_values = df.loc[df["languages"] == 1, "labels"].value_counts()
        en_values = df.loc[df["languages"] == 0, "labels"].value_counts()
        ch_values = df.loc[df["languages"] == 2, "labels"].value_counts()
        print(type(df.loc[df["languages"] == 1, "labels"].value_counts()))
        ger_values[3] = 0
        ger_values.sort_index(inplace=True)
        en_values.sort_index(inplace=True)
        ch_values.sort_index(inplace=True)

        width = 0.5
        
        fig, ax = plt.subplots()
        plt.grid(zorder=0, alpha=0.7)
        ax.bar(bar_labels, ger_values, width, label='German')
        ax.bar(bar_labels, en_values, width, bottom=ger_values, label='English')
        ax.bar(bar_labels, ch_values, width, bottom=en_values+ger_values, label='Chinese')

        ax.set_ylabel('Number of stanzas', fontsize=18)
        ax.legend(prop={'size': 18})
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.xticks(rotation=16)
        
        plt.show()

    
    # transform labels into one hot encodings
    one_hot_labels = to_categorical(labels)

    # analyze distribution of labels in dataset
    df = pd.DataFrame({"labels": labels})
    print(df['labels'].value_counts())

    # use pretrained multilingual model to encode sentences
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    embeddings = model.encode(stanzas)

    # shuffle data and split into train and test set
    all_data = [(embeddings[i],one_hot_labels[i], i) for i in range(len(embeddings))]
    random.shuffle(all_data)
    embeddings = [emb for emb, _, _ in all_data]
    labels = [lab for _, lab, _ in all_data]
    indices = [idx for _, _, idx in all_data]

    train_data = np.array(embeddings[:int(0.75*len(embeddings))])
    train_labels = np.array(labels[:int(0.75*len(embeddings))])
    dev_data = np.array(embeddings[int(0.75*len(embeddings)):int(0.875*len(embeddings))])
    dev_labels = np.array(labels[int(0.75*len(embeddings)):int(0.875*len(embeddings))])
    test_data = np.array(embeddings[int(0.875*len(embeddings)):])
    test_labels = np.array(labels[int(0.875*len(embeddings)):])


    # Hyperparameter Tuning
    if TUNING is True:
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
                    print("Training with following hyperparameters:", lr, epoch, middle_node)
                    adam = Adam(learning_rate=lr)
                    mdl = Sequential()
                    mdl.add(Dense(middle_node, input_dim=512, kernel_initializer="uniform",
                        activation="relu"))
                    mdl.add(Dense(9, activation="softmax", kernel_initializer="uniform"))
                    mdl.compile(loss="categorical_crossentropy", optimizer=adam,
                        metrics=["categorical_accuracy"])

                    mdl.fit(train_data, train_labels, epochs=epoch,
                        verbose=1)
                    print("evaluating on dev set...")
                    (loss, accuracy) = mdl.evaluate(dev_data, dev_labels, verbose=1)
                    print("loss: {:.4f}, accuracy: {:.4f}%".format(loss,
                        accuracy * 100))
                    losses.append(loss)
                    accuracies.append(accuracy)
                    if accuracy > min_acc:
                        min_acc = accuracy
                        max_config = (lr,epoch,middle_node)
        print(max_config)
        
    #max_config = (0.01, 7, 150)

    if TRAIN is True:
    # use final model
        adam = Adam(learning_rate=max_config[0])
        mdl = Sequential()
        mdl.add(Dense(max_config[2], input_dim=512, kernel_initializer="uniform", activation="relu"))
        mdl.add(Dense(9, activation="softmax", kernel_initializer="uniform"))
        mdl.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["categorical_accuracy"])

        mdl.fit(train_data, train_labels, epochs=max_config[1], verbose=1)
        print("evaluating on test set...")
        (loss, accuracy) = mdl.evaluate(test_data, test_labels, verbose=1)
        print("loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        #print("precision={:.4f}%".format(precision * 100))
        #print("recall={:.4f}%".format(recall * 100))
        mdl.save('emotion_classifier')
    
    mdl = keras.models.load_model('emotion_classifier')
    (loss, accuracy) = mdl.evaluate(test_data, test_labels, verbose=1)
    
    
    y_pred = mdl.predict(test_data, batch_size=test_data.shape[0])
        
    wrong_classified_idx = []
        
    for j, idx in enumerate(indices[int(0.875*len(embeddings)):]):
        if np.argmax(y_pred[j]) != np.where(test_labels[j] == 1.0)[0]:
            wrong_classified_idx.append(idx)

    print("These stanzas were wronlgy classified:")
    print(wrong_classified_idx)
        
    wrong_classified_en = [idx for idx in wrong_classified_idx if idx < 167]
    wrong_classified_ger = [idx for idx in wrong_classified_idx if (idx >= 167 and idx < 688)]
    wrong_classified_ch = [idx for idx in wrong_classified_idx if idx >= 688]
        
    print("Number of wrong classified stanzas - English: ", len(wrong_classified_en))
    print("Number of wrong classified stanzas - German: ", len(wrong_classified_ger))
    print("Number of wrong classified stanzas - Chinese: ", len(wrong_classified_ch))
    
    class_names = ['Sadness', 'Humor', 'Suspense', 'Nostalgia', 'Uneasiness', 'Annoyance', 'Awe / Sublime', 'Vitality', 'Beauty / Joy']
    #------------------------------------------------------------LIME--------------------------------------------------------------------------------------------
    # apply LIME to obtain explanations for a specific instance
    def pipeline(stanza, mdl=mdl, model=model):
        embedded = model.encode(stanza)
        return mdl.predict(embedded, batch_size=embedded.shape[0])

    if LIME is True:
        """
        idx = 78
        print("True Label: ", labels[idx])
        emb = np.array(model.encode(stanzas[idx]))
        emb = emb.reshape((512,1))
        emb = emb.T
        print("Predicted Probabilities: ", mdl.predict(emb, batch_size=1))

        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(stanzas[idx], pipeline, num_features=6, top_labels=2)
        top_labs = exp.available_labels()

        print("Explanation for class {}".format(top_labs[0]))
        print ('\n'.join(map(str, exp.as_list(label=top_labs[0]))))

        print("Explanation for class {}".format(top_labs[1]))
        print ('\n'.join(map(str, exp.as_list(label=top_labs[1]))))
        
        fig = exp.as_pyplot_figure(top_labs[0])
        plt.show()
        fig_2 = exp.as_pyplot_figure(top_labs[1])
        """
        
        for idx in itertools.chain(wrong_classified_ger[:5], wrong_classified_en[:5]):
            print("True Label: ", one_hot_labels[idx])
            emb = np.array(model.encode(stanzas[idx]))
            emb = emb.reshape((512,1))
            emb = emb.T
            print("Predicted Probabilities: ", mdl.predict(emb, batch_size=1))

            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(stanzas[idx], pipeline, num_features=6, top_labels=2)
            top_labs = exp.available_labels()

            print("Explanation for class {}".format(top_labs[0]))
            print ('\n'.join(map(str, exp.as_list(label=top_labs[0]))))

            print("Explanation for class {}".format(top_labs[1]))
            print ('\n'.join(map(str, exp.as_list(label=top_labs[1]))))
            
            fig = exp.as_pyplot_figure(top_labs[0])
            plt.show()
            fig_2 = exp.as_pyplot_figure(top_labs[1])
            plt.show()
        

    #----------------------------------------------------------ANCHOR---------------------------------------------------------------------------------------------
    def predict_label(stanza):
        embedded = model.encode(stanza)
        probs = mdl.predict(embedded, batch_size=embedded.shape[0])
        return [np.argmax(probs)]
    
    def predict_second_label(stanza, predicted_label):
        embedded = model.enoce(stanza)
        probs = mdl.predict(embedded, batch_size=embedded.shape[0])
        probs = np.delete(probs, predited_label)
        return [np.argmax(probs)]
    
    if ANCHOR is True:
        nlp = spacy.load('en_core_web_lg')
        explainer = anchor_text.AnchorText(nlp, class_names, use_unk_distribution=True)
        
        idx = 56
        text = stanzas[idx]
        print(predict_label([text]))
        pred = explainer.class_names[predict_label([text])[0]]
        alternative =  explainer.class_names[predict_second_label([text])[0]]
        print('Prediction: %s' % pred)
        exp = explainer.explain_instance(text, predict_label, threshold=0.95)
        
        
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print()
        print('Examples where anchor applies and model predicts %s:' % pred)
        print()
        print('\n'.join([x[0] for x in exp.examples(only_same_prediction=True)]))
        print()
        print('Examples where anchor applies and model predicts %s:' % alternative)
        print()
        print('\n'.join([x[0] for x in exp.examples(partial_index=0, only_different_prediction=True)]))


if __name__ == "__main__":
    main(LIME=True, TRAIN=False,TUNING=False)
