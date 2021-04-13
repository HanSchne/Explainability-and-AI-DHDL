# Digital Humanities and Deep Learning - Explainability and AI

This repository contains the code for our experiments of explainability techniques on a text classification model.
The model is trained to classify poem stanzas by their aesthetic emotions. The data, as well as the simplereader.py is taken from https://github.com/tnhaider/poetry-emotion.
The model uses a multilingual pretrained sentence transformer model (https://www.sbert.net/docs/pretrained_models.html).

Different explainability techniques were tested. This repository contains the code for LIME, Anchor and ProtoDash. 
LIME and ProtoDash were used in the AIX360 library (https://github.com/Trusted-AI/AIX360).

For applying the explainability techniques, the following notebooks were used as application guideline.
LIME: https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html
Anchor: https://github.com/marcotcr/anchor/blob/master/notebooks/Anchor%20for%20text.ipynb
ProtoDash: https://github.com/Trusted-AI/AIX360/blob/master/examples/protodash/Protodash%20Text%20example%20SPAM%20HAM.ipynb

