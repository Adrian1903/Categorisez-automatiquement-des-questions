# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from functions_app import *
import pandas as pd
import spacy
import pickle
import os

app = Flask(__name__)

# Initailisation non fonctionnelle si présente dans main (en bas)
# Initialisation du pipeline du nettoyage de texte
nlp = spacy.load('en_core_web_sm', disable=['ner'])
nlp.add_pipe(CleanBeforeTaggerComponent(nlp), first=True)
nlp.add_pipe(ContractionsComponent(nlp), after='CleanBeforeTagger')
clean = CleanAfterParserComponent(nlp)
bag_tags_lst = pd.read_csv('src/bag_tags.csv').list.to_list()
clean.set_protect(bag_tags_lst)
nlp.add_pipe(clean, after='parser')

# Initialisation des modèles de prédiction
df_topic_keywords = pd.read_csv('src/topic_keyword.csv')
count_vectorizer = pickle.load(open('src/count_vectorizer.sav', 'rb'))
unsupervised_model = pickle.load(open('src/unsupervised_model.sav', 'rb'))
tfidf_vectorizer = pickle.load(open('src/tfidf_vectorizer.sav', 'rb'))
binarizer = pickle.load(open('src/binarizer.sav', 'rb'))
supervised_model = pickle.load(open('src/supervised_model.sav', 'rb'))


@app.route('/')
def home():
    return render_template('form.html')


@app.route("/tags", methods=["POST"])
def predict():
    text = request.form['user_question']
    cleaned_text = str(nlp(text))
    sup = supervised_tags(cleaned_text,
                          tfidf_vectorizer,
                          binarizer,
                          supervised_model)
    unsup = unsupervised_tags(cleaned_text,
                              count_vectorizer,
                              df_topic_keywords,
                              unsupervised_model)

    return render_template('form.html',
                           question=text,
                           sup=sup[0],
                           unsup=unsup[0])


if __name__ == "__main__":
    app.run(debug=True)
