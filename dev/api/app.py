# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from functions_app import *
import pandas as pd
import en_core_web_sm
import spacy
import pickle


app = Flask(__name__)


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
                           sup=sup[0].split(','),
                           unsup=unsup.str.split(',')[0])


if __name__ == "__main__":
    path = "c:/Adrian - GDrive/Formation/Informatique - Digital/OpenClassroom/IML/P5_Catégorisez_automatiquement_des_questions/dev/api/src/"

    # Initialisation du pipeline du nettoyage de texte
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    nlp.add_pipe(CleanBeforeTaggerComponent(nlp), first=True)
    nlp.add_pipe(ContractionsComponent(nlp), after='CleanBeforeTagger')
    clean = CleanAfterParserComponent(nlp)
    bag_tags_lst = pd.read_csv(path + 'bag_tags.csv').list.to_list()
    clean.set_protect(bag_tags_lst)
    nlp.add_pipe(clean, after='parser')

    # Initialisation des modèles de prédiction
    df_topic_keywords = pd.read_csv(path + 'topic_keyword.csv')
    count_vectorizer = pickle.load(open(path + 'count_vectorizer.sav', 'rb'))
    unsupervised_model = pickle.load(open(path + 'unsupervised_model.sav',
                                          'rb'))
    tfidf_vectorizer = pickle.load(open(path + 'tfidf_vectorizer.sav', 'rb'))
    binarizer = pickle.load(open(path + 'binarizer.sav', 'rb'))
    supervised_model = pickle.load(open(path + 'supervised_model.sav', 'rb'))
    print('###############')
    print('Modèles chargés')
    app.run(debug=True)
