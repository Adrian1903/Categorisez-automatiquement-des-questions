import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from bs4 import BeautifulSoup
import unidecode
import en_core_web_sm
import re
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.multiclass import OneVsRestClassifier

nlp = spacy.load('en_core_web_sm', disable=['ner'])


def clean_before_tagger(text):
    """remove preformat text and image bloc"""
    # J'enlève les caractères accentuées
    text = unidecode.unidecode(text)

    # Je supprime les blocs de code et image
    soup = BeautifulSoup(text, "html.parser")
    for p in soup.select('pre'):
        p.extract()
    for i in soup.select('a'):
        i.extract()

    # Je renvoie le texte sans balise HTML
    # sans retour ligne, et avec le texte en minuscule
    return soup.get_text().replace('\n', '').lower()


def expand_contractions(text):
    # https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions

    flags = re.IGNORECASE | re.MULTILINE

    text = re.sub(r'`', "'", text, flags=flags)

    # starts / ends with '
    text = re.sub(
        r"(\s|^)'(aight|cause)(\s|$)", r'\g<1>\g<2>\g<3>',
        text, flags=flags
    )

    text = re.sub(
        r"(\s|^)'t(was|is)(\s|$)", r'\g<1>it \g<2>\g<3>',
        text, flags=flags
    )

    text = re.sub(
        r"(\s|^)ol'(\s|$)", r'\g<1>old\g<2>',
        text, flags=flags
    )

    # expand words without '
    text = re.sub(r"\b(aight)\b", 'alright', text, flags=flags)
    text = re.sub(r'\bcause\b', 'because', text, flags=flags)
    text = re.sub(r'\b(finna|gonna)\b', 'going to', text, flags=flags)
    text = re.sub(r'\bgimme\b', 'give me', text, flags=flags)
    text = re.sub(r"\bgive'n\b", 'given', text, flags=flags)
    text = re.sub(r"\bhowdy\b", 'how do you do', text, flags=flags)
    text = re.sub(r"\bgotta\b", 'got to', text, flags=flags)
    text = re.sub(r"\binnit\b", 'is it not', text, flags=flags)
    text = re.sub(r"\b(can)(not)\b", r'\g<1> \g<2>', text, flags=flags)
    text = re.sub(r"\bwanna\b", 'want to', text, flags=flags)
    text = re.sub(r"\bmethinks\b", 'me thinks', text, flags=flags)

    # one offs,
    text = re.sub(r"\bo'er\b", r'over', text, flags=flags)
    text = re.sub(r"\bne'er\b", r'never', text, flags=flags)
    text = re.sub(r"\bo'?clock\b", 'of the clock', text, flags=flags)
    text = re.sub(r"\bma'am\b", 'madam', text, flags=flags)
    text = re.sub(r"\bgiv'n\b", 'given', text, flags=flags)
    text = re.sub(r"\be'er\b", 'ever', text, flags=flags)
    text = re.sub(r"\bd'ye\b", 'do you', text, flags=flags)
    text = re.sub(r"\be'er\b", 'ever', text, flags=flags)
    text = re.sub(r"\bd'ye\b", 'do you', text, flags=flags)
    text = re.sub(r"\bg'?day\b", 'good day', text, flags=flags)
    text = re.sub(r"\b(ain|amn)'?t\b", 'am not', text, flags=flags)
    text = re.sub(r"\b(are|can)'?t\b", r'\g<1> not', text, flags=flags)
    text = re.sub(r"\b(let)'?s\b", r'\g<1> us', text, flags=flags)

    # major expansions involving smaller,
    text = re.sub(r"\by'all'dn't've'd\b",
                  'you all would not have had',
                  text, flags=flags)
    text = re.sub(r"\by'all're\b", 'you all are', text, flags=flags)
    text = re.sub(r"\by'all'd've\b", 'you all would have', text, flags=flags)
    text = re.sub(r"(\s)y'all(\s)", r'\g<1>you all\g<2>', text, flags=flags)

    # minor,
    text = re.sub(r"\b(won)'?t\b", 'will not', text, flags=flags)
    text = re.sub(r"\bhe'd\b", 'he had', text, flags=flags)

    # major,
    text = re.sub(r"\b(I|we|who)'?d'?ve\b", r'\g<1> would have',
                  text, flags=flags)
    text = re.sub(r"\b(could|would|must|should|would)n'?t'?ve\b",
                  r'\g<1> not have', text, flags=flags)
    text = re.sub(r"\b(he)'?dn'?t'?ve'?d\b", r'\g<1> would not have had',
                  text, flags=flags)
    text = re.sub(r"\b(daren|daresn|dasn)'?t", 'dare not', text, flags=flags)
    text = re.sub(r"\b(he|how|i|it|she|that|there|these|they|we|what|where|\
                  which|who|you)'?ll\b", r'\g<1> will', text, flags=flags)
    text = re.sub(r"\b(everybody|everyone|he|how|it|she|somebody|someone|\
                  something|that|there|this|what|when|where|which|who|why)\
                  '?s\b", r'\g<1> is', text, flags=flags)
    text = re.sub(r"\b(I)'?m'a\b", r'\g<1> am about to', text, flags=flags)
    text = re.sub(r"\b(I)'?m'o\b", r'\g<1> am going to', text, flags=flags)
    text = re.sub(r"\b(I)'?m\b", r'\g<1> am', text, flags=flags)
    text = re.sub(r"\bshan't\b", 'shall not', text, flags=flags)
    text = re.sub(r"\b(are|could|did|does|do|go|had|has|have|is|may|might|\
                  must|need|ought|shall|should|was|were|would)n'?t\b",
                  r'\g<1> not', text, flags=flags)
    text = re.sub(r"\b(could|had|he|i|may|might|must|should|these|they|\
                  those|to|we|what|where|which|who|would|you)'?ve\b",
                  r'\g<1> have', text, flags=flags)
    text = re.sub(r"\b(how|so|that|there|these|they|those|we|what|where|\
                  which|who|why|you)'?re\b", r'\g<1> are',
                  text, flags=flags)
    text = re.sub(r"\b(I|it|she|that|there|they|we|which|you)'?d\b",
                  r'\g<1> had', text, flags=flags)
    text = re.sub(r"\b(how|what|where|who|why)'?d\b", r'\g<1> did',
                  text, flags=flags)

    return text


def clean_after_parser(text, protect=[]):
    doc = nlp(text)
    txt = [token.lemma_ for token in doc
           if ((token.dep_ == 'ROOT' or
                token.pos_ == 'NOUN' or
                token.pos_ == 'ADJ' or
                token.pos_ == 'ADV') and
               token.text not in STOP_WORDS) or
           token.text in protect]

    # Attention aux mots qui peuvent être transformé en tags !!!
    # stemmer = PorterStemmer()
    # clean = [stemmer.stem(t) for t in txt]

    return ' '.join(txt)


class ContractionsComponent(object):
    name = "Contractions"

    nlp: Language

    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(expand_contractions(text))


class CleanBeforeTaggerComponent(object):
    name = "CleanBeforeTagger"

    nlp: Language

    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(clean_before_tagger(text))


class CleanAfterParserComponent(object):
    name = "CleanAfterParser"

    nlp: Language

    def __init__(self, nlp: Language):
        self.nlp = nlp

    def set_protect(self, bag):
        self.bag_tags_lst = bag

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(
            clean_after_parser(text, protect=self.bag_tags_lst))


def get_unsupervised_tag(doc_topic,
                         topic_keywords,
                         corpus,
                         n_tags=5,
                         threshold=0.09):
    """Récupère les tags prédits issue de l'analyse non supervisé

    Args:
        doc_topic (Dataframe): Document avec sujets dominants
        topic_keywords (Dataframe): Sujets avec mots-clés dominants
        corpus (Dataframe): Corpus de texte
        threshold (float, optional): Seuil de réglage afin que les sujets trop
        faible ne soit pas pris en compte.
        Peut générer des tags supplémentaires.
        Defaults to 0.09.

    Returns:
        [type]: [description]
    """
    eval = {}
    for doc in range(len(corpus)):
        # DOC_TOPIC
        # Je pondère le poids de chaque topic dans chaque document
        filter = doc_topic.loc[doc] > threshold
        value = doc_topic.loc[doc][filter].reset_index()
        value.columns = ['topic', 'value']
        lst_topic = value.topic.to_list()

        # Je calcule le nombre de mots à prendre dans chaque topic
        nword = value.value / sum(value.value.to_list()) * n_tags
        value['n_words'] = round(nword, 0).astype(int)

        diff_words = n_tags - value.n_words.sum()
        # Si j'ai moins de 5 tags, j'en rajoute là où la valeur est plus forte
        if diff_words > 0:
            index = value.sort_values(by='value').tail(1).reset_index()
            index = index.at[0, 'index']
            cond = value.index == index
            value.n_words[cond] = value.n_words[cond] + diff_words

        # Si j'ai plus de 5 tags, j'en supprimme la où la valeur
        # est plus faible
        elif diff_words < 0:
            index = value.sort_values(by='value').head(abs(diff_words))
            index = index.reset_index().loc[0:abs(diff_words) - 1, 'index']
            index = index.to_list()

            for i in index:
                cond = value.index == i
                value.n_words[cond] = value.n_words[cond] - 1
        # Je visualise le nombre de mots à récupérer dans chaque topic
        # display(value)

        # Je visualise les mots associés à chaque topic
        # display(topic_keywords.loc[lst_topic])

        # TOPIC_KEYWORD
        # Je sélectionne les tags en fonction de la pondération.
        # Je fais attention à ne pas ajouter 2 fois le même tag
        # car 1 mot peut être présent dans plusieurs topics
        lst = []
        for t in value.topic:
            # Je définis le nombre de mots que je dois aller chercher
            lst_word = np.arange(0, value[value.topic == t].n_words.item())
            for w in lst_word:
                tag = topic_keywords.at[t, 'Word ' + str(w)]
                # Si le tag a déjà été inséré dans la liste,
                # je prends le suivant
                if tag in lst:
                    tag = topic_keywords.at[t, 'Word ' + str(w + 1)]

                # J'ajoute le tag à la liste
                lst.append(tag)

        eval[doc] = lst

    pred_tag = pd.DataFrame.from_dict(eval, orient='index')
    pred_tag['unsupervised_tag'] = (pred_tag[0] + ',' + pred_tag[1] + ','
                                    + pred_tag[2] + ',' + pred_tag[3] + ','
                                    + pred_tag[4])

    pred_tag = pred_tag.drop(columns=[0, 1, 2, 3, 4])
    return pred_tag


def transform_tuple(tup):
    i = 0
    for sub in tup:
        tup[i] = ','.join(sub)
        i += 1
    return tup


def supervised_tags(cleaned_text, vectorizer, binarizer, supervised_model):
    tfidf_cleaned_text = vectorizer.transform([cleaned_text])
    pred = supervised_model.predict_proba(tfidf_cleaned_text)
    pred = pd.DataFrame(pred).applymap(lambda x: 1 if x > 0.11 else 0)
    pred = pred.to_numpy()
    return transform_tuple(binarizer.inverse_transform(pred))


def unsupervised_tags(cleaned_text, vectorizer,
                      df_topic_keywords, unsupervised_model):
    vect_cleaned_text = vectorizer.transform([cleaned_text])
    out = unsupervised_model.transform(vect_cleaned_text)
    doc_topic = round(pd.DataFrame(out), 2)
    df_tags = get_unsupervised_tag(doc_topic,
                                   df_topic_keywords,
                                   [cleaned_text])
    return df_tags.unsupervised_tag.to_list()
