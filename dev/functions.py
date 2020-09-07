import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import six
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import datetime as dt
from IPython.display import Image
from bs4 import BeautifulSoup
import unidecode
import en_core_web_sm
import re
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem.porter import *

# For multiclass classification
from sklearn.multiclass import OneVsRestClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

nlp = spacy.load('en_core_web_sm', disable=['ner'])


def display_circles(pcs, n_comp, pca,
                    axis_ranks, labels=None,
                    label_rotation=0, lims=None):
    # On affiche les 3 premiers plans factoriels
    # donc les 6 premières composantes
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7, 7))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(
                    pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche
            # pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]),
                           np.zeros(pcs.shape[1]),
                           pcs[d1, :],
                           pcs[d2, :],
                           angles='xy',
                           scale_units='xy',
                           scale=1,
                           color="grey")
            # voir la doc :
            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(
                    lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x,
                                 y,
                                 labels[i],
                                 fontsize='14',
                                 ha='center',
                                 va='center',
                                 rotation=label_rotation,
                                 color="blue",
                                 alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title("Cercle des corrélations (F{} et F{})"
                      .format(d1+1, d2+1))

            plt.savefig("img_circle_F{}_F{}.png".format(d1+1, d2+1),
                        dpi=500, quality=95, transparent=True)
            plt.close()
            display(Image("img_circle_F{}_F{}.png".format(d1+1, d2+1)))
            # plt.show(block=False)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks,
                             labels=None, alpha=1,
                             illustrative_var=None,
                             png_filename='projection'):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            fig = plt.figure(figsize=(7, 7))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1],
                                X_projected[selected, d2],
                                alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i],
                             fontsize='14', ha='center', va='center')

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title(
                "Projection des individus (sur F{} et F{})".format(d1+1, d2+1))

            plt.savefig("{}_F{}_F{}.png".format(png_filename, d1+1, d2+1),
                        transparent=True)
            plt.close()
            display(Image("{}_F{}_F{}.png".format(png_filename, d1+1, d2+1)))
            # plt.show(block=False)


def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


def plot_dendrogram(Z, names):
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels=names,
        orientation="left",
    )
    plt.show()


def export_png_table(data, col_width=2.2, row_height=0.625, font_size=10,
                     header_color='#7451eb', row_colors=['#f1f1f2', 'w'],
                     edge_color='w', bbox=[0, 0, 1, 1], header_columns=1,
                     ax=None, filename='table.png', **kwargs):
    ax = None
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])
                ) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox,
                         colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig.savefig(filename, transparent=True)
    return ax


def evaluate_estimators(X_train, X_test, y_train, y_test,
                        estimators, cv=5,
                        scoring='neg_root_mean_squared_error',
                        target_name='target'):
    """Evalue les modèles en estimant les meilleurs hyperparamètres.
    Crée un PNG des résultats.

    Args:
        X_train (object): Données d'entrainements
        X_test (object): Données de tests
        y_train (object): Données d'entrainements
        y_test (object): Données de tests
        estimators (dict): Contient les modèles et les hyperparamètres à tester
        cv (int, optional): Nombre de cross-validation. Defaults to 5.
        scoring (str, optional): Métrique d'évaluation des modèles.
        Defaults to 'neg_root_mean_squared_error'.
        target_name (str, optional): Nom de la cible. Defaults to 'target'.

    Returns:
        None
    """

    results = pd.DataFrame()
    for estim_name, estim, estim_params in estimators:
        print(f"{estim_name} en cours d'exécution...")
        model = GridSearchCV(estim, param_grid=estim_params,
                             cv=cv, scoring=scoring, n_jobs=4)
        model.fit(X_train, y_train)

        # Je stocke les résultats du GridSearchCV dans un dataframe
        model_results_df = pd.DataFrame(model.cv_results_)

        # Je sélectionne la meilleure observation
        condition = model_results_df["rank_test_score"] == 1
        model_results_df = model_results_df[condition]

        # J'ajoute le nom du modéle et les résultats sur les données de test
        model_results_df[target_name] = estim_name
        model_results_df['Test : R2'] = r2_score(y_test, model.predict(X_test))
        model_results_df['Test : RMSE'] = model.score(X_test, y_test)

        # Les hyperparamètres des estimateurs étant changeant,
        # je crée un nouveau dataframe à partir de la colonne params
        # des résultats. Je jointe les 2 dataframes à partir des index.
        # Cela me permet des flexible pour mon dataframe.
        model_results_df = pd.merge(model_results_df[[target_name,
                                                      'Test : RMSE',
                                                      'Test : R2',
                                                      'mean_test_score',
                                                      'std_test_score']],
                                    pd.DataFrame(model.cv_results_['params']),
                                    left_index=True, right_index=True)

        # Je stocke les résultats dans un nouveau dataframe.
        results = results.append(model_results_df)

    export_png_table(round(results, 4),
                     filename='img_results_' + target_name + '.png')
    return None


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


def get_month(x): return dt.datetime(x.year, x.month, 1)


def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])


def sortedgroupedbar(ax, x, y, groupby, data=None,
                     legend_anchor=None, width=0.8, **kwargs):
    order = np.zeros(len(data))
    df = data.copy()
    for xi in np.unique(df[x].values):
        group = data[df[x] == xi]
        a = group[y].values
        b = sorted(np.arange(len(a)), key=lambda x: a[x], reverse=True)
        c = sorted(np.arange(len(a)), key=lambda x: b[x])
        order[data[x] == xi] = c
    df["order"] = order
    u, df["ind"] = np.unique(df[x].values, return_inverse=True)
    step = width / len(np.unique(df[groupby].values))
    for xi, grp in df.groupby(groupby):
        ax.bar(grp["ind"] - width/2. + grp["order"]*step + step/2.,
               grp[y], width=step, label=xi, **kwargs)
    ax.legend(title=groupby, bbox_to_anchor=legend_anchor)
    ax.set_xticks(np.arange(len(u)))
    ax.set_xticklabels(u)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def get_next_event(x): return x['source'].shift(-1)


def affect_cluster_name(df, cluster, dict):
    """Affecte le nom des clusters à un dataframe donné via un dictionnaire
    établi préalablement. Supprime la variable intermédiaire cluster.

    Args:
        df (dataframe): Dataframe à transformer
        cluster (array): Liste des clusters ligne à ligne avec le df
        dict (dict): Nom des clusters

    Returns:
        dataframe: Retourne le dataframe avec les clusters
        correspondant aux observations
    """
    df['cluster'] = cluster
    df['cluster_name'] = np.nan

    for i, n in dict.items():
        df['cluster_name'][df.cluster == n] = i

    df.drop(columns='cluster', inplace=True)

    return df


# Function to preprocess text
def preprocess(text, model=nlp):
    """[summary]

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Create Doc object
    doc = model(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
                if lemma.isalpha() and lemma not in stopwords]

    return ' '.join(a_lemmas)


def find_persons(text, model=nlp):
    # Create Doc object
    doc = model(text)
    # Identify the persons
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

    # Return persons
    return persons


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


def token_text(text):
    doc = nlp.make_doc(text)
    tokens = [token.text for token in doc]
    return tokens


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


def get_doc_topic(model, corpus, model_out):
    topicnames = ['Topic_#' + str(i) for i in range(model.n_components)]
    # docnames = ['Doc_#' + str(i) for i in range(len(corpus))]
    docnames = corpus.index

    df = pd.DataFrame(np.round(model_out, 2),
                      columns=topicnames,
                      index=docnames)
    return df


def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


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


def get_jaccard_sim(str1, str2):
    a = set(str1)
    b = set(str2)
    c = a.intersection(b)
    return len(c) / (len(a) + len(b) - len(c))


def get_jaccard_score(df1, df2):
    origin = df1.str.split(',')
    pred = df2.str.split(',')
    res = pd.DataFrame(index=lst_index, columns=['jaccard_score'])
    for i in lst_index:
        res.at[i, 'jaccard_score'] = get_jaccard_sim(origin.loc[i],
                                                     pred.loc[i])

    return int(round(res.mean() * 100, 1))
