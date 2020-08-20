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
