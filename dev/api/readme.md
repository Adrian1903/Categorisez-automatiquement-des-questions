# Recommendation de tags pour le site StackOverFlow
Suggère des tags pour une question posée sur le site de questions/réponses stackoverflow

## Lien vers API
https://tags-suggestions.herokuapp.com/

## Description
L'API doit aider les utilisateurs débutants à insérer les bons tags à leurs questions.
De ce fait, le programme prédira un certains nombre de mot-clés. Cela se fera de manière non supervisée et de manière supervisée selon une question donnée.

# Choix du modèle de suggestion
2 modèles de suggestion sont disponibles : 
- models_10k ; entrainé sur 10 000 questions avec un score de Jaccard à 40%
- models_50k ; entrainé sur 50 000 questions avec un score de Jaccard à 44%
models_10k est déployé en production sur Heroku, 
models_50k est à déployer en local si vous souhaitez le tester.

## Installation en local
1. git clone https://github.com/Adrian1903/Categorisez-automatiquement-des-questions   
2. Aller dans le dossier api du projet   
3. Désarchiver le modèle à tester dans le dossier src (models_10k ou models_50k)   
4. Installer les composants requis dans "requirements.txt"   
5. Dans tagger.py, ligne 11 à 14, choisir le seuil de probabilité adapté au modèle à tester (0.11 si models_10k, 0.15 si models_50k)   
6. Exécuter tagger.py   
7. Tester le modèle en se connectant à [localhost](http://127.0.0.1:5000)   

## Support
Je reste disponible pour plus d'informations
adrian@datareporting.fr