
# IA.md — TP4 RAG Web

**Étudiante :** Nada Marnissi

## 1. Problème de mémoire sur Payara

J'ai utilisé ChatGPT pour comprendre une erreur HTTP 500 (`OutOfMemoryError: Java heap space`) rencontrée lors du chargement des deux PDF pour le RAG.

L'IA m'a aidée à identifier le problème et à augmenter la mémoire de Payara de 512 Mo à 2 Go.

## 2. Amélioration de la recherche dans les PDF

J'ai utilisé ChatGPT pour comprendre pourquoi Gemini ne retrouvait pas tous les noms des planètes dans le document `autre.pdf`.

L'IA m'a aidée à ajuster les paramètres du RAG (`maxResults` et `minScore`) pour récupérer davantage de passages pertinents.

Après modification, Gemini a correctement retrouvé les huit planètes du système solaire.
