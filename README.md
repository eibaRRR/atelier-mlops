# Atelier MLOps - Git + DVC + MLflow

Classification du dataset Iris avec un workflow MLOps complet :
Git (code), DVC (donnees et pipeline), MLflow (experimentations).

## Comparaison des experimentations

| Run | Modele | Parametres | Accuracy | Precision | Recall | F1-score |
|-----|--------|------------|----------|-----------|--------|----------|
| v1.0 | RandomForest | n_estimators=100, max_depth=5 | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| v1.0 | KNN | n_neighbors=5 | 1.0 | 1.0 | 1.0 | 1.0 |
| v2.0 | RandomForest | n_estimators=200, max_depth=8 | 0.90 | 0.9024 | 0.90 | 0.8997 |
| v2.0 | KNN | n_neighbors=5 | 1.0 | 1.0 | 1.0 | 1.0 |

Conclusion : augmenter le nombre d'arbres de 100 a 200 n'a pas ameliore le
Random Forest (0.9333 -> 0.90). Le KNN (k=5) reste le meilleur modele sur ce
split avec 100% sur toutes les metriques.

