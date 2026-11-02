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

## Reponses aux questions de reflexion

**15. Quelle est la difference entre Git et DVC ?**
Git versionne le code source et les petits fichiers de configuration (commits, branches, tags). DVC versionne les fichiers volumineux (datasets, modeles) via de petits fichiers `.dvc` qui contiennent une empreinte (hash) du contenu. Git suit ces fichiers `.dvc`, DVC gere le contenu reel dans son cache et un stockage distant.

**16. Pourquoi eviter de stocker un dataset volumineux directement dans Git ?**
Un depot Git devient rapidement trop lourd a cloner et pousser ; Git est optimise pour du texte, pas pour des fichiers binaires volumineux. Chaque modification duplique le fichier dans l'historique. DVC stocke le contenu une seule fois dans son cache (deduplique par hash) et ne met dans Git qu'une petite reference.

**17. Quelle est la difference entre `dvc add` et `dvc repro` ?**
`dvc add` commence a suivre un fichier existant (dataset, modele) en creant son fichier `.dvc`. `dvc repro` execute le pipeline defini dans `dvc.yaml` : il detecte les dependances qui ont change et relance uniquement les stages necessaires, puis met a jour `dvc.lock`.

**18. Quel est le role du fichier `dvc.lock` ?**
Il enregistre les versions resolues (hashs) des dependances, parametres et sorties de chaque stage. Il garantit la reproductibilite : deux personnes avec le meme `dvc.lock` obtiennent exactement les memes donnees et le meme pipeline.

**19. Comment MLflow permet-il de comparer deux experimentations ?**
Chaque run enregistre ses parametres, metriques et artefacts dans un backend de tracking. L'interface MLflow (`mlflow ui`) affiche un tableau comparatif des runs : on peut trier par metrique, filtrer par parametre, et visualiser des graphiques comparatifs (accuracy, F1, etc.) entre Random Forest et KNN par exemple.

**20. Comment retrouver les donnees et le modele associes a un commit Git precis ?**
`git checkout <tag-ou-commit>` restaure les fichiers `.dvc`, `dvc.yaml`, `dvc.lock` et `params.yaml` de ce commit. Puis `dvc checkout` restaure les donnees/modeles depuis le cache local, ou `dvc pull` les recupere depuis le remote DVC. La combinaison commit Git + hash DVC reconstitue l'etat exact du projet.

**21. Pourquoi la reproductibilite est-elle importante dans un projet d'IA ?**
Sans reproductibilite, impossible de debugger un modele en production, d'auditer une decision, de prouver un resultat scientifique ou de reentrainer apres un incident. Un meme code + memes donnees + memes parametres doit donner le meme modele et les memes metriques.

**22. Difference entre le versioning des modeles avec DVC et leur suivi avec MLflow ?**
DVC versionne le fichier du modele comme une donnee : il gere ses versions (hashes) et permet de restaurer une version anterieure. MLflow suit le modele comme une experimentation : il l'associe aux parametres, metriques, code et dataset qui l'ont produit, pour comparaison et selection. DVC repond a quelle version du fichier, MLflow repond a quel run l'a produit et avec quels resultats.

## Comment reproduire

```bash
pip install -r requirements.txt
dvc pull
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
