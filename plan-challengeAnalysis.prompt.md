# Plan: Notebook Compréhension & Optimisation

## TL;DR
Créer un nouveau notebook pédagogique qui reprend cell par cell le `data_instruction.ipynb`, en expliquant chaque concept, identifiant les erreurs/bugs, et proposant des corrections et optimisations. Le notebook existant contient ~10 bugs significatifs (split incorrect, labels corrompus, NaN silencieux, path Colab, pas de normalisation, pas d'early stopping, etc.).

---

## Phase 1 — EDA & Compréhension des données (Cells 1–3)

### Step 1.1 – Markdown d'introduction (reprendre cells 1-3)
- Reprendre la description du challenge, la structure des données, le split sujets
- **Signaler l'incohérence du split** : le markdown dit 24 train/val + 8 test, mais le code fait 20+7+5

### Step 1.2 – Exploration des fichiers bruts
- Charger et afficher un `classif.csv` et un `insoles.csv` (head, shape, dtypes, colonnes)
- Vérifier le format : séparateur `;`, colonne `Time`, colonnes pression/accélération/etc.
- Identifier les valeurs manquantes (NaN) dans `insoles.csv`
- Vérifier le nombre de séquences par sujet (tous ont 10)

### Step 1.3 – Statistiques descriptives des features
- Min, max, mean, std par type de feature (pression, accélération, angular, force, CoP)
- Visualiser les distributions (histogrammes, boxplots)
- Montrer que les échelles sont très différentes → justifier la normalisation

### Step 1.4 – Analyse des annotations
- Distribution des classes (nombre d'occurrences, durée moyenne par classe)
- Visualiser le déséquilibre des classes
- Timeline d'une séquence type (actions vs temps)

---

## Phase 2 — Dataset & DataLoader (Cells 4–6)

### Step 2.1 – Reprendre le code de chargement et l'expliquer ligne par ligne
- `ROOT`, `events_root`, `plantar_root` : construction des chemins
- `WindowedPlantarDataset` : expliquer le concept de fenêtrage temporel
- Expliquer `LabelEncoder`, `overlap_len`, `np.searchsorted`

### Step 2.2 – Identifier et corriger les bugs du Dataset

**Bug 1 : Split sujets incorrect**
- Code : `val_subjects = range(21, 28)` → S21-S27, `test_subjects = range(28, 33)` → S28-S32
- Spec : val = S21-S24, test = S25-S32
- Fix : `val_subjects = range(21, 25)`, `test_subjects = range(25, 33)`

**Bug 2 : Labels corrompus par la logique d'overlap**
- Quand ratio < 0.70, la fenêtre reçoit le label de l'annotation SUIVANTE → data poisoning
- Le fix : utiliser l'annotation avec le plus grand overlap, ou ne garder que les fenêtres avec ratio >= threshold et ignorer le reste

**Bug 3 : Suppression first/last annotations sans justification**
- `ann.iloc[1:-1]` supprime la première et dernière action de chaque séquence
- Hypothèse : peut-être T-pose initiale/finale, mais à vérifier et documenter

**Bug 4 : NaN silencieux**
- `insoles.csv` contient des cellules vides → NaN dans le DataFrame
- `.values.astype(np.float32)` propage les NaN → loss NaN, modèle corrompu
- Fix : imputation (ffill/bfill) ou détection et skip des samples NaN

**Bug 5 : Une seule fenêtre par annotation (pas de sliding window)**
- Sous-utilise les données, crée du déséquilibre
- Optimisation : sliding window avec stride (ex: 0.5s ou 1.0s)

### Step 2.3 – Corriger et re-exécuter
- Implémenter les corrections
- Afficher la distribution des classes corrigée
- Comparer avant/après

---

## Phase 3 — Normalisation et prétraitement (Nouveau)

### Step 3.1 – Calculer les statistiques de normalisation sur le train set
- StandardScaler fit sur train, transform sur val/test
- Stocker mean/std par feature

### Step 3.2 – Intégrer la normalisation dans le Dataset ou via transform

---

## Phase 4 — Modèle (Cells 8–10)

### Step 4.1 – Reprendre SimpleNet et expliquer l'architecture
- Conv1d : pourquoi 1D ? (signal temporel)
- Shape : [B, T, F] → transpose → [B, F, T] → Conv1d → AdaptiveAvgPool1d → Linear
- Expliquer le rôle de chaque couche

### Step 4.2 – Identifier les faiblesses
- **AdaptiveAvgPool1d(1)** : écrase toute l'info temporelle en 1 valeur par canal → perte de temporal dynamics
- **Pas de BatchNorm** → gradient instable
- **Pas de Dropout** → overfitting probable
- **2 couches seulement** → sous-capacité pour 50 features et ~30 classes
- **kernel_size=5, padding=2** → réceptif field de seulement 9 timesteps (0.09s) sur 300

### Step 4.3 – Corriger le path de sauvegarde
- **Bug 6** : `save_path = "/content/drive/MyDrive/best_model.pth"` → path Colab, crash en local
- Fix : utiliser un chemin relatif au workspace

### Step 4.4 – Proposer des améliorations
- Ajouter BatchNorm + Dropout
- Augmenter le nombre de couches
- Considérer un réseau plus adapté (LSTM, Transformer, ResNet-1D)

---

## Phase 5 — Entraînement (Cell 10)

### Step 5.1 – Expliquer la boucle d'entraînement
- `train_one_epoch` : forward → loss → backward → step
- `evaluate` : no_grad, accumulation des prédictions, confusion matrix

### Step 5.2 – Identifier les problèmes
- **Bug 7** : Seulement 19 epochs (`range(1, 20)` = 1..19), pas de early stopping
- **Pas de scheduler LR** → learning rate fixe 1e-3
- **Pas de class weighting** → biais vers classes majoritaires
- **Pas de seed/reproductibilité**

### Step 5.3 – Implémenter les corrections
- Early stopping avec patience
- ReduceLROnPlateau
- CrossEntropyLoss avec poids de classes (compute_class_weight)
- torch.manual_seed + np.random.seed

---

## Phase 6 — Évaluation & Reporting (Cells 12–14)

### Step 6.1 – Reprendre les visualisations
- Courbes loss/accuracy train vs val
- Matrice de confusion sur test set

### Step 6.2 – Ajouter des métriques manquantes
- F1-score macro et weighted
- classification_report complet (precision, recall, f1 par classe)
- Accuracy par classe
- Identifier les classes les plus confondues dans la matrice

---

## Fichiers concernés
- `data_instruction.ipynb` — notebook source (lecture seule, référence)
- Nouveau notebook à créer (ex: `challenge_analysis.ipynb`) — notebook de travail
- `DataChallenge_donnees/Events/*/Sequence_*/classif.csv` — annotations
- `DataChallenge_donnees/Plantar_activity/*/Sequence_*/insoles.csv` — capteurs

## Résumé des bugs identifiés
1. Split val/test incohérent avec la spec (range(21,28) et range(28,33) au lieu de range(21,25) et range(25,33))
2. Labels corrompus quand overlap < 70% (utilise la classe de l'annotation suivante)
3. Suppression first/last annotations sans justification
4. NaN silencieux dans insoles.csv non gérés
5. Une seule fenêtre par annotation (pas de sliding window)
6. Path de sauvegarde hardcodé Colab
7. Seulement 19 epochs, pas d'early stopping
8. Pas de normalisation des features
9. Pas de class weighting
10. Pas de BatchNorm/Dropout dans le modèle
11. AdaptiveAvgPool1d(1) écrase l'info temporelle
12. Pas de scheduler LR

## Vérification
1. Exécuter chaque cell du nouveau notebook séquentiellement — aucune erreur
2. Comparer la distribution des classes avant/après correction du split
3. Vérifier qu'aucun NaN ne persiste après preprocessing (assert not np.isnan(x).any())
4. Comparer accuracy train vs val pour détecter overfitting/underfitting
5. Vérifier que le F1-macro s'améliore par rapport au baseline SimpleNet original
