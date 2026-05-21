# Données du projet (non incluses dans Git)

Les dossiers suivants sont **volontairement exclus** du dépôt (taille > limite GitHub).

## 1. Base Roboflow (cours ARCHE)

1. Télécharger **Base d'apprentissage** sur [ARCHE — BE Twizy](https://arche.univ-lorraine.fr/course/view.php?id=18190).
2. Extraire dans le projet :

```text
RoadSignImageProcessing/dataset/
  train/   + _classes.csv
  valid/   + _classes.csv
  test/    + _classes.csv
```

3. Lancer **`DatasetFilter`** dans Eclipse → crée `dataset_filtered/`.

## 2. Images et vidéos de démo

Placer dans `external_images/` (depuis le cours ou vos fichiers) :

- `video1.avi`, `video2.avi`
- `p5.jpg`, `p9.jpg`, `ref90.jpg`, etc.

## 3. Modèles CNN

Soit copier `models/speed_cnn.onnx`, `other_cnn.onnx` et les `*_labels.txt` depuis le dépôt du groupe,

soit régénérer :

```bash
py -3 scripts/train_cnn.py
```
