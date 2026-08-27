# Dinomaly - detection d'anomalies industrielle sur MVTec-AD

Implementation minimale et fidele du papier [Dinomaly](https://arxiv.org/abs/2405.14325)
(CVPR 2025), croisee avec le code officiel
[guojiajeremy/Dinomaly](https://github.com/guojiajeremy/Dinomaly).

L'idee en une phrase: un **ViT pre-entraine gele** extrait les features de l'image, un
**decoder entraine sur des images normales** doit les reconstruire, et la **distance
encoder vs decoder**, point par point, localise ce que le decoder n'a jamais appris a
reconstruire: l'anomalie.

```
image --> [ViT DINOv2 gele] --> 8 feature maps --> [bottleneck MLP + Dropout]
      --> [decoder 8 blocs, linear attention] --> distance cosinus par patch
      --> carte d'anomalie --> score image (top 1%)
```

## Lancer

```bash
# environnement (une fois)
python -m venv .venv
.venv/bin/pip install torch torchvision pandas scikit-learn matplotlib

# dataset MVTec-AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
# a placer sous ../datasets/mvtec_anomaly_detection (ou DINOMALITY_DATA_ROOT)

python train.py        # indexe le dataset, entraine un modele par categorie -> checkpoints/<categorie>.pt
python inference.py    # recharge chaque checkpoint, AUROC par categorie + moyenne
```

Regimen par defaut: **DEMO**, defini dans `config.ini` (`[demo] enabled = true` - memes
mecaniques, metriques plus basses). Regimen papier complet: `enabled = false`.

Toute la configuration vit dans **`config.ini`** (sections dataset, model, preprocessing,
training, evaluation, demo - editer le fichier puis relancer). Pour pointer vers un autre
fichier (ex: un sweep d'experiences): `DINOMALITY_CONFIG=experiences/run42.ini python train.py`.

## Structure

| Fichier | Role |
|---|---|
| `config.py` | charge `config.ini` (valeurs du papier par defaut, regimen DEMO optionnel) |
| `preprocess.py` | pipeline image partage train/inference (resize -> crop -> normalisation ImageNet) |
| `dataset.py` | indexation MVTec-AD -> CSV plat, dataset, loaders train/valid/test |
| `model.py` | encodeur gele DINOv2, noisy bottleneck, decodeur linear attention, assemblage |
| `loss.py` | cosinus global + hard mining, optimiseur + schedule |
| `train.py` | boucle par epochs, checkpoint sauvegarde quand la loss de validation diminue |
| `inference.py` | cartes d'anomalie, score image, I-AUROC |
| `visualize.py` | figures: preview dataset, check preprocessing, courbe de loss, cartes |

## Contrat de donnees (index.csv)

La seule connaissance de l'arborescence officielle de MVTec-AD vit dans
`dataset.py::index_mvtec`. Tout le reste du code ne consomme que le CSV plat, ecrit a la
racine du dataset a chaque lancement (relit l'arborescence, ~1s, donc toujours a jour):

```
category,split,label,image_path,mask_path
hazelnut,train,0.0,train/good/000.png,
hazelnut,test,1.0,test/crack/000.png,ground_truth/crack/000_mask.png
```

Colonnes:

1. `category` / `split`: provenance de l'image (`train` ou `test`).
2. `label`: 0.0 = normal, 1.0 = anomalie.
3. `image_path`: chemin de l'image, relatif au dossier de la categorie.
4. `mask_path`: masque binaire (blanc = defaut), vide pour les images normales.

Pour brancher un autre dataset: produire un CSV dans ce format, rien d'autre ne change.

## Documentation

- `docs/method.md`: la methode pas a pas - pourquoi chaque brique (encoder gele, noisy
  bottleneck, linear attention, contrainte loose, hard mining).
- `docs/metrics.md`: les metriques (I-AUROC, P-AUROC, PRO, F1-max) et comment les
  expliquer au business.