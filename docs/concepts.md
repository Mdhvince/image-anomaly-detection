# Concepts cles

## Transformer et attention

- **Vision Transformer (ViT)**: l'image est decoupee en patchs (14x14 px ici); chaque
  patch devient un **token** (vecteur de 768 dimensions), traite comme un mot dans une
  phrase.
- **Self-attention**: chaque token pondere tous les autres pour se construire ("qui
  est pertinent pour moi?").
- **Multi-head attention**: plusieurs attentions en parallele avec des projections
  differentes, concatenees a la fin - chaque tete apprend une relation differente.
- **Bloc transformer**: attention + MLP + connexions residuelles + normalisation,
  empile 12 fois dans DINOv2; Dinomaly exploite les couches 2 a 9 (`encoder.blocks[2:10]`).
- **Linear Attention (decoder)**: variante sans softmax, incapable de se concentrer
  sur soi-meme - donc de copier l'identite (`model.py::LinearAttention2`).
- **Registres** (DINOv2-reg): tokens supplementaires appris qui absorbent les normes
  aberrantes; Dinomaly les ignore, il ne garde que les tokens patch.

## Le coeur du sujet

- **Anomaly detection non supervisee**: entrainer sur des images NORMALES seulement;
  a l'inference, ce qui est mal reconstruit est anormal. Aucun exemple de defaut
  n'est requis.
- **Encodeur / decodeur / reconstruction**: l'encodeur transforme l'image en features;
  le decodeur apprend a les reproduire. L'erreur de reconstruction = signal d'anomalie.
- **Bottleneck + Dropout**: goulot d'etranglement bruite qui empeche le decodeur
  d'apprendre l'identite - la propriete qui rend possible un modele unique pour
  toutes les categories (`model.py::Dinomaly.bottleneck`).
- **Similarite cosinus**: l'angle entre deux vecteurs, dans [-1, 1]; la loss vaut
  1 - cos entre features attendues et reconstruites, par groupe, puis par point pour
  la carte d'anomalie.
- **Hard example mining**: attenuer le gradient des points deja bien reconstruits
  pour concentrer l'apprentissage sur les difficiles (`loss.py`).

## Optimisation

- **AdamW**: l'optimiseur standard des transformers (momentum + adaptation par
  parametre, weight decay decouple).
- **Warmup + cosine schedule**: le learning rate monte doucement puis decroit en
  cosinus (`loss.py::compute_lr_ratio`).
- **Gradient clipping**: plafonner la norme du gradient (`max_grad_norm = 0.1`) pour
  absorber les pics dus au dropout du bottleneck.

## Evaluation

- **AUROC**: probabilite qu'une anomalie aleatoire ait un score superieur a une image
  normale aleatoire; image-level (detection) vs pixel-level (localisation) - voir
  [metrics.md](metrics.md).
- **Score image**: moyenne du top 1% de la carte d'anomalie (`inference.py::image_scores`).

## PyTorch

- **Hooks (`Tensor.register_hook`)**: intercepter un gradient pendant le backward -
  la mecanique du hard mining (`loss.py::modify_grad`).
