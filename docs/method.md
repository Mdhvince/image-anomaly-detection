# La methode pas a pas

Pourquoi chaque brique existe. Les renvois de code pointent vers les modules du depot.
Voir aussi: [concepts.md](concepts.md) (prerequis en bullet points) -
[metrics.md](metrics.md) (AUROC image vs pixel).

## 01 - Donnees: MVTec-AD

L'entrainement ne voit que des images **normales**: c'est le "unsupervised" de la
detection d'anomalies. Le jeu de test melange images normales et anormales, avec un
masque de verite terrain pour les anormales.

Le contrat entre les modules n'est pas une arborescence mais un **CSV d'index**,
`data_root/index.csv` produit par `dataset.py::index_mvtec` - une ligne par image:

    category,split,label,image_path,mask_path

1. `category` / `split`: provenance de l'image (ex: `bottle`, `train` ou `test`).
2. `label`: 0 normal, 1 anomalie (seulement `test` contient des 1).
3. `image_path`: chemin relatif au dossier de la categorie (le code prefixe avec
   `data_root / category`).
4. `mask_path`: masque pixel-level (vide pour les normales).

La seule cellule qui connait l'arborescence officielle (`train/good`, `test/<defaut>`,
`ground_truth/<defaut>/<nom>_mask.png`) est l'indexeur. Pour brancher un autre
dataset: produire un CSV dans ce format, rien d'autre ne change.

Le preprocessing (`preprocess.py::apply_preprocessing`) est le meme a l'entrainement
et a l'inference: resize -> crop central -> normalisation ImageNet. Les masques suivent
la meme geometrie mais restent binaires (resize en plus-proche-voisin, pas de
normalisation).

## 02 - Encodeur gele: DINOv2

Le backbone n'est pas entraine: Dinomaly prend un **ViT DINOv2 (avec registers)
pre-entraine**, gele - seul le bottleneck et le decoder apprennent. Deux choix du
papier:

- **Les couches du milieu (2 a 9 sur 12)**: les premieres couches voient des textures
  brutes, les dernieres une semantique trop abstraite pour des defauts fins. Les 8
  couches intermediaires donnent un melange de features bas niveau (utiles pour
  localiser) et haut niveau (utiles pour reconnaitre la nature du defaut).
- **Gele et toujours en eval**: le modele reconstruit une representation fixe du monde
  normal; s'il bouge, la distance encoder/decoder perd son sens. Le "no cls token, no
  registers" du forward: on ne garde que les tokens patch, un par patch de 14x14.

## 03 - Noisy Bottleneck

Les 8 couches d'encoder sont **fusionnees par moyenne** puis passent dans un seul
**MLP avec Dropout** (p = 0.2): Linear -> GELU -> Dropout -> Linear. C'est tout.

Le Dropout remplace les generateurs de pseudo-anomalies des autres methodes: il
corrompt les features normales, le decoder doit les **restaurer**. Le papier montre
que ce bruit simple est ce qui empeche le decoder de reconstruire aussi les anomalies
en multi-class, et qu'il est robuste: p = 0.1 a 0.4 donnent des resultats proches.

## 04 - Decodeur a Linear Attention

Le decoder empile 8 blocs Transformer classiques, mais son attention spatiale remplace
le Softmax par une **Linear Attention**:

- Softmax(QK^T) concentre le poids sur les positions similaires a la requete, y compris
  elle-meme -> canal ideal pour l'identity mapping (une matrice attention diagonale
  copie l'entree, comme un kernel de convolution 1 au centre).
- Linear Attention (phi(x) = elu(x) + 1) **ne sait pas se concentrer**: son attention
  s'etale sur toute l'image. Chaque position est restauree depuis le contexte global,
  jamais copiee d'elle-meme. Son defaut sur les taches supervisees est exactement la
  propriete recherchee ici.
- Bonus: en calculant K^T V d'abord, la complexite passe de O(N^2 d) a O(N d^2).

## 05 - Assemblage: contrainte loose et groupes croises

Les methodes classiques appaient chaque couche du decoder a une couche de l'encoder
(ou tout a la derniere). Plus la supervision est precise, plus le decoder **imite**
l'encoder - donc reconstruit aussi les anomalies. Dinomaly desserre la contrainte:

- les 8 couches d'encoder sont **sommees en 2 groupes** (couches 2-5 = bas niveau,
  bon pour la localisation; 6-9 = haut niveau) et le decoder doit reconstruire
  chaque groupe total, avec plus de liberte;
- detail du code officiel (non documente dans le papier): les sorties du decoder
  sont **inversees** avant le groupement -> le groupe profond du decoder reconstruit
  le groupe bas niveau de l'encoder et vice versa (croisement type U-Net);
- la fusion de groupe est une **moyenne** de couches (equivalente a la somme pour une
  distance cosinus, invariante d'echelle).

## 06 - Loose loss: cosinus global + hard mining

Deux mecanismes distincts (`loss.py::global_cosine_hm`):

1. **Valeur de la loss**: distance cosinus **globale** entre groupe encoder et groupe
   decoder (maps aplaties, une valeur par image et par groupe), moyennee sur les 2
   groupes et sur le batch.
2. **Gradients (le vrai hard mining)**: la distance cosinus est aussi calculee **par
   point**; les `mining_percent`% de points les **plus faciles** (distance la plus
   basse, deja bien reconstruits) voient leur gradient **multiplie par
   `shrink_factor`** (0.1) via un hook `Tensor.register_hook` pose sur la sortie du
   decoder, qui se declenche dans `loss.backward()`. A convergence, seuls les 10% de
   points les plus difficiles gardent un gradient plein - et ce sont eux qui comptent.

Vocabulaire de la config:

- **mining / hard example mining**: ignorer (presque) les points deja bien reconstruits
  pour concentrer l'apprentissage sur les plus difficiles.
- **`final_mining_percent = 0.9`**: a convergence, 90% des points sont consideres
  "faciles" (gradient x0.1); la loss est faite par le 10% le plus dur.
- **`mining_ramp_iters = 1000`**: le taux de mining monte **lineairement** de 0 a
  `final_mining_percent` sur les 1000 premieres iterations: la loss commence comme un
  simple cosinus global, puis se concentre (detail du code officiel, absent du papier).
  Miner trop tot serait absurde: au debut tout est difficile, rien n'est "facile".
- **`shrink_factor = 0.1`**: multiplicateur du gradient sur les points faciles (le
  sg(.)_shrink du papier). Pas 0: on attenue, on ne tue pas le gradient.

Optimiseur: le papier utilise StableAdamW (lr 2e-3, wd 1e-4, AMSGrad); on prend
l'alternative standard `torch.optim.AdamW(amsgrad=True)` + warmup/cosine + clip 0.1.

## 07 - Training

Mode **multi-class** (le coeur du papier): **un seul modele pour toutes les
categories**, entraine sur toutes les images normales fusionnees. C'est le dropout du
bottleneck (section 03) qui rend ce partage possible: un decodeur qui ne peut pas
copier l'identite se voit interdire de "reconnaitre" la categorie.

Boucle par epochs (`train.py::train_dinomaly`): phase train sur les images normales
(forward -> loss avec hooks de mining -> clip de gradient 0.1 -> step -> scheduler),
puis phase validation sur un split d'images normales mis de cote (`valid_ratio`, sans
backward). Le checkpoint est **re-sauvegarde a chaque fois que la loss de validation
diminue** - jamais en fin d'entrainement. Seuls les poids du bottleneck et du decoder
sont sauvegardes dans `checkpoints/dinomaly.pt`. A la fin de l'entrainement, les
poids de la meilleure validation sont recharges et un **seuil de decision** est calcule
(= score maximum parmi les images normales tenues a l'ecart), puis stocke dans le
checkpoint sous la cle `"threshold"`.

Suivi dans **TensorBoard**: losses train/validation par epoch (scalars) et la grille
du **1er batch d'entrainement** (denormalise, pour verifier le preprocessing):

    tensorboard --logdir runs

## 08 - Inference: carte d'anomalie et score image

La **carte d'anomalie** = distance cosinus **par point** entre groupes encoder et
decoder, moyennee sur les 2 groupes, puis upsamplee a la taille de l'image.
Le **score image** = moyenne du top 1% des valeurs de la carte (plus robuste que le max).
Une seule passe forward donne les trois taches: segmentation (la carte), detection
(score vs seuil), classification (anormal ou non). Evaluation (`inference.py`): le
checkpoint unique est recharge une fois, puis l'I-AUROC est calculee **par categorie**
+ en **moyenne** sur toutes (le protocole multi-class du papier), avec le **compte
d'anomalies detectees** au seuil calcule. Un checkpoint ancien (sans seuil) le fait
recalculer automatiquement depuis le split de validation.

Inference en condition reelle sur une image seule (n'importe quel fichier):
`python inference_one_image.py <chemin>` -> score image + **decision normale/ANOMALIE**
+ fenetre matplotlib (image | carte superposee).

## Recapitulatif

| Brique | Choix du papier | Dans ce depot |
|---|---|---|
| Encoder | DINOv2-R ViT-Base/14 gele, couches 2-9 | idem |
| Bottleneck | MLP 768->3072->768, Dropout 0.2 | idem |
| Decoder | 8 blocs, Linear Attention (elu+1) | idem |
| Contrainte | 2 groupes bas/haut niveau, croises | idem |
| Loss | cosinus global + hard mining x0.1, p: 0 -> 0.9 | idem |
| Training | un modele unique, toutes categories (multi-class) | idem |
| Score image | moyenne du top 1% de la carte | idem |

Regimen DEMO assume: 224px (au lieu de 392), 1500 iterations (au lieu de 10 000),
batch 4 (au lieu de 16), AdamW+AMSGrad (au lieu de StableAdamW).