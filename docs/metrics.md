# Metriques: lire l'I-AUROC et l'expliquer au metier

**L'I-AUROC (Area Under the ROC Curve, au niveau image)** est la metrique de reference
en detection d'anomalies industrielle. Definition concrete: c'est la probabilite qu'une
image defectueuse prise au hasard recoive un score d'anomalie **plus eleve** qu'une image
normale prise au hasard. Classifieur parfait: 1.0. Tirage a pile ou face: 0.5.

## Pourquoi cette metrique plutot qu'une accuracy

1. **Pas besoin de seuil.** Le modele produit un score continu; l'AUROC evalue la qualite
   du classement independamment du seuil d'alerte. Le seuil se choisit APRES, selon les
   couts metier (rater un defaut vs declencher une fausse alarme).
2. **Insensible au desequilibre.** En production, 99% des pieces sont bonnes. Un modele
   qui dit "tout est bon" affiche 99% d'accuracy et zero valeur. L'AUROC compare les
   scores deux a deux: la majorite ne peut pas le tricher.
3. **Comparable** entre categories, entre modeles et avec la litterature: c'est le
   standard de MVTec-AD et VisA (Dinomaly multi-class: 0.996).

## Les autres metriques du papier, au-dela du niveau image

- **P-AUROC** (pixel): l'AUROC calculee pixel par pixel - est-ce que la carte pointe le
  defaut, et pas seulement le detecter? Indispensable quand un operateur doit localiser.
- **PRO**: sensibilite par region de defaut - reflet plus juste des petits defauts que le
  P-AUROC, qui favorise les grandes zones.
- **F1-max**: meilleur equilibre precision/rappel sur tous les seuils - c'est la metrique
  a regarder pour choisir le seuil de production.

## Comment l'expliquer au business

- Version simple: "Sur 100 duos (une piece bonne, une piece defectueuse), le modele
  met la defectueuse en haut de la pile 99 fois sur 100."
- Ce que ce n'est pas: un taux de detection a un seuil donne. L'AUROC dit que le modele
  RANGE bien; la mise en production exige ensuite de fixer le seuil d'alarme, et la on
  negocie avec le cout des erreurs - rater un defaut (retour client, rappel) coute
  presque toujours plus cher qu'une fausse alarme (re-inspection humaine). Le F1-max
  fournit le seuil qui equilibre les deux taux.
- Garde-fou a annoncer: 0.996 veut dire 4 duos mal ordonnes sur 1000. La question a poser
  au metier: combien de defauts rates par lot est tolerable, et a quel cout de fausses
  alarmes correspond ce seuil?
