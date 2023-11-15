# siim-isic-melanoma-cvml
Step 1 : prétraitement:
  1. artif_mel.py génère des images mélanomes artif
  2. supp_demi_no_mel.py supprime la moitié des images pas mélanomes
  3. coherence vérifie si train-resized et train-labels.csv sont cohérents après suppression et ajout d'images

Step 2: Training: choisir resnet.py (resnet18 ou resnet34) ou seresnext.py ou encore efficientnet.py

Step 3 : testpred_nom du réseau utilisé pour l'entraînement -> on obtient le résultat .csv

Step 4: arrondi.py pour arrondir la colonne target du résultat

Step 5: Faire la moyenne entre resnet18, resnet34, seresnext, efficientnet_b0
