# siim-isic-melanoma-cvml
Step 1 : prétraitement:
  1. ink_hair_rotation.py génère des images mélanomes artif
  2. supp_demi_no_mel.py supprime la moitié des images pas mélanomes
  3. coherence vérifie si train-resized et train-labels.csv sont cohérents après suppression et ajout d'images

Step 2: Training: avec resnet50, seresnext50 et efficientnet_b3

Step 3 : testpred_.py utilisé pour l'entraînement -> on obtient le résultat .csv

Step 4: Faire la moyenne entre resnet50, seresnext50, efficientnet_b3
