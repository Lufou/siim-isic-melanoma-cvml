import pandas as pd
file_path1 = "28K_efficientnet_b3_15ep_64batch.csv"
file_path2 = "28K_resnet50_25e_batch64.csv"
file_path3 = "28K_seresnext50d_15ep_64batch.csv"
file_path4 = "44K_efficientnet_b3_15ep_64batch.csv"
file_path5 = "44K_seresnext50_15ep_64batch.csv"
#file_path6 = "44K_resnet50_30ep_batch64.csv"

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)
df5 = pd.read_csv(file_path5)
#df6 = pd.read_csv(file_path6)

# moyenne de la colonne "target"
df1['target'] = (2*df1['target'] + df2['target'] + 2*df3['target']+ df4['target']+ df5['target'])/7

output_file_path = "Moyenne.csv"
df1.to_csv(output_file_path, index=False)
print(f"La moyenne des colonnes 'target' a été sauvegardée dans {output_file_path}")
