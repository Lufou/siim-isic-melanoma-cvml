import pandas as pd
from tensorboardX import SummaryWriter

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("effnet_28K/training_results.csv")

# Create a TensorBoard SummaryWriter
writer = SummaryWriter()

# Log the training and validation loss
for epoch, row in df.iterrows():
    writer.add_scalars("Loss: Training(red) and Validation(blue)", {"Train": row["train_loss"], "Validation": row["val_loss"]}, epoch)

# Log the training and validation accuracy
for epoch, row in df.iterrows():
    writer.add_scalars("Accuracy: Training(orange) and Validation(green)", {"Train": row["train_accuracy"], "Validation": row["val_accuracy"]}, epoch)

# Close the SummaryWriter
writer.close()
