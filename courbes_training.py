import pandas as pd
from tensorboardX import SummaryWriter

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("effnet_28K/training_results.csv")

# Create a TensorBoard SummaryWriter
writer = SummaryWriter()

# Log the training loss and accuracy
for epoch, row in df.iterrows():
    writer.add_scalar("Train/Loss", row["train_loss"], epoch)
    writer.add_scalar("Train/Accuracy", row["train_accuracy"], epoch)

# Log the validation loss and accuracy
for epoch, row in df.iterrows():
    writer.add_scalar("Validation/Loss", row["val_loss"], epoch)
    writer.add_scalar("Validation/Accuracy", row["val_accuracy"], epoch)

# Close the SummaryWriter
writer.close()
