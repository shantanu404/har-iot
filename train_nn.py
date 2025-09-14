import glob
import os
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.io import arff
from seaborn import heatmap
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def from_activity_to_label(activity):
    if activity == "A":
        return "Walking"
    elif activity == "B":
        return "Jogging"
    elif activity == "C":
        return "Stairs"
    elif activity == "D":
        return "Sitting"
    elif activity == "E":
        return "Standing"
    elif activity == "F":
        return "Typing"
    elif activity == "G":
        return "Brushing Teeth"
    elif activity == "H":
        return "Eating Soup"
    elif activity == "I":
        return "Eating Chips"
    elif activity == "J":
        return "Eating Pasta"
    elif activity == "K":
        return "Drinking from Cup"
    elif activity == "L":
        return "Eating Sandwich"
    elif activity == "M":
        return "Kicking (Soccer Ball)"
    elif activity == "O":
        return "Playing Catch w/ Tennis Ball"
    elif activity == "P":
        return "Dribbling (Basketball)"
    elif activity == "Q":
        return "Writing"
    elif activity == "R":
        return "Clapping"
    elif activity == "S":
        return "Folding Clothes"
    else:
        return "Unknown"


def load_single_data_file(subject_id, device, sensor):
    data, _ = arff.loadarff(
        f"dataset/{device}_arff/{sensor}/data_{subject_id}_{sensor}_{device}.arff"
    )
    df = pd.DataFrame(data)
    df.columns = df.columns.str.strip('"')
    df.columns = map(
        lambda column_name: sensor.upper() + "_" + column_name
        if column_name[0] in ["X", "Y", "Z", "R"]
        else column_name,
        df.columns,
    )
    df["ACTIVITY"] = df["ACTIVITY"].str.decode("utf-8")
    df["class"] = df["class"].str.decode("utf-8")
    return df


def pipeline(data, name="Unknown"):
    X_full, y_full = data.drop(columns=["ACTIVITY"]), data["ACTIVITY"]
    print(X_full.shape, y_full.shape)

    # Plot tSNE
    tsne = TSNE(n_components=2, random_state=42)
    features = X_full.values
    tsne_results = tsne.fit_transform(features)
    categories = data["ACTIVITY"].astype("category").cat.categories
    codes = data["ACTIVITY"].astype("category").cat.codes

    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 8))
    for i, activity in enumerate(categories.unique()):
        idx = codes == i
        plt.scatter(
            tsne_results[idx, 0],
            tsne_results[idx, 1],
            color=colors[i % len(colors)],
            label=activity,
            alpha=0.7,
        )
    plt.legend(title="Activities", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(f"t-SNE of Sensor Data Colored by Activity - {name}")
    plt.savefig(f"neural_network/tsne_{name}.png")

    # Encode labels (one-hot)
    y_full_cat = pd.Categorical(y_full)
    y_full_codes = y_full_cat.codes
    class_names = list(y_full_cat.categories)

    # One-hot encode labels
    y_full_onehot = pd.get_dummies(y_full_cat)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full.values,
        y_full_onehot.values,
        test_size=0.2,
        random_state=42,
        stratify=y_full_codes,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full.argmax(axis=1),
    )

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Define MLP model
    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.Sigmoid(),

                nn.Linear(1024, 2048),
                nn.Sigmoid(),

                nn.Linear(2048, 2048),
                nn.Sigmoid(),

                nn.Linear(2048, 2048),
                nn.Sigmoid(),

                nn.Linear(2048, 1024),
                nn.Sigmoid(),

                nn.Linear(1024, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    print("Training MLP...")
    print(X_train.shape, y_train.shape)
    model = MLP(X_train.shape[1], y_train.shape[1]).to(DEVICE)

    input_sample = torch.randn(1, X_train.shape[1]).to(DEVICE)
    output_sample = model(input_sample)
    print("Input sample shape:", input_sample.shape)
    print("Output sample shape:", output_sample.shape)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 500
    training_losses = []
    validation_losses = []

    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        training_losses.append(epoch_loss)

        # Validation loss
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        validation_losses.append(val_epoch_loss)

        print(
            f"Epoch {epoch+1}/{n_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}"
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {name}")
    plt.legend()
    plt.savefig(f"neural_network/loss_{name}.png")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
    plt.figure(figsize=(10, 8))
    conf_mat = confusion_matrix(all_labels, all_preds)
    plt.title(f"Confusion Matrix for PyTorch MLP - {name}")
    heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.savefig(f"neural_network/confusion_matrix_{name}.png")


possible_devices = ["phone", "watch"]
possible_sensors = ["accel", "gyro"]

for device, sensor in product(possible_devices, possible_sensors):
    print(f"Device: {device}, Sensor: {sensor}")
    filelist = glob.glob(f"dataset/{device}*/{sensor}/*.arff")
    data = []

    for file in filelist:
        filename = os.path.basename(file)
        _, subject_id, _, _ = filename.replace(".arff", "").split("_")
        df = load_single_data_file(subject_id, device, sensor)
        data.append(df)

    data = pd.concat(data, axis=0)
    data["ACTIVITY"] = data["ACTIVITY"].apply(from_activity_to_label)
    data_activity = data.drop(
        columns=["class"]
    )  # Subject is not important for classification

    pipeline(
        data_activity,
        name=f"{device}-{sensor}",
    )
