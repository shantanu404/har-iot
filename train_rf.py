import glob
import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from scipy.io import arff
from seaborn import heatmap
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import optuna


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
    plt.savefig(f"random_forest/tsne_{name}.png")

    # Encode labels
    y_full_cat = pd.Categorical(y_full)
    y_full_codes = y_full_cat.codes
    class_names = list(y_full_cat.categories)

    # Save categorical codes to code.txt
    with open(f"random_forest/code.txt", "w") as f:
        for code, label in enumerate(class_names):
            f.write(f"{code}: {label}\n")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full.values,
        y_full_codes,
        test_size=0.2,
        random_state=42,
        stratify=y_full_codes,
    )

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 300),
            'max_depth': trial.suggest_int('max_depth', 15, 20),
        }
        model = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_train_full, y_train_full, cv=5, scoring='f1_weighted')
        return scores.mean()

    print("Running Optuna hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best trial:", study.best_trial.params)

    best_params = study.best_trial.params
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train_full, y_train_full)

    # Save model
    os.makedirs("random_forest", exist_ok=True)
    with open(f"random_forest/model_{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:20]
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [X_full.columns[i] for i in indices], rotation=90)
    plt.title(f"Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(f"random_forest/feature_importance_{name}.png")

    # Evaluation
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=class_names))
    plt.figure(figsize=(10, 8))
    conf_mat = confusion_matrix(y_test, preds)
    plt.title(f"Confusion Matrix for Random Forest - {name}")
    heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.savefig(f"random_forest/confusion_matrix_{name}.png")

os.makedirs("random_forest", exist_ok=True)

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
