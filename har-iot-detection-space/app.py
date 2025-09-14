import gradio as gr
import pickle
from scipy.io import arff
import numpy as np
import pandas as pd
import os

# Load label codes
def load_codes(code_path):
    code_map = {}
    with open(code_path, 'r') as f:
        for line in f:
            if ':' in line:
                code, label = line.strip().split(':', 1)
                code_map[int(code.strip())] = label.strip()
    return code_map

# Load model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

code_path = os.path.join(os.path.dirname(__file__), '../random_forest/code.txt')
model_path = os.path.join(os.path.dirname(__file__), '../random_forest/model_phone-accel.pkl')
code_map = load_codes(code_path)
model = load_model(model_path)

def predict_arff(arff_file):
    data, meta = arff.loadarff(arff_file.name)
    df = pd.DataFrame(data)
    # Remove label column if present
    print(df.columns)
    feature_cols = [col for col in df.columns if col.lower() not in ['activity', 'subject_id']]
    print(feature_cols)
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    preds = model.predict(X)
    mapped = [code_map.get(int(p), str(p)) for p in preds]
    df_result = pd.DataFrame({'Prediction': mapped})
    return df_result

demo = gr.Interface(
    fn=predict_arff,
    inputs=gr.File(label="Upload ARFF file"),
    outputs=gr.Dataframe(label="Predicted Activities"),
    title="HAR IoT Random Forest Predictor",
    description="Upload an ARFF file to get activity predictions using the trained Random Forest model."
)

if __name__ == "__main__":
    demo.launch()
