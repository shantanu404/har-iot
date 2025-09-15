import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

ACTIVITY_MAP = {
    'A': 'walking', 'B': 'jogging', 'C': 'stairs', 'D': 'sitting', 'E': 'standing',
    'F': 'typing', 'G': 'teeth', 'H': 'soup', 'I': 'chips', 'J': 'pasta',
    'K': 'drinking', 'L': 'sandwich', 'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
    'Q': 'writing', 'R': 'clapping', 'S': 'folding'
}
ACTIVITY_CODE = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17
}
CODE_TO_LABEL = {v: ACTIVITY_MAP[k] for k, v in ACTIVITY_CODE.items()}

class HARCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HARCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=128, kernel_size=11, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'har_cnn_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

def split_df_into_segments(df, time_col='timestamp', threshold=pd.Timedelta('1s')):
    time_diffs = df[time_col].diff().fillna(pd.Timedelta(seconds=0))
    segment_ids = (abs(time_diffs) > threshold).cumsum()
    segments = [group.reset_index(drop=True) for _, group in df.groupby(segment_ids)]

    return segments

def preprocess_csv(csv_file, window_size, step_size):
    df = pd.read_csv(csv_file.name)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')

    # Expect columns: timestamp, x, y, z, activity_code
    segments = split_df_into_segments(df, time_col='timestamp', threshold=pd.Timedelta('1s'))

    windows = []
    window_times = []
    window_segment_ids = []
    for seg_id, segment in enumerate(segments):
        for start in range(0, len(segment) - window_size + 1, step_size):
            window = segment.iloc[start:start + window_size]
            xs = window['x'].values
            ys = window['y'].values
            zs = window['z'].values
            window_np = np.stack([xs, ys, zs], axis=0)  # shape (3, window_size)
            # Get start and end timestamp for the window
            start_dt = window['timestamp'].iloc[0]
            end_dt = window['timestamp'].iloc[-1]
            # Convert to pandas datetime with ns unit
            windows.append(window_np)
            window_times.append((start_dt, end_dt))
            window_segment_ids.append(seg_id)
    if not windows:
        return None, None, None
    features = np.stack(windows, axis=0)  # shape (num_windows, 3, window_size)
    return features, window_times, window_segment_ids

def predict_csv(csv_file):
    window_size = 20
    step_size = 20
    features, window_times, window_segment_ids = preprocess_csv(csv_file, window_size, step_size)
    if features is None:
        return pd.DataFrame({'Error': ['Not enough data for windowing']})
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32).to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = [CODE_TO_LABEL.get(int(p), str(p)) for p in preds]
    # Build DataFrame for windows
    df_windows = pd.DataFrame({
        'Segment ID': window_segment_ids,
        'Start Datetime': [t[0] for t in window_times],
        'End Datetime': [t[1] for t in window_times],
        'Prediction': labels
    })
    # Merge windows of a single segment together
    merged = []
    for seg_id, group in df_windows.groupby('Segment ID'):
        seg_start = group['Start Datetime'].min()
        seg_end = group['End Datetime'].max()
        # Most frequent activity
        activity = group['Prediction'].value_counts().idxmax()
        merged.append({'start': seg_start, 'end': seg_end, 'activity': activity})
    df_result = pd.DataFrame(merged)

    import plotly.graph_objects as go
    fig = go.Figure()
    # Add line for each segment
    for _, row in df_result.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['start'], row['end']],
            y=[row['activity'], row['activity']],
            mode='lines+markers',
            name=str(row['activity']),
            line=dict(width=3),
            marker=dict(size=10),
            hoverinfo='x+y+name'
        ))
    fig.update_layout(
        xaxis_title='Datetime',
        yaxis_title='Activity',
        dragmode='zoom',
        hovermode='closest',
        title='Prediction Timeline'
    )
    return gr.Plot(fig)

demo = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(label="Upload CSV file"),
    outputs=gr.Plot(label="Prediction Timeline"),
    title="HAR IoT CNN Predictor",
    description="Upload a CSV file to get activity predictions using the trained CNN model."
)

if __name__ == "__main__":
    demo.launch()
