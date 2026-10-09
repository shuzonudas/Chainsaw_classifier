import sys
from pathlib import Path

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def main():
    # 1. Prompt the user for the folder containing audio files
    # folder_input = input("Please enter the folder path containing the audio files: ").strip()
    # We will use the /app/audio path and set this path in docker
    folder_input = "/app/audio"
    if not folder_input:
        print("Error: No folder path entered.")
        sys.exit(1)
        
    input_folder = Path(folder_input)
    
    # 2. Check if the folder exists
    if not input_folder.exists():
        print(f"Error: The folder '{folder_input}' does not exist.")
        sys.exit(1)
        
    if not input_folder.is_dir():
        print(f"Error: '{folder_input}' is not a directory.")
        sys.exit(1)
        
    print(f"Folder '{input_folder}' found. Loading ML packages...")
    
    # 3. Import ML dependencies here to keep the folder check responsive/instant
    try:
        import numpy as np
        import pandas as pd
        import librosa
        import tensorflow as tf
        # tf.get_logger().setLevel("ERROR")
        from tensorflow import keras
    except ImportError as e:
        print(f"Error importing required ML libraries: {e}")
        print("Please make sure numpy, pandas, librosa, and tensorflow are installed in your environment.")
        sys.exit(1)
        
    print(f"TensorFlow version: {tf.__version__}")
    
    # Constants
    SAMPLE_RATE = 16000
    
    def convert_to_MFCC(audio_data) -> pd.DataFrame:
        """Compute MFCCs (kept for compatibility with original notebook)."""
        mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=30)
        return mfccs
        
    class ClassifierEngine:
        """Wrapper for ML model inference."""

        def __init__(self, model_path: str):
            self.model = keras.models.load_model(model_path)

        def classify(self, features: np.ndarray) -> tuple[str, float]:
            """Runs inference on features and returns (label, confidence)."""
            # Reshape for model (batch, n_mfcc, time, channel)
            input_data = features[np.newaxis, ..., np.newaxis]
            predictions = self.model.predict(input_data, verbose=0)
            
            chainsaw_prob = predictions[0, 0]
            label = "Chainsaw Detected" if chainsaw_prob > 0.6 else "No Chainsaw"
            return label, round(float(chainsaw_prob), 2)
            
    # Load model from the expected path
    model_path = "ML_model/ML_MFCC_V5.keras"
    script_dir = Path(__file__).resolve().parent
    resolved_model_path = script_dir / "ML_model" / "ML_MFCC_V5.keras"
    
    print(f"Attempting to load model from: {resolved_model_path}")
    
    engine = None
    try:
        if resolved_model_path.exists():
            engine = ClassifierEngine(str(resolved_model_path))
        else:
            # Fall back to relative path if directory structure differs
            engine = ClassifierEngine(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to initialize system (could not load model): {e}")
        print(f"Please ensure your model file is saved at: {resolved_model_path}")
        sys.exit(1)
        
    def classify_audio_file(file_name, audio):
        chunk_duration = 3  # seconds
        hop_duration = 1    # seconds
        
        chunk_samples = chunk_duration * SAMPLE_RATE
        hop_samples = hop_duration * SAMPLE_RATE
        
        audio_info = []
        
        for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
            end = start + chunk_samples
            mfcc_feature = librosa.feature.mfcc(y=audio[start:end], sr=SAMPLE_RATE, n_mfcc=30)
            label, confidence = engine.classify(mfcc_feature)
            if label == "Chainsaw Detected":
                audio_info.append((
                    file_name,
                    start / SAMPLE_RATE,  # start time
                    end / SAMPLE_RATE,    # end time
                    # label,
                    confidence
                ))
        return audio_info
        
    def get_audio_files(directory):
        AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus", ".aiff", ".aif"}
        directory = Path(directory)
        if directory.exists():
            print("Audio source directory verified.")
        else:
            print("Audio source directory does not exist.")
            return None

        audio_files = []
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            audio_files.append(str(file_path))

        return audio_files

    # 4. Find all audio files in the folder
    audio_files = get_audio_files(input_folder)
    if not audio_files:
        print("No supported audio files found to process.")
        return
        
    print(f"Found {len(audio_files)} audio file(s) to process.")
    all_audio_info = []
    
    # 5. Process files and run inference
    for file in audio_files:
        file_name = Path(file).name
        try:
            audio, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)
            print(f"Currently processing: {file_name}")
            audio_info = classify_audio_file(file_name, audio)
            all_audio_info.extend(audio_info)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            
    if not all_audio_info:
        print("No predictions generated.")
        return
        
    # 6. Save results to CSV
    df = pd.DataFrame(
        all_audio_info,
        columns=[
            "file_name",
            "start_sec",
            "end_sec",
            # "label",
            "confidence"
        ]
    )

    print("\nProcessed Data Sample:")
    print(df.head())
    
    # output_csv = script_dir / "audio_classification.csv"
    output_csv = input_folder / "audio_classification.csv"
    try:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved classification results successfully to: {output_csv}")
    except Exception as e:
        print("\nFailed to save file:", e)

if __name__ == "__main__":
    main()
