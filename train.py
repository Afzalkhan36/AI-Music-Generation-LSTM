"""
Training Pipeline for AI Music Generation LSTM Model
Author: Senior Deep Learning Engineer
Description: Loads preprocessed dataset arrays, instantiates the LSTM model
             defined in model.py, sets up training callbacks (Checkpoint,
             EarlyStopping, LR Decay, Logging), trains the model, and serializes
             the training history.
"""

import os
import pickle
import sys
from typing import Tuple
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Import reusable configuration loading and model creation functions from model.py
from model import create_model, load_config

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
# Define directory paths and configuration paths
SAVED_MODEL_DIR: str = "saved_model"
X_NPY_PATH: str = os.path.join(SAVED_MODEL_DIR, "X.npy")
Y_NPY_PATH: str = os.path.join(SAVED_MODEL_DIR, "y.npy")
CONFIG_PATH: str = os.path.join(SAVED_MODEL_DIR, "config.json")

# Training hyperparameters (professional defaults for music LSTMs)
EPOCHS: int = 3
BATCH_SIZE: int = 64
VALIDATION_SPLIT: float = 0.2

# Output files generated during training
MODEL_SAVE_PATH: str = os.path.join(SAVED_MODEL_DIR, "music_model.keras")
LOG_CSV_PATH: str = os.path.join(SAVED_MODEL_DIR, "training_log.csv")
HISTORY_PKL_PATH: str = os.path.join(SAVED_MODEL_DIR, "history.pkl")


# ==============================================================================
# STEP 4: DATA LOADING
# ==============================================================================
def load_training_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads normalized input sequences (X) and integer encoded targets (y) from disk.
    
    Args:
        x_path (str): Path to the X.npy file.
        y_path (str): Path to the y.npy file.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Input array (X) and Target array (y).
        
    Raises:
        FileNotFoundError: If either npy file is missing.
        ValueError: If files are corrupted or have mismatched shapes.
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Error: Missing training input file '{x_path}'. Please run 'prepare_data.py' first.")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Error: Missing training target file '{y_path}'. Please run 'prepare_data.py' first.")
        
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except Exception as e:
        raise ValueError(f"Error: Failed to read numpy arrays. Files may be corrupted. Details: {e}")
        
    # Validation: Ensure samples match between X and y
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Error: Shape mismatch! X has {X.shape[0]} samples, "
            f"but y has {y.shape[0]} samples."
        )
        
    return X, y


# ==============================================================================
# STEP 6: CALLBACKS CREATION
# ==============================================================================
def get_training_callbacks() -> list:
    """
    Constructs and returns the list of Keras training callbacks.
    
    Returns:
        list: A list of configured Keras Callback objects.
    """
    # 1. ModelCheckpoint: Saves only the best performing model based on validation loss.
    # Saves in Keras native .keras format which preserves architecture, weights, and optimizer state.
    checkpoint = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    
    # 2. EarlyStopping: Prevents overfitting by halting training if val_loss ceases to improve.
    # patience=10: allows the model to train for 10 epochs without progress before stopping.
    # restore_best_weights: ensures the final model weights are rolled back to the best epoch.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )
    
    # 3. ReduceLROnPlateau: Decreases learning rate if validation loss plateaus.
    # factor=0.2: reduces learning rate to 20% of its current value (e.g. 0.001 -> 0.0002).
    # patience=5: waits 5 epochs without improvement before triggering decay.
    # min_lr: safety floor to prevent learning rate from decaying to absolute zero.
    lr_reduction = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        mode="min",
        min_lr=0.00001,
        verbose=1
    )
    
    # 4. CSVLogger: Streams training logs (loss, accuracy, epoch index) directly into a CSV file.
    # Prevents loss of metrics in case the terminal scrollback is cleared.
    csv_logger = CSVLogger(
        filename=LOG_CSV_PATH,
        separator=",",
        append=False
    )
    
    return [checkpoint, early_stop, lr_reduction, csv_logger]


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def train_pipeline() -> None:
    """
    Orchestrates the entire model training pipeline.
    """
    print("=========================================")
    print("      AI Music LSTM Training Pipeline     ")
    print("=========================================")
    
    # Step 3: Load Configuration
    try:
        sequence_length, vocabulary_size = load_config(CONFIG_PATH)
        print("[STATUS] Model configuration loaded successfully.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Step 4: Load Prepared Arrays
    try:
        X, y = load_training_data(X_NPY_PATH, Y_NPY_PATH)
        print(f"[STATUS] Datasets loaded successfully:")
        print(f"  - Input shape (X) : {X.shape} (Samples, Seq_Len, Features)")
        print(f"  - Output shape (y): {y.shape} (Integer Labels)")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Step 5: Instantiating the Model Architecture
    print("\n[STATUS] Instantiating neural network model...")
    try:
        model = create_model(sequence_length, vocabulary_size)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to construct model: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Configure callbacks
    callbacks = get_training_callbacks()
    
    # Step 7: Model Training Loop
    print(f"\n[STATUS] Starting training for {EPOCHS} epochs with a batch size of {BATCH_SIZE}...")
    try:
        # Train model and store history mapping
        history = model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        print("[STATUS] Training successfully completed!")
        
    except KeyboardInterrupt:
        # Gracefully capture user interrupt (Ctrl+C) and prevent crashing
        print("\n[WARNING] Training interrupted by user. Cleaning up and saving progress...")
        history = None
        
    # Step 8: Save History Object
    if history is not None:
        print(f"[STATUS] Saving training history dictionary to '{HISTORY_PKL_PATH}'...")
        try:
            os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
            with open(HISTORY_PKL_PATH, "wb") as f:
                pickle.dump(history.history, f)
            print("[STATUS] History saved successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to save history pickle file: {e}", file=sys.stderr)
            
    # Step 9: Display Training Session Summary
    print("\n================== TRAINING SUMMARY ==================")
    print("Status: COMPLETED" if history is not None else "Status: INTERRUPTED")
    print(f"Best Model File Saved : {MODEL_SAVE_PATH}")
    print(f"Training Log CSV Saved: {LOG_CSV_PATH}")
    if history is not None:
        print(f"History File Saved    : {HISTORY_PKL_PATH}")
        completed_epochs = len(history.history["loss"])
        final_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        final_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        print(f"Total Epochs Run      : {completed_epochs}")
        print(f"Final Training Loss   : {final_loss:.4f}")
        print(f"Final Validation Loss : {final_val_loss:.4f}")
        print(f"Final Training Acc    : {final_acc:.4f}")
        print(f"Final Validation Acc  : {final_val_acc:.4f}")
    else:
        print("Training was interrupted. Metrics display is unavailable.")
    print("======================================================")


if __name__ == "__main__":
    train_pipeline()