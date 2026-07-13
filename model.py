"""
LSTM Model Architecture for AI Music Generation
Author: Senior Deep Learning Engineer
Description: This script defines the LSTM network architecture for learning
             musical patterns, compiles it with the Adam optimizer, and prints
             a summary. It reads configuration parameters directly from config.json.
"""

import os
import json
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# ==============================================================================
# STEP 4: LOAD CONFIGURATION
# ==============================================================================
# Define config path constant.
SAVED_MODEL_DIR: str = "saved_model"
CONFIG_PATH: str = os.path.join(SAVED_MODEL_DIR, "config.json")

def load_config(config_path: str = CONFIG_PATH) -> tuple[int, int]:
    """
    Loads model configuration parameters from a JSON file.
    
    Args:
        config_path (str): Path to the config.json file.
        
    Returns:
        tuple[int, int]: A tuple containing (sequence_length, vocabulary_size).
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        KeyError: If required keys are missing in the JSON data.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Error: The configuration file '{config_path}' was not found. "
            "Please run 'prepare_data.py' first to generate the configuration and training files."
        )
        
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            msg=f"Error: The file '{config_path}' contains invalid JSON. Details: {e}",
            doc=e.doc,
            pos=e.pos
        )
        
    # Check for required keys in the JSON config
    if "sequence_length" not in config or "vocabulary_size" not in config:
        raise KeyError(
            f"Error: Config file '{config_path}' is missing required keys: "
            "'sequence_length' or 'vocabulary_size'."
        )
        
    return config["sequence_length"], config["vocabulary_size"]


# ==============================================================================
# STEPS 5 TO 14: MODEL ARCHITECTURE CREATION
# ==============================================================================
def create_model(sequence_length: int, vocabulary_size: int) -> Sequential:
    """
    Constructs, compiles, and returns the sequential LSTM model.
    
    Args:
        sequence_length (int): The number of historical notes the model reviews.
        vocabulary_size (int): The total number of unique notes and chords.
        
    Returns:
        Sequential: A compiled Keras Sequential model.
    """
    model = Sequential()
    
    # Step 6: Input Layer
    # Defines the expected input shape for 3D tensors: (samples, time_steps, features).
    model.add(Input(shape=(sequence_length, 1)))
    
    # Step 7: First LSTM Layer
    # 256 memory cells. return_sequences=True is required to stack another LSTM layer.
    model.add(LSTM(256, return_sequences=True))
    
    # Step 8: First Dropout Layer
    # Prevents overfitting by randomly setting 30% of inputs to 0.
    model.add(Dropout(0.3))
    
    # Step 9: Second LSTM Layer
    # 256 memory cells. return_sequences=False because the next layer is Dense.
    model.add(LSTM(256, return_sequences=False))
    
    # Step 10: Second Dropout Layer
    model.add(Dropout(0.3))
    
    # Step 11: Dense Output Layer
    # Dense layer mapped to vocab size. Softmax outputs a probability distribution.
    model.add(Dense(vocabulary_size, activation="softmax"))
    
    # Step 12: Compile Model
    # Uses CategoricalCrossentropy loss and Adam optimizer.
    loss_fn = SparseCategoricalCrossentropy()
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    
    return model


def main() -> None:
    """
    Main runner to verify model architecture compilation.
    """
    print("=========================================")
    print("      AI Music LSTM Model Builder        ")
    print("=========================================")
    
    # Load configuration
    try:
        sequence_length, vocabulary_size = load_config()
        print("[STATUS] Configuration loaded successfully:")
        print(f"  - Sequence Length: {sequence_length}")
        print(f"  - Vocabulary Size: {vocabulary_size}")
    except Exception as e:
        print(f"[CRITICAL ERROR] Configuration loading failed: {e}", file=sys.stderr)
        return
        
    # Build and compile model
    try:
        model = create_model(sequence_length, vocabulary_size)
        print("[STATUS] Model successfully built and compiled.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Model generation failed: {e}", file=sys.stderr)
        return
        
    # Step 13: Model Summary Output
    print("\n=================== MODEL SUMMARY ===================")
    model.summary()
    print("=====================================================")


if __name__ == "__main__":
    main()