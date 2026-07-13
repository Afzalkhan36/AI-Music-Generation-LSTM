"""
Inference Pipeline for AI Music Generation using LSTM
Author: Senior AI Engineer
Description: Loads a trained Keras LSTM model, configures mapping files, selects
             a random seed sequence from training data, predicts a sequence of
             new notes/chords using temperature-based sampling, and converts
             the sequence into a playable MIDI file.
"""

import os
import pickle
import json
import random
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from music21 import stream, note, chord, instrument

# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================
SAVED_MODEL_DIR: str = "saved_model"
GENERATED_MUSIC_DIR: str = "generated_music"

CONFIG_PATH: str = os.path.join(SAVED_MODEL_DIR, "config.json")
MODEL_PATH: str = os.path.join(SAVED_MODEL_DIR, "music_model.keras")
NOTE_TO_INT_PATH: str = os.path.join(SAVED_MODEL_DIR, "note_to_int.pkl")
INT_TO_NOTE_PATH: str = os.path.join(SAVED_MODEL_DIR, "int_to_note.pkl")
X_NPY_PATH: str = os.path.join(SAVED_MODEL_DIR, "X.npy")

# Inference hyperparameters
GENERATION_LENGTH: int = 500  # Number of notes/chords to generate
TEMPERATURE: float = 0.8     # Controls randomness (creativity vs coherence)


# ==============================================================================
# STEP 3: RESOURCE LOADING
# ==============================================================================
def load_inference_resources() -> Tuple[dict, Dict[str, int], Dict[int, str], tf.keras.Model, np.ndarray]:
    """
    Loads all required configuration files, dictionary maps, trained model, and input data.
    
    Returns:
        Tuple[dict, Dict[str, int], Dict[int, str], tf.keras.Model, np.ndarray]:
            - config: Project configuration dictionary.
            - note_to_int: String to integer mapping dictionary.
            - int_to_note: Integer to string mapping dictionary.
            - model: The loaded compiled Keras model.
            - X: Input sequences numpy array.
        
    Raises:
        FileNotFoundError: If any of the files are missing.
        RuntimeError: If files are corrupted or fail to load.
    """
    # 1. Check file existence
    paths = {
        "config": CONFIG_PATH,
        "note_to_int": NOTE_TO_INT_PATH,
        "int_to_note": INT_TO_NOTE_PATH,
        "model": MODEL_PATH,
        "X_data": X_NPY_PATH
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Error: Missing required resource '{name}' at path '{path}'. "
                "Ensure preprocess.py, prepare_data.py, and train.py have run successfully."
            )
            
    # 2. Load JSON configuration
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error: Failed to parse configuration JSON: {e}")
        
    # 3. Load dictionary mapping pickles
    try:
        with open(NOTE_TO_INT_PATH, "rb") as f:
            note_to_int = pickle.load(f)
        with open(INT_TO_NOTE_PATH, "rb") as f:
            int_to_note = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load mapping pickle files: {e}")
        
    # 4. Load training sequences array
    try:
        X = np.load(X_NPY_PATH)
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load input sequences (X.npy): {e}")
        
    # 5. Load Keras model
    try:
        print("[STATUS] Loading trained neural network model...")
        model = load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error: Failed to load trained Keras model: {e}")
        
    return config, note_to_int, int_to_note, model, X


# ==============================================================================
# STEP 5: TEMPERATURE SAMPLING
# ==============================================================================
def sample_with_temperature(probabilities: np.ndarray, temperature: float) -> int:
    """
    Applies temperature scaling to a probability distribution and samples an index.
    
    Args:
        probabilities (np.ndarray): The Softmax outputs from the model (shape: (vocab_size,)).
        temperature (float): The randomness factor. 
                             Values closer to 0.0 make predictions deterministic.
                             Values > 1.0 make predictions highly creative but chaotic.
                             
    Returns:
        int: The selected note/chord index.
    """
    # Safeguard: if temperature is extremely low, fall back to simple argmax (greedy search)
    if temperature <= 0.01:
        return int(np.argmax(probabilities))
        
    # Take logarithm of probabilities (convert back to log-space / logits)
    # We add a tiny epsilon (1e-7) to prevent taking log of zero.
    logits = np.log(probabilities + 1e-7)
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Re-apply Softmax to obtain a modified probability distribution
    exp_logits = np.exp(scaled_logits)
    scaled_probs = exp_logits / np.sum(exp_logits)
    
    # Randomly sample an index based on the scaled probability distribution
    choices = range(len(scaled_probs))
    return int(np.random.choice(choices, p=scaled_probs))


# ==============================================================================
# STEP 8: CREATE MIDI STREAM
# ==============================================================================
def convert_to_midi_stream(prediction_output: List[str]) -> stream.Stream:
    """
    Converts a sequence of note and chord string representations back into a 
    music21 Stream object.
    
    Args:
        prediction_output (List[str]): List of predicted note names ('C4') and chords ('0.4.7').
        
    Returns:
        stream.Stream: The populated music21 Stream containing notes and chords.
    """
    midi_stream = stream.Stream()
    # Add a Piano instrument track
    midi_stream.append(instrument.Piano())
    
    offset = 0.0  # Tracks timing offset (beats) of notes
    
    for pattern in prediction_output:
        # Pattern represents a Chord (notes separated by dots)
        if ('.' in pattern) or pattern.isdigit():
            # Extract note components of the chord
            chord_notes = pattern.split('.')
            notes_in_chord = []
            
            for current_note in chord_notes:
                # Convert pitch class integer back to a Note object
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes_in_chord.append(new_note)
                
            # Create a Chord object containing the note list
            new_chord = chord.Chord(notes_in_chord)
            new_chord.offset = offset
            midi_stream.append(new_chord)
            
        # Pattern represents a single Note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            midi_stream.append(new_note)
            
        # Increment offset to prevent notes from stacking. 
        # 0.5 offset represents an eighth note duration. This creates steady rhythm.
        offset += 0.5
        
    return midi_stream


# ==============================================================================
# ORCHESTRATION PIPELINE
# ==============================================================================
def run_generation_pipeline() -> None:
    """
    Main execution pipeline for generating music from the trained model.
    """
    print("=========================================")
    print("     AI Music Generation Inference       ")
    print("=========================================")
    
    # 1. Load resources
    try:
        config, note_to_int, int_to_note, model, X = load_inference_resources()
        print("[STATUS] Configuration and model files loaded successfully.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load resources: {e}", file=sys.stderr)
        return
        
    sequence_length = config["sequence_length"]
    vocab_size = config["vocabulary_size"]
    
    # 2. Step 4: Pick a random seed sequence from X
    print("[STATUS] Selecting seed sequence from dataset...")
    # Select a random sample index from our inputs
    start_idx = random.randint(0, len(X) - 1)
    seed_sequence = X[start_idx]
    
    # Flatten the selected seed sequence (which has shape (sequence_length, 1))
    # into a flat list of normalized floats.
    pattern = list(seed_sequence.flatten())
    
    generated_notes: List[str] = []
    
    # 3. Step 6: Inference prediction loop
    print(f"[STATUS] Generating {GENERATION_LENGTH} notes using Temperature = {TEMPERATURE}...")
    for step in range(GENERATION_LENGTH):
        # Reshape pattern to input tensor shape (1, sequence_length, 1)
        model_input = np.reshape(pattern, (1, len(pattern), 1))
        
        # Predict probability distribution over vocab
        prediction = model.predict(model_input, verbose=0)[0]
        
        # Sample an index using temperature
        predicted_idx = sample_with_temperature(prediction, TEMPERATURE)
        
        # Step 7: Map predicted index back to note string
        predicted_note = int_to_note[predicted_idx]
        generated_notes.append(predicted_note)
        
        # Normalize the predicted index to feed it back as input
        normalized_val = predicted_idx / float(vocab_size)
        
        # Append the new value to our sequence and slide the window
        pattern.append(normalized_val)
        pattern = pattern[1:]
        
        # Output progress visualizer
        if (step + 1) % 100 == 0:
            print(f"  -> Generated {step + 1}/{GENERATION_LENGTH} symbols.")
            
    # 4. Step 8: Convert predicted notes to a MIDI stream
    print("[STATUS] Generating MIDI stream...")
    midi_stream = convert_to_midi_stream(generated_notes)
    
    # 5. Step 9: Save the MIDI file with a unique timestamp
    os.makedirs(GENERATED_MUSIC_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_filename = f"music_{timestamp}.mid"
    output_path = os.path.join(GENERATED_MUSIC_DIR, output_filename)
    
    try:
        # Write the stream to a MIDI file
        midi_stream.write('midi', fp=output_path)
    except PermissionError:
        print(f"[ERROR] Permission denied when writing to directory '{GENERATED_MUSIC_DIR}'.", file=sys.stderr)
        return
    except Exception as e:
        print(f"[ERROR] Failed to write MIDI file: {e}", file=sys.stderr)
        return
        
    # Step 10: Print final summary
    print("\n================ GENERATION SUMMARY ================")
    print("Status: SUCCESS")
    print(f"Model Loaded          : {MODEL_PATH}")
    print(f"Vocabulary Size       : {vocab_size} unique symbols")
    print(f"Seed Sequence Index   : {start_idx}")
    print(f"Seed Sequence Length  : {sequence_length}")
    print(f"Generated Notes/Chords: {len(generated_notes)}")
    print(f"Saved Output MIDI Path: {output_path}")
    print("====================================================")


if __name__ == "__main__":
    run_generation_pipeline()