"""
Data Preparation Pipeline for AI Music Generation using LSTM
Author: Senior AI Mentor
Description: This script loads the preprocessed notes pickle file, prepares
             vocabulary maps, creates input-output sequences using a sliding
             window, normalizes inputs, one-hot encodes targets, and serializes
             the arrays to disk.
"""

import os
import pickle
import sys
from typing import List, Tuple, Dict
import numpy as np
# from tensorflow.keras.utils import to_categorical
import json

# ==============================================================================
# STEP 3: CONSTANTS
# ==============================================================================
# Constants defined at module-level in UPPERCASE.
# Relative paths are chosen to ensure project portability across filesystems.
SAVED_MODEL_DIR: str = "saved_model"
NOTES_PKL_PATH: str = os.path.join(SAVED_MODEL_DIR, "notes.pkl")
SEQUENCE_LENGTH: int = 100

# File names for output arrays and mapping files
X_NPY_PATH: str = os.path.join(SAVED_MODEL_DIR, "X.npy")
Y_NPY_PATH: str = os.path.join(SAVED_MODEL_DIR, "y.npy")
NOTE_TO_INT_PATH: str = os.path.join(SAVED_MODEL_DIR, "note_to_int.pkl")
INT_TO_NOTE_PATH: str = os.path.join(SAVED_MODEL_DIR, "int_to_note.pkl")
CONFIG_PATH: str = os.path.join(SAVED_MODEL_DIR, "config.json")


# ==============================================================================
# STEP 4: LOAD PREPROCESSED NOTES
# ==============================================================================
def load_notes(file_path: str) -> List[str]:
    """
    Loads preprocessed musical symbols from a serialized pickle file.

    Args:
        file_path (str): Path to the notes.pkl file.

    Returns:
        List[str]: The flat list of notes and chords.

    Raises:
        FileNotFoundError: If notes.pkl does not exist.
        pickle.UnpicklingError: If the pickle file is corrupted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: The preprocessed file '{file_path}' was not found. "
            "Please run 'preprocess.py' first to generate this file."
        )

    try:
        with open(file_path, "rb") as filepath:
            notes = pickle.load(filepath)
    except (pickle.UnpicklingError, EOFError) as e:
        raise pickle.UnpicklingError(
            f"Error: The file '{file_path}' is corrupted and cannot be read. "
            f"Details: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Error: An unexpected error occurred while reading '{file_path}': {e}"
        )

    if not notes:
        raise ValueError(f"Error: The file '{file_path}' is empty.")

    return notes


# ==============================================================================
# STEPS 5 & 6: VOCABULARY AND MAPPING
# ==============================================================================
def create_vocab_mappings(
    notes: List[str],
) -> Tuple[List[str], int, Dict[str, int], Dict[int, str]]:
    """
    Identifies the unique musical vocabulary and constructs bidirectional mappings.

    Args:
        notes (List[str]): List of all note/chord symbols in the dataset.

    Returns:
        Tuple[List[str], int, Dict[str, int], Dict[int, str]]:
            - Sorted list of unique notes.
            - Vocabulary size (total unique notes).
            - Dictionary mapping note strings to integer indices.
            - Dictionary mapping integer indices back to note strings.
    """
    # Extract unique symbols and sort them to ensure deterministic order
    unique_notes = sorted(list(set(notes)))
    vocab_size = len(unique_notes)

    # Dictionary Comprehension: Map strings to indices and vice versa
    note_to_int = {note_sym: i for i, note_sym in enumerate(unique_notes)}
    int_to_note = {i: note_sym for i, note_sym in enumerate(unique_notes)}

    return unique_notes, vocab_size, note_to_int, int_to_note


# ==============================================================================
# STEPS 8, 9, 10 & 11: DATASET GENERATION PIPELINE
# ==============================================================================
def prepare_sequences(
    notes: List[str], note_to_int: Dict[str, int], vocab_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts note strings to integers, constructs input-output sequence windows,
    normalizes the inputs, and one-hot encodes targets.

    Args:
        notes (List[str]): Full sequence of raw notes and chords.
        note_to_int (Dict[str, int]): Note-to-integer dictionary mapping.
        vocab_size (int): Size of the musical vocabulary.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Normalized input sequence array of shape (n_samples, sequence_length, 1).
            - y: One-hot encoded target output array of shape (n_samples, vocab_size).

    Raises:
        ValueError: If dataset size is smaller than the sequence window length.
    """
    if len(notes) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Error: Dataset length ({len(notes)}) is too small for sequence length ({SEQUENCE_LENGTH}). "
            "Please add more MIDI files to your dataset."
        )

    network_input = []
    network_output = []

    # Step 8: Sliding Window Sequence Generation
    for i in range(len(notes) - SEQUENCE_LENGTH):
        # Extract a slice of notes of fixed window size
        sequence_in = notes[i : i + SEQUENCE_LENGTH]
        # Extract the target note immediately following that sequence
        sequence_out = notes[i + SEQUENCE_LENGTH]

        # Convert strings to integers using mapping
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Step 9 & 10: Shape and Normalization of inputs (X)
    # Reshape input to conform to LSTM dimensions: (samples, sequence_length, features)
    X = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))

    # Normalize inputs: map integer classes [0, vocab_size - 1] to floats [0.0, 1.0]
    X_normalized = (X / float(vocab_size)).astype(np.float32)

    # Step 11: Store integer labels instead of one-hot encoding
    y_labels = np.array(network_output, dtype=np.int32)

    return X_normalized, y_labels


# ==============================================================================
# STEP 12 & 13: EXECUTABLE RUNNER
# ==============================================================================
def run_preparation_pipeline() -> None:
    """
    Main orchestrator function for the data preparation pipeline.
    """
    print("=========================================")
    print("  AI Music Data Preparation Pipeline Start")
    print("=========================================")

    # 1. Load the preprocessed file
    try:
        notes = load_notes(NOTES_PKL_PATH)
        print(f"[STATUS] Successfully loaded '{NOTES_PKL_PATH}'.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load preprocessed data: {e}", file=sys.stderr)
        return

    # 2. Extract vocabulary & mapping dictionaries
    unique_notes, vocab_size, note_to_int, int_to_note = create_vocab_mappings(notes)
    print(f"[STATUS] Unique vocabulary size: {vocab_size}")

    # 3. Create training sequences, normalize, and one-hot encode
    try:
        X, y = prepare_sequences(notes, note_to_int, vocab_size)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to prepare sequences: {e}", file=sys.stderr)
        return

    # 4. Save prepared variables
    print(f"\n[STATUS] Writing prepared outputs to '{SAVED_MODEL_DIR}'...")
    try:
        os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

        # Save numpy binary arrays
        np.save(X_NPY_PATH, X)
        np.save(Y_NPY_PATH, y)

        # Save dictionary mappings as pickles
        with open(NOTE_TO_INT_PATH, "wb") as f:
            pickle.dump(note_to_int, f)
        with open(INT_TO_NOTE_PATH, "wb") as f:
            pickle.dump(int_to_note, f)

        # Save project configuration
        config = {
            "sequence_length": SEQUENCE_LENGTH,
            "vocabulary_size": vocab_size,
            "total_notes": len(notes)
        }

        with open(CONFIG_PATH, "w") as file:
            json.dump(config, file, indent=4)    

        print("[STATUS] Data preparation completed successfully!")

    except PermissionError:
        print(
            f"[CRITICAL ERROR] Permission denied when writing to '{SAVED_MODEL_DIR}'. "
            "Please check folder permissions or close any programs locking files.",
            file=sys.stderr
        )
        return
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to save variables: {e}", file=sys.stderr)
        return

    # 5. Preprocessing Summary Display
    print("\n================= PREPROCESSING SUMMARY =================")
    print(f"Total Notes Extracted  : {len(notes)}")
    print(f"Unique Vocabulary Size : {vocab_size}")
    print(f"Sequence Window Length : {SEQUENCE_LENGTH}")
    print(f"Total Sequences Created: {X.shape[0]}")
    print(f"Input Shape (X)        : {X.shape} -> (Samples, Seq_Len, Features)")
    print(f"Output Shape (y)       : {y.shape} -> (Samples, Vocab_Size)")
    print("---------------------------------------------------------")
    print("Saved Files:")
    print(f"  - {X_NPY_PATH} (Input array)")
    print(f"  - {Y_NPY_PATH} (Target array)")
    print(f"  - {NOTE_TO_INT_PATH} (String to Integer map)")
    print(f"  - {INT_TO_NOTE_PATH} (Integer to String map)")
    print("=========================================================")


if __name__ == "__main__":
    run_preparation_pipeline()