"""
Preprocessing Pipeline for AI Music Generation using LSTM
Author: Senior AI Mentor
Description: This script parses MIDI files, extracts notes and chords,
             and serializes the resulting sequence data for training.
"""

import os
import pickle
import sys
from music21 import converter, instrument, note, chord, stream

# ==============================================================================
# STEP 3: CONSTANTS
# ==============================================================================
# Using uppercase for constants is a PEP8 standard.
# Relative paths are used here to keep the project portable.
DATASET_DIR = os.path.join("dataset", "classical")
SAVED_MODEL_DIR = "saved_model"
PICKLE_FILE_PATH = os.path.join(SAVED_MODEL_DIR, "notes.pkl")

# ==============================================================================
# STEP 4: COLLECT MIDI FILES
# ==============================================================================
def get_midi_files(directory):
    """
    Scans the specified directory for MIDI files (.mid or .midi).
    
    Args:
        directory (str): The folder containing raw MIDI files.
        
    Returns:
        list: A list of full file paths to the found MIDI files.
        
    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
        ValueError: If no MIDI files are found.
    """
    # Verify if the folder exists on the disk
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Error: The directory '{directory}' does not exist.")
    
    # Verify if it is a directory and not a file
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Error: The path '{directory}' is not a directory.")
        
    midi_files = []
    
    # os.walk scans the directory tree recursively
    for root, _, files in os.walk(directory):
        for file in files:
            # Check for extensions case-insensitively (.mid and .midi)
            if file.lower().endswith(('.mid', '.midi')):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
                
    # Handle folder containing zero MIDI files
    if not midi_files:
        raise ValueError(f"Error: No MIDI files (.mid or .midi) found in '{directory}'.")
        
    return midi_files

# ==============================================================================
# STEPS 6, 7, 8 & 9: PROCESS A SINGLE MIDI FILE
# ==============================================================================
def extract_notes_and_chords(file_path):
    """
    Parses a MIDI file and extracts its notes and chords.
    
    Args:
        file_path (str): The full path to the MIDI file.
        
    Returns:
        list: A list of extracted note strings (e.g. 'C4') and chords (e.g. '0.4.7').
    """
    # Step 6: Load and parse the MIDI file
    try:
        print(f"Parsing: {os.path.basename(file_path)}...")
        midi_data = converter.parse(file_path)
    except Exception as e:
        # Gracefully handle corrupted files without crashing the entire training pipeline
        print(f"  [ERROR] Failed to parse '{file_path}'. File might be corrupted. Details: {e}", file=sys.stderr)
        return []

    # Step 7: Partition the stream by instrument (Piano, Violin, etc.)
    try:
        parts = instrument.partitionByInstrument(midi_data)
    except Exception as e:
        print(f"  [WARNING] Instrument partitioning failed for '{file_path}': {e}. Using flat stream.")
        parts = None

    # Isolate notes based on track structure
    if parts:
        # If multiple parts exist, take the first one (typically the main piano track)
        notes_to_parse = parts.parts[0].recurse()
    else:
        # Fallback to flattening the entire file if partitioning fails
        notes_to_parse = midi_data.flatten().notes

    extracted_symbols = []
    
    # Loop through musical objects inside the track
    for element in notes_to_parse:
        
        # Step 8: Handle Note objects
        if isinstance(element, note.Note):
            # Extract pitch (e.g., 'C4', 'F#3') as string representation
            extracted_symbols.append(str(element.pitch))
            
        # Step 9: Handle Chord objects
        elif isinstance(element, chord.Chord):
            # Convert chord pitch classes (normalOrder) to a dot-separated string (e.g. '0.4.7')
            chord_rep = '.'.join(str(n) for n in element.normalOrder)
            extracted_symbols.append(chord_rep)

    return extracted_symbols

# ==============================================================================
# STEPS 10 & 11: PIPELINE EXECUTION
# ==============================================================================
def preprocess_dataset():
    """
    The orchestrator function that executes the entire preprocessing pipeline.
    """
    print("=========================================")
    print("  AI Music Preprocessing Pipeline Start  ")
    print("=========================================")
    
    # 1. Gather all raw data files
    try:
        midi_files = get_midi_files(DATASET_DIR)
        print(f"[STATUS] Found {len(midi_files)} MIDI files in '{DATASET_DIR}'.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Preprocessing halted: {e}")
        return

    all_extracted_symbols = []
    
    # 2. Extract notes and chords file-by-file
    for i, file_path in enumerate(midi_files):
        print(f"[{i+1}/{len(midi_files)}] ", end="")
        symbols = extract_notes_and_chords(file_path)
        
        # Count chords (contains dot) and notes (does not contain dot)
        chord_count = sum(1 for sym in symbols if '.' in sym)
        note_count = len(symbols) - chord_count
        
        print(f"Extracted {len(symbols)} elements ({note_count} notes, {chord_count} chords).")
        all_extracted_symbols.extend(symbols)

    # Step 10: Show summary statistics
    print("\n========================= SUMMARY =========================")
    total_symbols = len(all_extracted_symbols)
    total_chords = sum(1 for sym in all_extracted_symbols if '.' in sym)
    total_notes = total_symbols - total_chords
    print(f"Total Notes Extracted  : {total_notes}")
    print(f"Total Chords Extracted : {total_chords}")
    print(f"Total Extracted Symbols: {total_symbols}")
    print("===========================================================")
    
    if total_symbols == 0:
        print("[ERROR] Preprocessing failed: No musical symbols were extracted.")
        return

    # Step 11: Serialize data to saved_model/notes.pkl
    print(f"\n[STATUS] Saving dataset to '{PICKLE_FILE_PATH}'...")
    try:
        # Create output folder if it doesn't exist
        os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        
        # Open the file in write-binary mode
        with open(PICKLE_FILE_PATH, 'wb') as filepath:
            pickle.dump(all_extracted_symbols, filepath)
            
        print("[STATUS] Preprocessing successfully completed & saved!")
    except Exception as e:
        print(f"[ERROR] Failed to save pickle file: {e}")

if __name__ == "__main__":
    preprocess_dataset()