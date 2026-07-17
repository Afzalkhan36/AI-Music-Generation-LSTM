# 🎵 AI Music Generation using LSTM

An AI-powered music generation system built using **Python, TensorFlow/Keras, LSTM, and Music21**. The model learns musical patterns from MIDI files and generates new piano melodies in MIDI format.

---

# 🚀 Features

- 🎼 Learn musical patterns from MIDI datasets
- 🧠 Deep Learning model using stacked LSTM layers
- 🎹 Generate completely new music
- 🎵 Save generated music as MIDI files
- 📊 Training logs and history tracking
- 💾 Automatic model checkpoint saving
- ⚡ GPU training supported (Google Colab)

---

# 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Music21
- Pickle
- JSON
- Google Colab (GPU Training)

---

# 📂 Project Structure

```
AI-Music-Generation/
│
├── dataset/
│   ├── classical/
│   ├── jazz/
│   └── pop/
│
├── generated_music/
│
├── saved_model/
│   ├── music_model.keras
│   ├── notes.pkl
│   ├── note_to_int.pkl
│   ├── int_to_note.pkl
│   ├── config.json
│   ├── history.pkl
│   └── training_log.csv
│
├── preprocess.py
├── prepare_data.py
├── model.py
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Workflow

```
Dataset
   │
   ▼
preprocess.py
   │
   ▼
prepare_data.py
   │
   ▼
model.py
   │
   ▼
train.py
   │
   ▼
generate.py
   │
   ▼
Generated MIDI Music
```

---

# 📊 Model Architecture

- Input Layer
- LSTM (256 Units)
- Dropout (0.3)
- LSTM (256 Units)
- Dropout (0.3)
- Dense Softmax Output Layer

Optimizer:
- Adam

Loss Function:
- Sparse Categorical Crossentropy

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/your-username/AI-Music-Generation.git
```

Move into project folder

```bash
cd AI-Music-Generation
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

### Step 1

Preprocess MIDI files

```bash
python preprocess.py
```

### Step 2

Prepare dataset

```bash
python prepare_data.py
```

### Step 3

Build model

```bash
python model.py
```

### Step 4

Train model

```bash
python train.py
```

### Step 5

Generate music

```bash
python generate.py
```

Generated music will be saved inside:

```
generated_music/
```

---

# 📁 Dataset

The project is trained using MIDI files collected from:

- Classical Music
- Jazz
- Pop

Dataset should be placed inside:

```
dataset/
```

---

# 📈 Future Improvements

- Bidirectional LSTM
- Attention Mechanism
- Transformer-based Music Generation
- Web Interface (React + FastAPI)
- Music Style Selection
- Live Music Generation
- Temperature & Top-k Sampling
- Piano Roll Visualization

---

# 👨‍💻 Author

**Afzal Khan**

Computer Science Engineering Student

Passionate about Artificial Intelligence, Machine Learning and Deep Learning.

---

# ⭐ If you like this project

Please give this repository a ⭐ on GitHub.
