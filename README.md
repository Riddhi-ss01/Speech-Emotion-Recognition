# Speech-Emotion-Recognition
# 🎵 Speech and Song Emotion Classification using Deep Features and Neural Networks

This project focuses on classifying emotions from audio files containing **speech** data. It leverages deep audio feature extraction using **Librosa**, a **CNN-BiLSTM-Attention** based neural network model, and various regularization techniques to improve generalization and prevent overfitting.

---

## 📂 Dataset Structure

The dataset consists of two main categories:
- **Speech Data:** `Audio_Speech_Actors_01-24`
- **Song Data:** `Audio_Song_Actors_01-24`

Each file is labeled with emotion codes which are mapped to the following emotions:
- `01`: Neutral
- `02`: Calm
- `03`: Happy
- `04`: Sad
- `05`: Angry
- `06`: Fearful
- `07`: Disgust
- `08`: Surprised

---

## ⚙️ Project Workflow

1. **Mount Google Drive:**
   - The dataset is loaded directly from Google Drive.

2. **Feature Extraction:**
   - MFCCs, Delta, Delta2
   - Chroma, Mel Spectrogram
   - Spectral Contrast, Tonnetz
   - Zero Crossing Rate, RMS Energy
   - All features are stacked and padded to a uniform length.

3. **Data Processing:**
   - Label encoding
   - Stratified Train-Test split considering both emotion and modality (speech/song).

4. **Model Architecture:**
   - 1D CNN block with L2 regularization and dropout
   - BiLSTM layer with recurrent dropout
   - Attention mechanism
   - Dense layers with L2 regularization and dropout
   - Softmax output for multi-class classification

5. **Training:**
   - EarlyStopping with `val_accuracy` monitoring
   - Epochs: 80
   - Batch Size: 32

---

## 📈 Model Performance

- **Test Accuracy:** 82.08%
- **Macro F1 Score:** 82.46%

### Classification Report
| Emotion    | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Angry      | 1.00      | 0.79   | 0.88     |
| Calm       | 0.89      | 0.91   | 0.90     |
| Disgust    | 0.81      | 0.89   | 0.85     |
| Fearful    | 0.68      | 0.82   | 0.74     |
| Happy      | 0.83      | 0.88   | 0.85     |
| Neutral    | 0.85      | 0.92   | 0.88     |
| Sad        | 0.81      | 0.59   | 0.68     |
| Surprised  | 0.73      | 0.90   | 0.80     |

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## 🛠️ Technologies Used
- Python
- Google Colab
- TensorFlow / Keras
- Librosa
- Scikit-learn
- Matplotlib / Seaborn
- NumPy / Pandas

---

## 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/speech-song-emotion-classification.git
cd speech-song-emotion-classification

# Install required packages
pip install -r requirements.txt
