 # Human Activity Recognition with CNN-LSTM (LRCN)

Deep learning project for **video-based Human Activity Recognition (HAR)**, using a **Long-term Recurrent Convolutional Network (LRCN)** — a hybrid architecture that combines a **CNN** (spatial feature extraction per frame) with an **LSTM** (temporal sequence modeling) — trained on the **UCF50** dataset.

## Overview

The goal of this project is to classify short video clips into human activity categories by learning both the spatial appearance of each frame and the temporal dynamics across frames. Five action classes from UCF50 were used:

- Basketball
- SkateBoarding
- Skiing
- HorseRiding
- RockClimbingIndoor

## Pipeline

1. **Data loading & preprocessing** — videos are read with OpenCV; from each video, a fixed number of frames (`sequence_length = 25`) are extracted at equally spaced intervals, resized to `64x64`, and normalized.
2. **Dataset construction** — frames are grouped per video into sequences, labels are one-hot encoded.
3. **Train/test split** — 80% train / 20% test, stratified with a fixed random seed for reproducibility.
4. **Model — LRCN architecture**:
   - 4 blocks of `TimeDistributed(Conv2D + MaxPooling2D + Dropout)` (16 → 32 → 64 → 64 filters) to extract spatial features independently from each frame in the sequence.
   - `TimeDistributed(Flatten())` to prepare the per-frame feature vectors.
   - `LSTM(64)` to model temporal dependencies across the frame sequence.
   - `Dense(softmax)` output layer for final classification.
5. **Training** — compiled with categorical cross-entropy loss and Adam optimizer, trained for up to 100 epochs (batch size 4, 20% validation split), with `EarlyStopping` (patience = 15, restoring best weights) to prevent overfitting.
6. **Evaluation** — loss/accuracy curves plotted for train vs. validation.
7. **Inference on real-world video** — a YouTube video is downloaded (via `yt-dlp`) and run through the trained model to predict the activity being performed.

## Results

- **Validation accuracy:** ~99% (**validation loss:** ~0.2%)
- **Test set (post-training evaluation) accuracy:** ~85% (loss: ~54%)
- Early stopping halted training around **epoch 51**, avoiding overfitting.
- Successfully demonstrated end-to-end prediction on an external YouTube video (not part of the training/test set).

## Tech Stack

- Python, TensorFlow / Keras
- OpenCV (`cv2`) for video/frame processing
- NumPy, scikit-learn (train/test split)
- Matplotlib (training curves)
- `yt-dlp` for downloading YouTube videos for inference
- Google Colab (GPU runtime) for training

## Repository Contents

- `HAR_CNN_LSTM.ipynb` — full notebook: data preprocessing, model definition, training, evaluation, and video inference.

## Dataset

[UCF50 — Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF50.php). A subset of 5 classes was used for this project (see above).

## How to Run

1. Open the notebook in Google Colab (or a local Jupyter environment with a GPU).
2. Mount your Google Drive and place the UCF50 videos (organized in per-class subfolders) under the dataset directory referenced in the notebook.
3. Run the cells sequentially: preprocessing → model construction → training → evaluation → inference on a custom video.

## Author

**Cheick Mohamed Rachid** ("TheGeek")
MSc in Artificial Intelligence and Intelligent Systems, Gümüşhane University

## License

This project is provided for educational and research purposes.
