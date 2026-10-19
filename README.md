# EEG Deep Learning: CNN, LSTM, Hybrid and Siamese Networks

Deep learning models that classify EEG recordings, built for my MSc research in Applied Computer and Information Technology (AI) at Oslo Metropolitan University (2023).

## What's inside

| Notebook | What it does |
|---|---|
| `EEG1.ipynb` | Classifies 19-channel EEG recordings from healthy controls vs. patients. Compares a 1D CNN, a stacked LSTM and a hybrid CNN + LSTM (TensorFlow/Keras). |
| `Deep-learning-Siamnse.ipynb` | Classifies normal vs. abnormal recordings from the NMT scalp EEG dataset with a stacked LSTM, then an LSTM Siamese network trained with triplet loss to learn from very few recordings (PyTorch). |
| `contrastive_loss_keras.ipynb` | Contrastive-loss Siamese network experiments in Keras. |

## Method

1. **Preprocessing (MNE):** load EDF files, set the average reference, band-pass filter at 1–45 Hz, and cut the signal into fixed-length 25-second epochs.
2. **Scaling:** standardize each channel and fit the scaler on training data only.
3. **Models:** 1D CNN, stacked LSTM, hybrid CNN + LSTM, and an LSTM Siamese network.
4. **Evaluation:** subject-wise cross-validation with `GroupKFold`, so a person's recordings never appear in both training and validation. Reports accuracy, precision, recall and F1.

## Tech

Python · PyTorch · TensorFlow/Keras · MNE · scikit-learn · NumPy · Matplotlib

## Run it

```bash
pip install mne torch tensorflow scikit-learn numpy matplotlib tqdm
jupyter notebook
```

The datasets are not included. Download the EEG datasets and update the file paths at the top of each notebook.

## Related

- [Fall detection with wearable sensors](https://github.com/beck2127/fall-detection-wearable-sensors)
