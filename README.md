# Household Power Consumption Intensity Classification

## Overview

**Dataset:** Individual household electric power consumption dataset from the UCI Machine Learning Repository.

**Objective:** Predict power consumption intensity category (low/medium/high) based on past readings.

**Approach:**

- Data cleaning and preprocessing to handle missing values, format conversions, and remove inconsistent days.
- Feature engineering to categorize intensity into classes.
- Time-series sequence generation for LSTM input.
- LSTM model implemented in PyTorch for multi-class classification.
- Experimentation with different sequence strides and learning rates to optimize training.

---

## Setup and Installation
bash:
pip install ucimlrepo pandas numpy matplotlib torch scikit-learn

# Usage

## Data Loading and Preprocessing

- Fetch dataset using the `ucimlrepo` package.
- Convert date and time columns into datetime objects.
- Clean dataset by handling missing values (`'?'` replaced with `NaN`, forward fill and interpolation).
- Remove incomplete or inconsistent days.
- Categorize `global_intensity` into three classes:
  - Low (0)
  - Medium (1)
  - High (2)

---

## Model

A simple LSTM-based RNN implemented using PyTorch.

**Input features:**

- Global_active_power
- Global_reactive_power
- Voltage
- Global_intensity
- Sub_metering_1
- Sub_metering_2
- Sub_metering_3

**Output:** 3-class classification (intensity class).

---

## Training

- Train-test split: 80/20.
- Time series sequences created with configurable sequence length and stride.
- Loss function: `CrossEntropyLoss`.
- Optimizer: Adam.
- Training loop supports GPU acceleration if available.
- Experimented with different strides and learning rates to optimize performance.

---

## Key Code Sections

- **Data Preprocessing:** Handling missing values, date/time conversion, filtering inconsistent records.
- **Feature Engineering:** Categorizing `global_intensity` into discrete classes.
- **Sequence Creation:** Sliding window approach for time-series samples.
- **Model:** `SimpleRNN` class using PyTorch LSTM + Linear layer.
- **Training & Evaluation:** Loop over epochs with loss printing and test accuracy calculation.
- **Learning Rate Range Test:** Visual tool to pick suitable learning rate.
- **Stride Experiment:** Compare training losses with different temporal strides.

---

## Results

- Training loss steadily decreases with an appropriate stride (e.g., stride=10).
- Step size (stride) affects model learning speed and redundancy in training data.
- Learning rate tuning is critical to avoid plateaus and improve convergence.
- Final test accuracy is 93%

---

## Future Work

- Hyperparameter tuning (hidden size, batch size, learning rate schedules).
- Implement padding and masking for variable-length sequences.
- Explore alternative architectures like GRU or Transformer.
- Extend analysis with feature importance or explainability methods.

---

## References

- UCI Machine Learning Repository: Individual household electric power consumption dataset  
  https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

- PyTorch LSTM documentation:  
  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

---

