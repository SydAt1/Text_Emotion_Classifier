# Text Emotion Classifier

This project aims to classify emotions from text using the DistilBERT model. The model is fine-tuned on a dataset of text samples labeled with various emotions.

## Project Structure

```
Text_Emotion_Classifier
├── src
│   ├── data
│   │   └── preprocess.py       # Functions for loading and preprocessing text data
│   ├── models
│   │   └── distilbert_emotion.py # Class for DistilBERT emotion classification
│   ├── train.py                 # Script to train the emotion classification model
│   ├── evaluate.py              # Functions to evaluate model performance
│   └── utils.py                 # Utility functions for the project
├── 01_data_preparation.ipynb    # Jupyter notebook for data preparation and EDA
├── 02_BERT_Model.ipynb          # Jupyter notebook for fine-tuning DistilBERT
├── requirements.txt              # List of dependencies
└── README.md                     # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Text_Emotion_Classifier
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Use `01_data_preparation.ipynb` for initial data loading and exploratory data analysis.
- Fine-tune the DistilBERT model using `02_BERT_Model.ipynb`.
- Train the model by running `src/train.py`.
- Evaluate the model's performance with `src/evaluate.py`.
