# Text Emotion Classifier

Classify emotions from text using a fine-tuned DistilBERT model. This repository includes data preprocessing, model training, evaluation, and deployment via FastAPI.

## Project Structure

```
TEXT_EMOTION_CLASSIFIER
├── dataset
│   ├── sparse_tfidf
│   ├── cleaned_go_emotions_data.csv
│   ├── go_emotions_dataset.csv
│   └── preprocessed_go_emotions_dataset.csv
├── front_end
│   └── main_page.html
├── model
│   ├── Hyperparameter_tuning
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
├── notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_BERT_modeling.ipynb
│   ├── 03_explanations.ipynb
│   ├── 04_evaluation.ipynb
│   └── preprocessed_go_emotions_dataset.csv
├── src
│   ├── __pycache__
│   └── main.py
├── Dockerfile
├── README.md
└── requirements.txt
```

## Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/SydAt1/Text_Emotion_Classifier.git
   cd Text_Emotion_Classifier
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **(Optional) Build and run with Docker:**
   ```
   docker build -t text-emotion-classifier .
   docker run -p 8000:8000 text-emotion-classifier
   ```

## Usage

- Use `notebooks/01_data_preparation.ipynb` for data cleaning and exploration.
- Fine-tune DistilBERT with `notebooks/02_BERT_Model.ipynb`.
- Evaluate with `notebooks/03_evaluation.ipynb`.
- Serve predictions via FastAPI using `app.py` or Docker.

## API

After starting FastAPI, access the API at:  
`http://localhost:8000`

## Notes
- Update file paths in notebooks/scripts if your data location changes.
