# 📈 Financial Sentiment + Stock Movement Prediction

This project aims to **predict financial market sentiment** from news headlines and determine **whether a stock's price will rise or fall** using that sentiment. The model uses FinBERT (a financial-domain BERT model) and historical stock price data from Yahoo Finance.

---

## 🧠 Project Overview

Given a dataset of financial headlines with dates and tickers, this pipeline:

1. Fetches stock price movement after the headline date using `yfinance`.
2. Labels each headline as **positive** or **negative** based on whether the stock price went up or down.
3. Fine-tunes `FinBERT` for sentiment classification.
4. Predicts sentiment of new financial headlines.

---

## 📁 Directory Structure

```
sentiment-analysis-llm/
├── data/
│   └── headlines.csv             # Input data with headline, date, ticker
├── models/
│   └── sentiment_model.pt        # Trained FinBERT model
├── utils/
│   └── data_loader.py            # Data preprocessing and stock movement labeling
├── main.py                       # End-to-end runner
├── train_sentiment.py           # Model training script
├── predict_movement.py          # Prediction script for new headlines
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🧪 Dataset Format

The CSV should look like:

```csv
headline,date,ticker
"Apple releases new product",2024-12-15,AAPL
"Amazon profits decline",2024-12-16,AMZN
```

Each row represents a financial headline and the stock ticker it affects.

---

## ⚙️ How It Works

### 1. Labeling: `utils/data_loader.py`

- Downloads stock price for each `ticker` on the given `date` using `yfinance`.
- If the **next day's closing price** is higher than the opening price, the label is **1 (positive)**, else **0 (negative)**.

### 2. Training: `train_sentiment.py`

- Loads labeled data.
- Tokenizes headlines using FinBERT's tokenizer.
- Trains a binary classifier using HuggingFace's `Trainer` API.
- Saves the model in the `models/` directory.

### 3. Prediction: `predict_movement.py`

- Loads saved FinBERT model.
- Predicts sentiment of custom headline list.

---

## 🚀 Running the Project

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Add Your Data

Place your `headlines.csv` inside the `data/` folder in this format:

```
headline,date,ticker
"Apple shares soar",2024-12-15,AAPL
"Tesla stock falls",2024-12-16,TSLA
```

### Step 3: Train the Model

```bash
python train_sentiment.py
```

### Step 4: Run Predictions

```bash
python predict_movement.py
```

Or run the full pipeline via:

```bash
python main.py
```

---

## 🛠️ Tech Stack

- Python 3.11
- HuggingFace Transformers (`FinBERT` model)
- PyTorch
- Datasets (from HuggingFace)
- yfinance
- scikit-learn, pandas, tqdm, requests, beautifulsoup4

---

## ✨ Example Output

```
Apple shares soar after record iPhone sales => Positive
Tesla stock falls after Elon Musk comments => Negative
```

---

## 📌 TODO

- [ ] Improve model performance with more training data
- [ ] Add more fine-tuning options (learning rate, batch size, etc.)
- [ ] Visualize results and evaluation metrics
- [ ] Deploy as a web app

---

## 👤 Author

**Arnav Kulkarni**  
🔗 [GitHub](https://github.com/arnavkulkarni2005)

---

## 📄 License

This project is licensed under the MIT License.