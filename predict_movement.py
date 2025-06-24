from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/"

def predict_sentiment(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

def main():
    test_texts = [
        "The company posted a strong profit for the quarter.",
        "Revenue dropped significantly compared to last year."
    ]
    results = predict_sentiment(test_texts)
    for text, label in zip(test_texts, results):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"{text} => {sentiment}")

if __name__ == "__main__":
    main()
