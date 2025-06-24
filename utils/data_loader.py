import pandas as pd

def load_sentences(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Sentence', 'Sentiment'], inplace=True)
    df = df[['Sentence', 'Sentiment']]
    df.rename(columns={'Sentence': 'headline', 'Sentiment': 'label'}, inplace=True)
    return df
