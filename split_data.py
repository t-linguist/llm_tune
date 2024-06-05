import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def sep_sents(df):
    return df["en"].tolist(), df["es"].tolist(), df["ru"].tolist()


def prepare_data(en_sents, es_sents, ru_sents):
    data = []
    for en, es, ru in zip(en_sents, es_sents, ru_sents):
        data.append({
            "instruction": "Translate the following text from English to Spanish\n\n",
            "text": f"### English:\n{en}\n\n### Spanish:\n",
            "translation": es
        })
        data.append({
            "instruction": "Translate the following text from English to Russian\n\n",
            "text": f"### English:\n{en}\n\n### Russian:\n",
            "translation": ru
        })
        data.append({
            "instruction": "Translate the following text from Spanish to English\n\n",
            "text": f"### Spanish:\n{es}\n\n### English:\n",
            "translation": en
        })
        data.append({
            "instruction": "Translate the following text from Spanish to Russian\n\n",
            "text": f"### Spanish:\n{es}\n\n### Russian:\n",
            "translation": ru
        })
        data.append({
            "instruction": "Translate the following text from Russian to English\n\n",
            "text": f"### Russian:\n{ru}\n\n### English:\n",
            "translation": en
        })
        data.append({
            "instruction": "Translate the following text from Russian to Spanish\n\n",
            "text": f"### Russian:\n{ru}\n\n### Spanish:\n",
            "translation": es
        })
    return data


full_df = pd.read_excel("data/cs_data.xlsx", usecols=["ru", "en", "es"]).drop_duplicates()
full_df = full_df.replace(u"\xa0", u" ", regex=True)
train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=2024)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

en_train_sents, es_train_sents, ru_train_sents = sep_sents(train_df)
en_test_sents, es_test_sents, ru_test_sents = sep_sents(test_df)

train_data = prepare_data(en_train_sents, es_train_sents, ru_train_sents)
test_data = prepare_data(en_test_sents, es_test_sents, ru_test_sents)

with open("data/train.pkl", "wb") as file1:
    pickle.dump(train_data, file1)
with open("data/test.pkl", "wb") as file2:
    pickle.dump(test_data, file2)
