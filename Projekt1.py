import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# 1. Načítanie dát
df = pd.read_csv("financial_news_events.csv")   # uisti sa, že súbor je v rovnakom priečinku
print("Počet riadkov a stĺpcov:", df.shape)
print("Stĺpce:", df.columns)

# 2. Príprava textu
if "Content" in df.columns:
    df["text"] = df["Headline"].fillna("") + " " + df["Content"].fillna("")
else:
    df["text"] = df["Headline"].fillna("")

df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace("[^a-z ]", "", regex=True)

# 3. Počet slov
df["word_count"] = df["text"].str.split().apply(len)

plt.figure(figsize=(6,4))
plt.hist(df["word_count"], bins=40, color="skyblue", edgecolor="black")
plt.title("Histogram počtu slov v článkoch")
plt.xlabel("Počet slov")
plt.ylabel("Frekvencia")
plt.tight_layout()
plt.savefig("histogram_word_count.png")   # obrázok uložený do súboru
plt.show()

# 4. WordCloud
all_text = " ".join(df["text"].dropna())
wc = WordCloud(width=800, height=400, max_words=100, background_color="white").generate(all_text)

plt.figure(figsize=(8,4))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Najčastejšie slová vo finančných správach")
plt.tight_layout()
plt.savefig("wordcloud.png")
plt.show()

# 5. Sentiment analýza
df["sentiment"] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
df["sentiment_category"] = pd.cut(df["sentiment"], bins=[-1,-0.1,0.1,1], labels=["neg","neutral","pos"])

sentiment_counts = df["sentiment_category"].value_counts()

plt.figure(figsize=(5,4))
sentiment_counts.plot(kind="bar", color=["red","gray","green"])
plt.title("Počet článkov podľa sentimentu")
plt.xlabel("Sentiment")
plt.ylabel("Počet článkov")
plt.tight_layout()
plt.savefig("sentiment_bar.png")
plt.show()

print("Výsledky sentimentu:", sentiment_counts.to_dict())
