import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

# 1. Load your feedback data
df = pd.read_csv("C:\\Users\\apugh\\Desktop\\data\\feedback.csv")
# Assume feedback column name - modify if needed
column_name = "We're delighted to hear that you had a great time with us! What made your experience special?"
feedback = df[column_name].dropna().astype(str)

# 2. Initialize tools
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# 3. Data structure to hold word polarity counts
# Example: word_polarity_counts['great'] = {'pos': 5, 'neu': 1, 'neg': 0}
word_polarity_counts = defaultdict(lambda: {'pos': 0, 'neu': 0, 'neg': 0})

# 4. Process each feedback entry
for review in feedback:
    # Tokenize sentence(s)
    sentences = nltk.sent_tokenize(review)
    for sentence in sentences:
        # Get sentiment scores
        scores = sia.polarity_scores(sentence)
        # Classify based on compound score thresholds
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'pos'
        elif compound <= -0.05:
            sentiment = 'neg'
        else:
            sentiment = 'neu'

        # Tokenize words, clean them
        words = nltk.word_tokenize(sentence.lower())
        filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]

        # Update counts per sentiment
        for w in filtered_words:
            word_polarity_counts[w][sentiment] += 1

# 5. Summarize: Total mentions per word = sum of pos+neu+neg counts
total_counts = {w: sum(counts.values()) for w, counts in word_polarity_counts.items()}

# 6. Get top 20 words by total mentions
top_20_words = Counter(total_counts).most_common(20)

# 7. Display the results in a clean output
print(f"{'Word':15} {'Total':>6} {'Positive':>8} {'Neutral':>8} {'Negative':>8}")
print("-" * 50)
for word, total in top_20_words:
    counts = word_polarity_counts[word]
    print(f"{word:15} {total:6} {counts['pos']:8} {counts['neu']:8} {counts['neg']:8}")

# 8. Optional: Visualize top 20 words with polarity stacked bar chart
labels = []
pos_counts = []
neu_counts = []
neg_counts = []

for word, _ in top_20_words:
    labels.append(word)
    pos_counts.append(word_polarity_counts[word]['pos'])
    neu_counts.append(word_polarity_counts[word]['neu'])
    neg_counts.append(word_polarity_counts[word]['neg'])

plt.figure(figsize=(10, 8))
plt.barh(labels, pos_counts, color='green', label='Positive')
plt.barh(labels, neu_counts, left=pos_counts, color='grey', label='Neutral')
# For stacking negative bars, we add positive+neutral counts for left
lefts = [pos_counts[i] + neu_counts[i] for i in range(len(neg_counts))]
plt.barh(labels, neg_counts, left=lefts, color='red', label='Negative')

plt.xlabel('Number of Mentions')
plt.title('Top 20 Words with Polarity Counts')
plt.legend()
plt.tight_layout()
plt.show()
