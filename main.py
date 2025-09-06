import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'Budget': np.random.randint(5000000, 100000000, num_movies),
    'EarlyReview': [f'This movie is {np.random.choice(["amazing", "good", "okay", "bad", "terrible"])}.' for _ in range(num_movies)],
    'BoxOfficeGross': np.random.randint(100000, 1000000000, num_movies)
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['SentimentScore'] = df['EarlyReview'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Data Cleaning (Handling potential outliers - example)---
# This is a placeholder.  A real-world application would need more sophisticated outlier detection.
df = df[df['BoxOfficeGross'] < 500000000] #remove extreme outliers
# --- 4. Analysis ---
# Calculate correlation between sentiment and box office gross
correlation = df['SentimentScore'].corr(df['BoxOfficeGross'])
print(f"Correlation between Sentiment Score and Box Office Gross: {correlation}")
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='SentimentScore', y='BoxOfficeGross', data=df)
plt.title('Sentiment Score vs. Box Office Gross')
plt.xlabel('Sentiment Score')
plt.ylabel('Box Office Gross')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'sentiment_vs_boxoffice.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 6.  Further analysis (example) ---
#  In a real-world scenario, you would build a predictive model here (e.g., linear regression, etc.)
#  using features like SentimentScore and Budget to predict BoxOfficeGross.  This is omitted for brevity.