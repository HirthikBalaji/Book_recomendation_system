import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

file_path = 'bestsellers with categories.csv'
df = pd.read_csv(file_path)

df.fillna('', inplace=True)

df['combined_features'] = (
        df['Name'] + ' ' +
        df['Author'] + ' ' +
        df['Genre']
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Name'].str.lower())

scaler = MinMaxScaler()
df[['norm_rating', 'norm_reviews']] = scaler.fit_transform(
    df[['User Rating', 'Reviews']]
)


def recommend_books(book_title, top_n=10, weight_content=0.7, weight_popularity=0.3):
    """
    Recommends similar books combining content similarity and popularity.

    Parameters:
    - book_title: str
    - top_n: int
    - weight_content: float
    - weight_popularity: float

    Returns:
    - pandas DataFrame of recommendations
    """
    book_title = book_title.lower()

    if book_title not in indices:
        return f"Book '{book_title}' not found in dataset."

    idx = indices[book_title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    recommendations = []
    for i, sim_score in sim_scores:
        popularity_score = (
                df.loc[i, 'norm_rating'] * 0.6 +
                df.loc[i, 'norm_reviews'] * 0.4
        )
        combined_score = (weight_content * sim_score) + (weight_popularity * popularity_score)
        recommendations.append((i, combined_score))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    recommended_books = []
    count = 0
    for i, score in recommendations:
        if i == idx:
            continue
        recommended_books.append({
            'Title': df.loc[i, 'Name'],
            'Author': df.loc[i, 'Author'],
            'Genre': df.loc[i, 'Genre'],
            'User Rating': df.loc[i, 'User Rating'],
            'Reviews': df.loc[i, 'Reviews'],
            'Price': df.loc[i, 'Price'],
            'Year': df.loc[i, 'Year'],
            'Recommendation Score': round(score, 4)
        })
        count += 1
        if count >= top_n:
            break

    return pd.DataFrame(recommended_books)


def plot_recommendation_scores(book_title, top_n=10):
    results = recommend_books(book_title, top_n=top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(results['Title'], results['Recommendation Score'], color='lightblue')
    plt.xlabel("Recommendation Score")
    plt.title(f"Top {top_n} Recommendations for '{book_title}'")
    plt.gca().invert_yaxis()
    plt.show()


plot_recommendation_scores("A Gentleman in Moscow: A Novel", top_n=5)
