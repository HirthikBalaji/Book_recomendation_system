# 📚 Advanced Book Recommendation System

An intelligent hybrid book recommendation system built in Python, using a dataset of Amazon bestsellers. This project combines **content-based filtering** with **popularity-based scoring** (ratings and reviews), offering highly relevant and customizable book recommendations.

---

## 🔍 Features

- ✅ **Content-Based Filtering** using TF-IDF on book title, author, and genre
- ✅ **Popularity Scoring** based on normalized user ratings and review counts
- ✅ **Hybrid Recommendation Engine** that combines textual similarity with popularity
- ✅ **Customizable Weighting** between content similarity and popularity
- ✅ **Visualization Support** for recommendation scores
- ✅ Clean, readable, and extensible code

---

## 📂 Dataset

The dataset used is:

```

bestsellers with categories.csv

````

### Columns:

| Column Name   | Description                         |
|---------------|-------------------------------------|
| `Name`        | Book title                          |
| `Author`      | Author's name                       |
| `User Rating` | Average user rating (1.0–5.0)       |
| `Reviews`     | Number of reviews                   |
| `Price`       | Price in USD                        |
| `Year`        | Year of appearance in the bestseller list |
| `Genre`       | Book genre (Fiction or Non Fiction) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HirthikBalaji/Book_recomendation_system.git
cd Book_recomendation_system
````

### 2. Install dependencies

```bash
pip install pandas scikit-learn matplotlib
```

### 3. Run the recommender

```python
from book_recommender import recommend_books

results = recommend_books("The Subtle Art of Not Giving a F*ck", top_n=5)
print(results)
```

---

## 🧠 How It Works

### 🔸 Content-Based Filtering

* Uses **TF-IDF vectorization** on combined features: `Title + Author + Genre`
* Computes **cosine similarity** to find books with similar textual attributes

### 🔸 Popularity Score

* Normalized scoring based on:

  * `User Rating` (60% weight)
  * `Reviews` (40% weight)

### 🔸 Hybrid Scoring Formula

```python
Final Score = (weight_content * content_similarity) + (weight_popularity * popularity_score)
```

Weights are customizable per query.

---

## 📊 Visualization

```python
from book_recommender import plot_recommendation_scores

plot_recommendation_scores("Atomic Habits", top_n=5)
```

Generates a bar chart of recommendation scores for top N suggestions.

---

## ⚙️ Customization

You can adjust the influence of content and popularity:

```python
# More content-driven
recommend_books("Atomic Habits", weight_content=0.9, weight_popularity=0.1)

# More popularity-driven
recommend_books("Atomic Habits", weight_content=0.4, weight_popularity=0.6)
```

---

## 📁 File Structure

```
book-recommendation-system/
│
├── book_recommender.py       # Core recommendation engine
├── bestsellers with categories.csv  # Dataset file
├── README.md                 # Project documentation
└── requirements.txt          # (Optional) Dependency list
```

---

## ✅ To Do

* [ ] Add collaborative filtering (user-based or item-based)
* [ ] Build a Streamlit or Flask web app interface
* [ ] Deploy as an API

---

## 📄 License

MIT License. Feel free to use and adapt.

---

## 🙋‍♂️ Contact

Made with ❤️ by Hirthik Balaji C

Feel free to reach out or open an issue!
