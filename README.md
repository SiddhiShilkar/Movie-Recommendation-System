# Movie Recommendation System with TMDB Dataset

Welcome to our Movie Recommendation System! This system utilizes the TMDB (The Movie Database) dataset from Kaggle to provide personalized movie recommendations based on user preferences and movie content.

# Dataset Source:

- The dataset used in this project is sourced from Kaggle's TMDB 5000 Movie Dataset.
- The dataset consists of two CSV files: tmdb_5000_credits.csv containing movie credits information and tmdb_5000_movies.csv containing movie details.

# Features:

- Pandas: We use the Pandas library in Python for data manipulation and analysis.
- Scikit-learn: We utilize Scikit-learn for feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and calculating cosine similarity scores.

# Usage:

- Data Loading and Cleaning: The datasets are loaded into Pandas DataFrames and merged on the 'id' column to combine movie information with credits.
Unnecessary columns such as 'homepage', 'title_x', 'title_y', 'status', and 'production_countries' are dropped to clean the data.
- TF-IDF Vectorization:
Genres and keywords columns are filled with empty strings where missing.
A new 'content' column is created by combining 'genres' and 'keywords' for each movie.
- TF-IDF vectorization is performed on the 'content' column to convert text data into numerical vectors.
- Cosine Similarity Calculation: Cosine similarity scores are computed using the TF-IDF matrix, representing similarities between movies based on their content.
- Recommendation Function: The get_recommendations function takes a movie title as input and returns top recommended movies based on similarity scores. It finds the index of the input movie, sorts similarity scores in descending order, excludes the input movie itself, and returns the titles of recommended movies.
- Example Usage: An example is provided where recommendations are generated for the movie 'The Dark Knight Rises' using the recommendation function.

# How to Run:

- Ensure you have Python installed on your system along with the required libraries (pandas, numpy, scikit-learn).
- Download the TMDB dataset provided and place the CSV files in the same directory as the code.
- Run the Python script to load the data, preprocess it, calculate recommendations, and see the results.
