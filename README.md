# 1. Project Overview

The system predicts whether a user will click on a specific news article based on their historical reading behavior. It processes large-scale news data and user logs to build:

Content-Based Models: Using TF-IDF and Cosine Similarity.

Collaborative Filtering: Implementing SVD (Singular Value Decomposition) and KNN via the surprise library.

Deep Learning Models: Implementing Neural Collaborative Filtering (NCF) and sequence-based models using LSTM and GRU in TensorFlow.


# 2. Environment Setup

To run this notebook, we need a Python environment with the following libraries installed:

a. Core Data Processing

pip install pandas numpy tqdm

b. Visualization

pip install matplotlib seaborn wordcloud

c. Machine Learning & NLP

pip install scikit-learn gensim scikit-surprise

d. Deep Learning

pip install tensorflow


# 3. Data Requirements

The code expects two primary tab-separated (.tsv) files in the same directory as the notebook:

a. news.tsv: Contains article metadata (news_id, category, subcategory, title, abstract, etc.).

b. behaviors.tsv: Contains user logs (user_id, time, history of clicked news, and impression logs).


# 4. Step-by-Step Execution

Step 1: Data Preprocessing

Load the datasets and handle missing values in abstracts and user histories.

Analyze the distribution of news categories and subcategories to understand content variety.

Generate word clouds to visualize the most common topics in news titles.

Step 2: Feature Engineering

Combine titles and abstracts into a single full_text feature.

Use TfidfVectorizer to convert text into numerical vectors for similarity calculations.

Generate user profiles by averaging the vectors of all articles they have previously clicked.

Step 3: Model Training & Evaluation

The notebook executes several modeling approaches:

Content-Based: Recommends articles similar to the user's profile using cosine similarity.

Collaborative Filtering: Uses the surprise library to train SVD and KNN models on user-item interaction matrices.

Neural Network (NCF): Build a dual-input model (User History & Candidate News) using Embedding layers and Dense layers.

Sequence Models: Uses LSTM/GRU to capture the temporal order of user clicks.

Step 4: Performance Metrics

Models are evaluated using standard recommendation metrics:

AUC-ROC: To measure the model's ability to distinguish between clicked and non-clicked articles.

NDCG: To evaluate the ranking quality of recommendations.


# 5. Running the Code

Open the News Recommendation.ipynb in Jupyter Notebook or Google Colab.

Run the first cell to import all necessary dependencies.

Execute the cells sequentially. The deep learning sections (TensorFlow) may require a GPU for faster training, though the code is configured to run on a standard CPU.