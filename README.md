# NLP-Assigment

# News NLP TF-IDF Classifier

## Project Overview

This project is a Natural Language Processing (NLP) based News Article Classification System.

The system takes news articles as input, performs different NLP preprocessing techniques, converts the processed text into numerical features using TF-IDF, and uses Logistic Regression to predict the category of a news article.

The project also provides an interactive query system. After the model is trained, the user can enter any news-related query and the system will display the complete NLP processing and predicted category.

---

## Dataset

The project uses a news article dataset containing 2,000 records.

### Dataset Columns

- `headlines` - News article headline
- `description` - Short description of the article
- `content` - Full article content
- `url` - URL of the article
- `category` - News category

The `url` column is not used for NLP processing.

The following columns are combined to create the input text:

```text
headlines + description + content


Installation and Requirements
Requirements

Make sure you have the following installed:

Python 3.8 or above
Pandas
NLTK
Scikit-learn
Google Colab or VS Code (for running the project)
Install Required Libraries

Open the terminal in VS Code or your command prompt and run:

pip install pandas nltk scikit-learn
NLTK Requirements

The Python program automatically downloads the required NLTK resources:

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

No separate installation is required for these.

Google Colab

If you are using Google Colab, run this in a code cell before running the project:

!pip install pandas nltk scikit-learn

Then upload:

business_data(1).csv

and run the main Python code.

VS Code

Install Python first, then open the VS Code terminal and run:

pip install pandas nltk scikit-learn

Then run the project:

python news_nlp_tfidf_classifier.py
Python Libraries Used
Library	Purpose
pandas	Dataset loading and data processing
nltk	Tokenization, stop-word removal and stemming
scikit-learn	TF-IDF, Logistic Regression and evaluation
re	Text cleaning
System Requirements

Recommended:

RAM: 4 GB or more
Storage: 500 MB or more
Python: 3.8+
