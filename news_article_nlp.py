import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ============================================================
# DOWNLOAD NLTK DATA
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("LOADING IMDb DATASET")
print("==========================================")

df = pd.read_csv(
    "/content/IMDB_dataset.csv"
)

print(
    "Dataset loaded successfully!"
)

print(
    "Original reviews:",
    len(df)
)


# ============================================================
# CHECK DATASET
# ============================================================

print("\nColumns:")

print(
    df.columns.tolist()
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=[
        "review",
        "sentiment"
    ]
).reset_index(
    drop=True
)


# ============================================================
# REDUCE DATASET
# 2500 POSITIVE + 2500 NEGATIVE
# ============================================================

print("\n==========================================")
print("REDUCING DATASET")
print("==========================================")

positive = df[
    df["sentiment"] == "positive"
].sample(
    n=2500,
    random_state=42
)

negative = df[
    df["sentiment"] == "negative"
].sample(
    n=2500,
    random_state=42
)


df = pd.concat([
    positive,
    negative
])


# Shuffle

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


print(
    "Reviews used:",
    len(df)
)

print(
    "\nPositive:",
    len(
        df[
            df["sentiment"] == "positive"
        ]
    )
)

print(
    "Negative:",
    len(
        df[
            df["sentiment"] == "negative"
        ]
    )
)


# ============================================================
# NLP TOOLS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove numbers and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Tokenization
    tokens = word_tokenize(
        text
    )

    # Stop word removal
    filtered = [
        word
        for word in tokens
        if word not in stop_words
        and word.isalpha()
    ]

    # Stemming
    stemmed = [
        stemmer.stem(word)
        for word in filtered
    ]

    return " ".join(
        stemmed
    )


# ============================================================
# PROCESS DATASET
# ============================================================

print("\n==========================================")
print("NLP PREPROCESSING")
print("==========================================")

print(
    "Tokenization + Stop Word Removal + Stemming"
)

df["processed_review"] = df[
    "review"
].apply(
    preprocess
)

print(
    "Preprocessing completed!"
)


# ============================================================
# SHOW DATASET PREPROCESSING EXAMPLE
# ============================================================

print("\n==========================================")
print("DATASET PREPROCESSING EXAMPLE")
print("==========================================")

print("\nORIGINAL REVIEW:")

print(
    df["review"].iloc[0]
)

print("\nPROCESSED REVIEW:")

print(
    df["processed_review"].iloc[0]
)


# ============================================================
# SENTIMENT CONVERSION
# ============================================================

df["sentiment_value"] = df[
    "sentiment"
].map({

    "positive": 1,

    "negative": 0

})


# ============================================================
# INPUT AND OUTPUT
# ============================================================

X = df[
    "processed_review"
]

y = df[
    "sentiment_value"
]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\n==========================================")
print("TRAIN TEST SPLIT")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "Training reviews:",
    len(X_train)
)

print(
    "Testing reviews:",
    len(X_test)
)


# ============================================================
# TF-IDF
# ============================================================

print("\n==========================================")
print("TF-IDF")
print("==========================================")

tfidf = TfidfVectorizer(

    max_features=3000,

    ngram_range=(1, 2)
)


# Fit on training data

X_train_tfidf = tfidf.fit_transform(
    X_train
)


# Transform test data

X_test_tfidf = tfidf.transform(
    X_test
)


print(
    "TF-IDF completed!"
)

print(
    "Number of features:",
    len(
        tfidf.get_feature_names_out()
    )
)


# ============================================================
# TF-IDF TABLE
# ============================================================

print("\n==========================================")
print("TF-IDF TABLE")
print("==========================================")

features = tfidf.get_feature_names_out()


tfidf_table = pd.DataFrame(

    X_train_tfidf[:5].toarray(),

    columns=features

)


print(
    "Showing first 20 TF-IDF features:"
)

display(
    tfidf_table.iloc[
        :,
        :20
    ].round(4)
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n==========================================")
print("MODEL TRAINING")
print("==========================================")

model = LogisticRegression(

    max_iter=1000,

    random_state=42

)


model.fit(

    X_train_tfidf,

    y_train

)


print(
    "Model training completed!"
)


# ============================================================
# TEST MODEL
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n==========================================")
print("MODEL ACCURACY")
print("==========================================")

print(
    "Accuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Negative",
            "Positive"
        ]

    )
)


# ============================================================
# COMPLETE QUERY ANALYSIS FUNCTION
# ============================================================

def analyze_query(query):


    # ========================================================
    # 1. ORIGINAL QUERY
    # ========================================================

    print("\n")
    print("==========================================")
    print("             NLP ANALYSIS")
    print("==========================================")


    print("\n1. ORIGINAL TEXT")
    print("------------------------------------------")

    print(
        query
    )


    # ========================================================
    # 2. CLEAN TEXT
    # ========================================================

    clean_text = str(
        query
    ).lower()


    # Remove HTML

    clean_text = re.sub(
        r"<.*?>",
        " ",
        clean_text
    )


    # Remove URLs

    clean_text = re.sub(
        r"http\S+|www\S+",
        " ",
        clean_text
    )


    # Remove special characters

    clean_text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        clean_text
    )


    # ========================================================
    # 3. TOKENIZATION
    # ========================================================

    tokens = word_tokenize(
        clean_text
    )


    print("\n2. TOKENIZATION")
    print("------------------------------------------")

    print(
        tokens
    )


    # ========================================================
    # 4. STOP WORDS FOUND
    # ========================================================

    found_stopwords = [

        word

        for word in tokens

        if word in stop_words

    ]


    print("\n3. STOP WORDS FOUND")
    print("------------------------------------------")


    if len(found_stopwords) > 0:

        print(
            found_stopwords
        )

    else:

        print(
            "No stop words found."
        )


    # ========================================================
    # 5. AFTER STOP WORD REMOVAL
    # ========================================================

    filtered_tokens = [

        word

        for word in tokens

        if word not in stop_words

        and word.isalpha()

    ]


    print("\n4. AFTER STOP WORD REMOVAL")
    print("------------------------------------------")

    print(
        filtered_tokens
    )


    # ========================================================
    # 6. STEMMING
    # ========================================================

    stemmed_tokens = [

        stemmer.stem(
            word
        )

        for word in filtered_tokens

    ]


    print("\n5. STEMMING")
    print("------------------------------------------")


    stemming_table = pd.DataFrame({

        "Original Word":
            filtered_tokens,

        "Stemmed Word":
            stemmed_tokens

    })


    display(
        stemming_table
    )


    # ========================================================
    # 7. PROCESSED TEXT
    # ========================================================

    processed = " ".join(
        stemmed_tokens
    )


    print("\n6. FINAL PROCESSED TEXT")
    print("------------------------------------------")

    print(
        processed
    )


    # ========================================================
    # 8. TF-IDF
    # ========================================================

    vector = tfidf.transform(

        [processed]

    )


    # Get values

    values = vector.toarray()[0]


    # Get vocabulary

    words = tfidf.get_feature_names_out()


    # Create table

    tfidf_query = pd.DataFrame({

        "Word": words,

        "TF-IDF": values

    })


    # Only words present in query

    tfidf_query = tfidf_query[

        tfidf_query[
            "TF-IDF"
        ] > 0

    ]


    # Highest first

    tfidf_query = tfidf_query.sort_values(

        by="TF-IDF",

        ascending=False

    )


    print("\n7. TF-IDF VALUES")
    print("------------------------------------------")


    if len(tfidf_query) > 0:

        display(

            tfidf_query.round(4)

        )

    else:

        print(
            "No query words found in "
            "the trained TF-IDF vocabulary."
        )


    # ========================================================
    # 9. SENTIMENT PREDICTION
    # ========================================================

    prediction = model.predict(

        vector

    )[0]


    # Probability

    probability = model.predict_proba(

        vector

    )[0]


    # ========================================================
    # 10. SENTIMENT
    # ========================================================

    if prediction == 1:

        sentiment = "POSITIVE"

        confidence = (
            probability[1] * 100
        )

    else:

        sentiment = "NEGATIVE"

        confidence = (
            probability[0] * 100
        )


    # ========================================================
    # 11. FINAL RESULT
    # ========================================================

    print("\n8. FINAL SENTIMENT")
    print("------------------------------------------")

    print(
        "Sentiment:",
        sentiment
    )

    print(
        "Confidence:",
        round(
            confidence,
            2
        ),
        "%"
    )


    print("\n==========================================")
    print("             ANALYSIS COMPLETE")
    print("==========================================")


# ============================================================
# USER QUERY LOOP
# ============================================================

print("\n==========================================")
print("       IMDb NLP SENTIMENT ANALYZER")
print("==========================================")

print(
    "\nMODEL IS READY!"
)

print(
    "\nEnter any movie review or sentence."
)

print(
    "\nThe program automatically performs:"
)

print(
    "Tokenization"
)

print(
    "↓"
)

print(
    "Stop Word Identification"
)

print(
    "↓"
)

print(
    "Stop Word Removal"
)

print(
    "↓"
)

print(
    "Stemming"
)

print(
    "↓"
)

print(
    "TF-IDF"
)

print(
    "↓"
)

print(
    "Sentiment Prediction"
)

print(
    "\nType 'exit' to stop."
)


# ============================================================
# CONTINUOUS QUERY LOOP
# ============================================================

while True:

    query = input(
        "\nEnter your query: "
    )


    # Exit

    if query.lower().strip() == "exit":

        print(
            "\nProgram stopped."
        )

        break


    # Empty query

    if query.strip() == "":

        print(
            "Please enter some text."
        )

        continue


    # Analyze everything

    analyze_query(
        query
    )
