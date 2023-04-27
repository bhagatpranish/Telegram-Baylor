import os
import re
import PyPDF2
from PyPDF2 import PdfReader
import pdfplumber
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')


def tokenize_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def extract_keywords_tfidf(text, n_keywords=10):
    """
    Extracts keywords from a given text using TF-IDF.

    :param text: str, input text
    :param n_keywords: int, number of keywords to extract (default: 10)
    :return: list of tuple, extracted keywords and their scores
    """
    # Tokenize the text into sentences
    sentences = tokenize_sentences(text)

    # Create a custom stopword list
    custom_stopwords = set(nltk.corpus.stopwords.words('english')).union({'additional', 'stopwords'})

    # Create a TF-IDF vectorizer with stopwords removed and n-grams enabled
    vectorizer = TfidfVectorizer(stop_words=custom_stopwords, ngram_range=(1, 3))
    word_vectors = vectorizer.fit_transform(sentences)

    # Get the feature names (words) and their scores
    feature_names = vectorizer.get_feature_names_out()
    scores = word_vectors.toarray().sum(axis=0)

    # Get the indices of the top n_keywords
    top_indices = scores.argsort()[-n_keywords:]

    # Create a list of tuples with the keywords and their scores
    keywords = [(feature_names[i], scores[i]) for i in top_indices]

    return keywords


def extract_text_from_pdf(file_path):
    """
    Extracts text from a given PDF file.

    :param file_path: str, path to the PDF file
    :return: str, extracted text from the PDF
    """
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_keywords(text, n_keywords=10):
    """
    Extracts keywords from a given text.

    :param text: str, input text
    :param n_keywords: int, number of keywords to extract (default: 5)
    :return: list of tuple, extracted keywords and their scores
    """
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=n_keywords)
    return keywords


def strip_and_remove_duplicates(pdf_keywords_without_scores):
    """
    This function takes a dictionary containing lists of keywords and performs the following operations:
    1. Splits keywords on spaces and underscores.
    2. Strips leading and trailing special characters (_, ., -) from the keywords.
    3. Removes keywords that occur in multiple places in the dictionary, including partial matches.

    Args:
        pdf_keywords_without_scores (dict): A dictionary containing lists of keywords.
            The keys represent the name of the PDF file, and the values are lists of keywords associated with the PDF.

    Returns:
        dict: A dictionary containing lists of unique and stripped keywords.
            The keys represent the name of the PDF file, and the values are lists of unique and stripped keywords
            associated with the PDF.
    """
    stripped_keywords = {}

    for key, values in pdf_keywords_without_scores.items():
        split_values = []
        for value in values:
            split_values.extend(value.split(" "))
            split_values.extend(value.split("_"))

        stripped_values = [value.strip("_,.-") for value in split_values]
        stripped_keywords[key] = stripped_values

    unique_keywords = {}

    for key, values in stripped_keywords.items():
        unique_values = []
        for value in values:
            if sum([any(value in val for val in val_list) for val_list in stripped_keywords.values()]) == 1:
                unique_values.append(value)
        unique_keywords[key] = unique_values

    return unique_keywords


def main():
    """
    Main function to process PDF files and extract keywords.
    """
    article_directory = 'articles'
    pdf_files = [f for f in os.listdir(article_directory) if f.endswith('.pdf')]

    pdf_keywords = {}  # Initialize an empty dictionary to store keywords for each PDF file
    pdf_keywords_without_scores = {}  # Initialize an empty dictionary to store keywords without scores

    # Iterate through PDF files, extract text and keywords, and update the dictionaries
    for pdf_file in pdf_files:
        file_path = os.path.join(article_directory, pdf_file)
        text = extract_text_from_pdf(file_path)
        keywords = extract_keywords_tfidf(text)

        pdf_keywords[
            pdf_file] = keywords  # Update the dictionary with the PDF filename as the key and the keywords as the value
        pdf_keywords_without_scores[pdf_file] = [keyword for keyword, _ in
                                                 keywords]  # Update the dictionary without scores

    # Call strip_and_remove_duplicates function on the pdf_keywords_without_scores dictionary
    pdf_keywords_without_scores = strip_and_remove_duplicates(pdf_keywords_without_scores)

    pdf_keywords_without_duplicates = {}

    for key, value in pdf_keywords_without_scores.items():
        pdf_keywords_without_duplicates[key] = list(set(value))

    # Write the keyword dictionaries to a new Python file called keyword_dict.py
    with open("keyword_dict.py", "w") as f:
        f.write("pdf_keywords = {\n")
        for pdf_name, keywords in pdf_keywords.items():
            f.write(f"    \"{pdf_name}\": {keywords},\n")
        f.write("}\n\n")

        f.write("pdf_keywords_without_scores = {\n")
        for pdf_name, keywords in pdf_keywords_without_duplicates.items():
            f.write(f"    \"{pdf_name}\": {keywords},\n")
        f.write("}\n")

    print("Keyword dictionaries written to keyword_dict.py")


if __name__ == "__main__":
    main()
