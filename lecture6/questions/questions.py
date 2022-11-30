import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    files = dict()  # Dictionary of files and their contents

    # Loop over files in directory
    for filename in os.listdir(directory):

        # Check if file ends with ".txt"
        if not filename.endswith(".txt"):
            continue

        # Open file
        with open(os.path.join(directory, filename), encoding="utf8") as f:
            # Read contents into memory
            files[filename] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Tokenize words
    words = nltk.word_tokenize(document)

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Remove stopwords
    words = [word
             for word in words
             if word not in nltk.corpus.stopwords.words("english")]

    # Remove punctuation
    words = [word for word in words if word not in string.punctuation]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    words = set()  # Set of all words in all documents
    idfs = dict()  # Dictionary of words and their idf values

    # Get all words in documents
    # Loop over documents
    for document in documents:

        # Update words with words in document
        words.update(documents[document])

    # Calculate IDF for each word
    # Loop over words
    for word in words:

        # Get number of documents word appears in
        n = sum([word in documents[document] for document in documents])

        # Calculate IDF
        idfs[word] = math.log(len(documents) / n)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    tf_idfs = dict()  # Dictionary of files and their tf-idf values

    # Loop over files
    for file in files:

        # Initialize tf-idf value
        tf_idfs[file] = 0

        # Loop over words in query
        for word in query:

            # Get tf value
            tf = files[file].count(word)

            # Add tf-idf value to total
            tf_idfs[file] += tf * idfs[word]

    # Sort files by tf-idf value
    files = sorted(tf_idfs, key=tf_idfs.get, reverse=True)

    # Return top n files
    return files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    score = dict()  # dictionary of sentences and their scores

    # Loop over sentences
    for sentence in sentences:

        # Matchin word measure
        measure = sum(idfs[word]
                      for word in query
                      if word in sentences[sentence])

        # Query word density
        density = sum(word in query
                      for word in sentences[sentence]) / len(sentences[sentence])

        # Add score to dictionary
        score[sentence] = (measure, density)

    # Sort sentences by score
    sentences = sorted(score, key=score.get, reverse=True)

    # Return top n sentences
    return sentences[:n]


if __name__ == "__main__":
    main()
