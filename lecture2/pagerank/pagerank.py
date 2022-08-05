import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}

    corpus_length = len(corpus.keys())
    pages_length = len(corpus[page])

    random_factor = (1 - damping_factor) / corpus_length
    even_factor = damping_factor / pages_length

    for key in corpus.keys():
        if pages_length == 0:
            probability_distribution[key] = 1 / corpus_length

        else:
            if key not in corpus[page]:
                probability_distribution[key] = random_factor
            else:
                probability_distribution[key] = even_factor + random_factor

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize a dictionary with all pages as keys and values as 0 so all have 0 probability
    samples = {key: 0 for key in corpus.keys()}

    # Randomly select a page to start with
    key = random.choice(list(corpus.keys()))

    # Iterate over the n samples
    for _ in range(n):
        probabilty_distribution = transition_model(corpus, key, damping_factor) 
        prob_dist_list = list(probabilty_distribution.keys()) 
        weights = [probabilty_distribution[i] for i in probabilty_distribution] 
        key = random.choices(prob_dist_list, weights, k=1)[0] 
        samples[key] += 1

    return normalize(samples, n)


def normalize(dict, n):
    """
    Return a dictionary where the values are normalized to sum to 1.
    """
    for item in dict:
        dict[item] /= n

    return dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Constants
    THRESHOLD = 0.001
    N = len(corpus)

    ranks = {}
    
    for key in corpus:
        ranks[key] = 1 / N


    while True:
        count = 0

        for key in corpus:
            add = 0
            new_probability = (1 - damping_factor) / N

            for page in corpus:
                if key in corpus[page]:
                    links = len(corpus[page])
                    add += ranks[page] / links

            new_probability += damping_factor * add

            # Within 0.001 ( < )
            if abs(ranks[key] - new_probability) < THRESHOLD:
                count += 1

            ranks[key] = new_probability 

        if count == N:
            break

    return ranks


if __name__ == "__main__":
    main()
