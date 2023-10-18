import string

def preprocess(sentence):
    """
    Convert `sentence` to a list of its words. Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.word_tokenize(sentence)  # Tokenize 
    # Filter out words that don't contain alphabetic characters
    words = [word.lower() for word in words if any(c.isalpha() for c in word)]
    return words

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []
    
    # Helper function to check if a subtree contains other NP subtrees
    def contains_np(subtree):
        return len(list(subtree.subtrees(lambda t: t.label() == 'NP'))) > 1

    for subtree in tree.subtrees():
        if subtree.label() == 'NP' and not contains_np(subtree):
            np_chunks.append(subtree)
    
    return np_chunks
