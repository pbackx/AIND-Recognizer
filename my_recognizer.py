import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # check every word in the test set one by one
    for word_id, Xlengths in test_set.get_all_Xlengths().items():
        X, lengths = Xlengths
        best_score = float("-inf")
        best_word = None
        scores = {}
        # go over all models and find the one with the highest probability
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
                scores[word] = logL
                if logL > best_score:
                    best_score = logL
                    best_word = word
            except:
                scores[word] = float("-inf")
        probabilities.append(scores)
        guesses.append(best_word)

    return probabilities, guesses
