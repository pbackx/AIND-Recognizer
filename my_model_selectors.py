import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        return self.model(num_states, self.X, self.lengths)

    def model(self, num_states, X, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)
            try:
                logL = model.score(self.X, self.lengths)
                # the number of parameters is the sum of:
                # 1. the transition probabilities between the different states: you can only go to the next state, so: n-1
                # 2. the mean and variance for each feature in every state: 2 * n * len(model.means_[0])
                p = n - 1 + 2 * n * len(model.means_[0])
                N = len(self.X)
                logN = np.log(N)
                score = -2 * logL + p * logN
                if score < best_score:
                    best_score = score
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        M = len(self.words)
        best_score = float("-inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n)
            try:
                logPXi = model.score(self.X, self.lengths)
                sum = 0
                for word in self.hwords:
                    if not word == self.this_word:
                        X, lengths = self.hwords[word]
                        sum += model.score(X, lengths)
                score = logPXi - (sum / (M - 1))
                if score > best_score:
                    best_score = score
                    best_model = model
            except:
                pass

        #https://ai-nd.slack.com/files/petetanru/F4Y0WQW0K/my_DIC_result.txt
        # TODO implement model selection based on DIC scores
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_num_states = None

        # First construct the training and test folds.
        # I'm sure there is a much more elegant way to write this using list comprehensions
        splits = []
        if len(self.sequences) == 1:
            # we can't make any fold, so we just search the best score
            for n in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(n)
                try:
                    logL = model.score(self.X, self.lengths)
                    if logL > best_score:
                        best_score = logL
                        best_num_states = n
                except:
                    pass
            return self.base_model(best_num_states)
        elif len(self.sequences) == 2:
            X1, lengths1 = combine_sequences([0], self.sequences)
            X2, lengths2 = combine_sequences([1], self.sequences)
            splits = [
                (X1, lengths1, X2, lengths2),
                (X2, lengths2, X1, lengths1),
            ]
        else:
            for cv_train_idx, cv_test_idx in KFold().split(self.sequences):
                X_training, lengths_training = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                splits.append((X_training, lengths_training, X_test, lengths_test))

        for n in range(self.min_n_components, self.max_n_components+1):
            score = 0
            for X_training, lengths_training, X_test, lengths_test in splits:
                model = self.model(n, X_training, lengths_training)
                try:
                    logL = model.score(X_test, lengths_test)
                    score += logL
                except:
                    pass
            if score > best_score:
                best_score = score
                best_num_states = n

        return self.base_model(best_num_states)
