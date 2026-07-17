# classifier.py
#
# Built off of the part Lect2 on Logistics Regression using sigmoid, I researched Multiclass Logistics Regression (Softmax Regression)
#
# notations and vars used:
# N      - num training samples
# D      - num input features
# Db     - D+1, feature count after bias col is added
# K      - num unique classes
# B      - current mini-batch size
# W      - weight matrix learned in fit()
# Z      - logits (raw class scores before softmax)
# P      - class probs after softmax
# Y      - one-hot encoded labels
# X      - input feature matrix
# Xb     - X with bias col of 1s appended
# Xbatch - current mini-batch of Xb
# Ybatch - current mini-batch of Y
# y      - raw target labels as ints
# y_idx  - y remapped to 0..K-1
# expZ   - exp(Z) used in softmax
# lr     - learning rate
# l2     - l2 regularisation strength
# rng    - random num generator
# perm   - shuffled indices for each epoch (permutation)

import numpy as np

# Classifier class: trains a softmax regression model to predict pacman's next move

class Classifier:
    def __init__(self):
        # hyperparams were tuned off trial and error to work on a variety of data sets
        self.lr = 0.2
        self.l2 = 0.0001
        self.batch_size = 32
        self.seed = 0
        self.max_epochs = 200     # hard safety cap of 200
        self.patience = 6          # wait time to check for improvement by this many epochs
        self.min_delta = 0.004   # min loss we count as progress, used with patience for early stop

        # model params (learned in fit() so set to NONE for now)
        self.W = None               
        self.classes_ = None        
        self.is_trained = False

    def reset(self):
        """wipe learned params, safe to call between runs"""
        self.W = None
        self.classes_ = None
        self.is_trained = False

    # helper methods: help to prep data and perform core maths
    def _as_2d(self, X):
        """takes a 1D or 2D array and returns a reshaped 2D array"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _add_bias(self, X2d):
        """
        adds a col of 1s to each row so model learns a bias term
        
        args:
            X2d: 2D feature matrix (N, D)
        returns:
            np.ndarray: X2d with bias col appended (N,D+1)
        """
        n = X2d.shape[0]
        bias = np.ones((n, 1), dtype=np.float64)
        return np.hstack([X2d, bias])

    def _softmax(self, Z):
        """
        computes row-wise softmax subtracting row max before exp to avoid overflow (numerical stability)
        done because as e^n approaches inf we get a lot of NaN causing crashes 

        args:
            Z: logits (N,K)
        returns:
            np.ndarray: softmax probs (N,K)
        """
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _one_hot(self, y_idx, K):
        """
        converts class indices to one-hot encoded labels

        args:
            y_idx: class indices array (N,)
            K: num of classes
        returns:
            np.ndarray: one-hot matrix (N,K)
        """
        Y = np.zeros((y_idx.shape[0], K), dtype=np.float64)
        Y[np.arange(y_idx.shape[0]), y_idx] = 1.0
        return Y

    # fit: trains model utilising mini-batch gradient descent. uses early stopping to prevent overfitting
    def fit(self, data, target):
        """
        trains model on data/target pairs

        args:
            data: feature matrix (N,D) of 0/1 ints
            target: move labels (N,) encoded as ints 0-3
        """
        X = self._as_2d(data)
        y = np.asarray(target, dtype=np.int64)

        # check for empty data
        if X.shape[0] == 0:
            self.reset()
            return

        # get unique classes and map them to 0..K-1 as labels aren't guaranteed to start at 0
        self.classes_ = np.unique(y)
        K = self.classes_.shape[0]

        # trivial model made if only 1 class in data (can't do multiclass with 1 class)
        if K == 1:
            self.W = None
            self.is_trained = True
            return

        label_idx = {int(c): i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_idx[int(c)] for c in y], dtype=np.int64)

        # adds bias col
        Xb = self._add_bias(X)
        N, Db = Xb.shape # Db is just D+1, i.e original feats +1 for bias col

        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(loc=0.0, scale=0.01, size=(Db, K))

        Y = self._one_hot(y_idx, K)

        # early stopping added, makes code more robust to varied data sets 
        best_loss = float("inf")
        patience_left = self.patience

        # stops early when converged but otherwise trains til predef max epochs
        for epoch in range(self.max_epochs):
            perm = rng.permutation(N)

            # mini-batch gradient descent
            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                Xbatch = Xb[idx]
                Ybatch = Y[idx]
                B = Xbatch.shape[0]

                logits = Xbatch @ self.W
                P = self._softmax(logits)

                # cross-entropy loss gradient: (P - Y) scaled by batch size
                grad = (Xbatch.T @ (P - Ybatch)) / B 

                # l2 penalty on weights but not the bias col
                grad[:-1, :] += self.l2 * self.W[:-1, :]

                self.W -= self.lr * grad

            # checking for convergence by checking loss on full dataset each epoch
            logits_all = Xb @ self.W
            probs_all = self._softmax(logits_all)
            loss = -np.mean(np.sum(Y * np.log(probs_all + 1e-10), axis=1)) # used e-10 to stop it crashing on log(0)

            # if loss improved enough reset patience, else count down. "enough" is defined by min_delta
            if loss < best_loss - self.min_delta:
                best_loss = loss
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
           
        self.is_trained = True

    # predict: predicts best legal move given the curr game state
    def predict(self, data, legal=None):
        """
        returns predicted move as int 0-3 (north, east, south, west)
        filters by legal moves if given, ignores stop.

        args:
            data: feature vector for curr game
            legal: list of legal moves as strings (optional)
        returns:
            int: predicted move label as int 0-3
        """

        # default to east if not trained like before
        if not self.is_trained or self.classes_ is None:
            return 1

        # always predict the class seen if only 1 class seen
        if self.classes_.shape[0] == 1:
            return int(self.classes_[0])

        # fallback if weights are missing
        if self.W is None:
            return 1

        # forward pass to get class probs
        x = self._as_2d(data) # single sample
        xb = self._add_bias(x) # single sample w/ bias col
        probs = self._softmax(xb @ self.W)[0]

        # map numerical labels to legal action strings
        num_dir = {0: "North", 1: "East", 2: "South", 3: "West"}

        # creates dict lookup to find where each label is in the list of classes
        label_idx = {int(c): i for i, c in enumerate(self.classes_)}

        # grabs prob for each label from list of probs
        label_probs = {label: float(probs[idx]) for label, idx in label_idx.items()}

        # returns best overall label if no legal given
        if legal is None or len(legal) == 0:
            return max(label_probs, key=label_probs.get)

        # chooses best label that corresponds to a legal direction ignoring stop
        legal_set = set(legal)
        best_label = None
        best_prob = -1.0

        for label, direction in num_dir.items():
            if direction in legal_set and label in label_probs:
                curr_prob = label_probs[label]
                if curr_prob > best_prob:
                    best_prob = curr_prob
                    best_label = label

        # if no matches found default to best overall label
        if best_label is None:
            return max(label_probs, key=label_probs.get)

        return int(best_label)


