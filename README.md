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
