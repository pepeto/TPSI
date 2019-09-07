''''
Fill reasonable values for key inputs:
learning_rate: 0.01
n_estimators: 100 if the size of your data is high, 1000 is if it is medium-low
max_depth: 3
subsample: 0.8
colsample_bytree: 1
gamma: 1

Run model.fit(eval_set, eval_metric) and diagnose your first run, specifically the n_estimators parameter

Optimize max_depth parameter. It represents the depth of each tree, which is the maximum number of different features
used in each tree. I recommend going from a low max_depth (3 for instance) and then increasing it incrementally by 1,
and stopping when there’s no performance gain of increasing it. This will help simplify your model and avoid overfitting


Now play around with the learning rate and the features that avoids overfitting:

- learning_rate: usually between 0.1 and 0.01. If you’re focused on performance and have time in front of you, decrease
incrementally the learning rate while increasing the number of trees.

- subsample, which is for each tree the % of rows taken to build the tree. I recommend not taking out too many rows, as
performance will drop a lot. Take values from 0.8 to 1.

- colsample_bytree: number of columns used by each tree. In order to avoid some columns to take too much credit for the
prediction (think of it like in recommender systems when you recommend the most purchased products and forget about the
long tail), take out a good proportion of columns. Values from 0.3 to 0.8 if you have many columns (especially if you
did one-hot encoding), or 0.8 to 1 if you only have a few columns.

- gamma: usually misunderstood parameter, it acts as a regularization parameter. Either 0, 1 or 5.
'''