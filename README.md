# machine-learning-hole-offset
Attempt to use machine learning to approximate the Stress Intensity Factors (SIF, or K-solutions) for finite-width offset hole specimen geometries

## Evaluation Metric Ideas
- Current AFGROW method involves linear-interpolation between 2D datapoints along "splines". To reduce the dataset to 2D, the splines/surfaces defining the nearest of each parameters are used. This makes it a combination between nearest-neighbor and linear interpolation
    - For example, if the parameters are (a/c= , a/t=, r/t=, w/R = ), then the dataset is reduced to the nearest (w/r), then sliced to the nearest (r/t), then the nearest (a/t), then a linear interpolation between the two nearest (a/c) parameters is performed.
    - This makes it tricky to evaluate against, since the AFGROW's true parameter is not the actual true parameters
- The proposed evaluation metric is therefore:
    - Extract a wider range of K-solutions between the current values using StressCheck
    - Train on the original subset of K-solutions
    - Perform the AFGROW-style regression on the extra data to get a baseline (AFRGROW vs. true K) error \epsilon_AFGROW
    - Perform the trained model regression on the extra data to get a (ML vs. true K) error \epsilon_MLmodel
    - Include computation time metric

- A simpler metric without extracting additional ML data would be:
    - Compute \epsilon_AFGORW regressing on training dataset, predicting on test dataset. Compare with \epsilon_MLmodel from test loss
    - 

## Models under investigation
- Standard NN (4-5 layer)
- Ridge Regression NN (4-layer)
    - Minimizes weights on unimportant terms. 1e-4 / 1e-5 regularization weight works well, converges mostly over 500 epochs, good loss performance. Unclear on overfitting, validation set looks good
- Lasso Regression NN (4-layer)
    - Same notes as above
- Gaussian Process Regression
    - 
- DeepONet 
- Fourier Neural Operator (FNO)
- BINGO


## To-Do List
### Phase 1: Exploring the problem and ML Regression as a valid K-solution fitting approach
[x] Import .dat dataset files
[x] Extract vertices
[x] Train basic NN model
[x] Experiment with different basic NN models
[x] Add Lasso/Ridge Regression
[ ] Implement classical regression techniques to understand applicability

### Phase 2: Developing robust comparison metric pipeline and implementing state-of-the-art/novel methods
[x] Understand geometric limits on hole offset dataset
[ ] Experiment with reducing dataset size (removing outliers, and training on much smaller subsets)
[ ] Organize GitHub
[ ] AFGROW-style prediction model with evaluation metric
[ ] Equivalent ML prediciton model with evaluation metric
[ ] Implement GPR
[ ] Implement operator learning strategy
[ ] Implement SVR
[ ] Implement BINGO

### Phase 3: Integration with AFGROW


### Phase 4: Expanding to new datasets
[ ] 
