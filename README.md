Kaggle June-Playground Classification practice 

In efforts to be improve my data analysis skills, I attempted to model the classification dataset Kaggle released for their June competition.  After taking a data mining course in R, I am familiar with the concepts employed, but applying them to a large dataset and doing so in Python is something I'd like to improve at.

This dataset consists of 75 variables, or features, and each datapoint is classified into one of nine categories.  Initially, logistic regression is used to classify this data and is optimized with Stochastic Gradient Descent (SGDClassifier).  This proved successful as boosted logistic regression is a quick and efficient classifier that is relatively resistant to overfitting.

The submission for this competition is a csv file giving predicted classification probabilities for each data point in the given testing data.  For this reason, log loss is the metric used for model evaluation.  Log loss gives a score based on class prediction and prediction confidence.  Models are penalized for incorrect prediction and the how confident the model is in that incorrect prediction.  The winning log loss in this competition hovered around 1.73.  This logistic regression model achieved a log loss of approximately 1.82, which is serviceable and an achievement but clearly leaves a lot of meat left on the bone.

A second model using Random Forrest Classification was used.  RF is a slower model but it works very well with a large range of features as is present in this dataset.  Overfitting could be a bigger issue here, so cross-validation is necessary, as is testing the RF model with various sizes (# of trees) and maximum depths.  Once completed, this model, boosted with an LGBMClassifier for tree-optimization, a log loss of approximately 1.76 is achieved.

There is still much room for improvement, as the distance between 1.73 and 1.76 is large for a dataset as such, but this is certainly a humongous jump from the logistic regression model.
