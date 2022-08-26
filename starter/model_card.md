# Model Card

## Model Details
João Pires created this model. It is a Random Forest model trained with the default sklearn hyperparameters, except n_estimators=20 and min_samples_split=5.

## Intended Use
This model should be used to predict if a person's salary is higher or lower than 50k based on a handful of attributes.

## Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 32561 rows, and an 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
The model was evaluated based on precision, recall and F-beta score. The values were 0.738, 0.629 and 0.679, respectively.

## Ethical Considerations
The Aequitas package was used to run bias checks on race, sex, education, marital status and work class. The results were that the model is overall fair as no bias was detected either at the unsupervised or supervised level. This implies fairness in the underlying data and in the model.

## Caveats and Recommendations
No hyperparameter optimization was performed, the model performance could be improved if the optimization was performed.