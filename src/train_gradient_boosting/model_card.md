# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is created by Md Imdadul Haque Milon. The model is a Fradient Boosting calssifier from Scikit-Learn.

## Intended Use
The trained model is used to predict whether income exceeds $50K/yr based on census data.

## Training Data
The training data for the model is collected from: https://archive.ics.uci.edu/ml/datasets/census+income. The model is trianed on the 80% of the data.

## Evaluation Data
The evaluation data for the model is also collected from: https://archive.ics.uci.edu/ml/datasets/census+income. The model is evaluated on the 20% of the data.

## Metrics
Accuracy metric is used to evaluate the model and the accuracy score was: 0.8646.

## Ethical Considerations
The dataset contains data related to origin country, race and gender and might prone to some sort of discrimination. 

## Caveats and Recommendations
Only male and female genders are provided and more data is required for more general use cases.