import tensorflow as tf
import pandas as pd



CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on


#_______________________________________________________________input_fuction
def input_function(features, labels, training=True, batch_size=256):
    """
    Function to create a tf.data.Dataset object from the features and labels.
    This function will be used to create the input pipeline for the model.
    Args:
        features: A dictionary of feature names and their corresponding values.
        labels: A tensor of labels.
        training: A boolean indicating whether the model is in training mode or not.
        batch_size: The size of the batches to be used in the input pipeline.
    Returns:
        A tf.data.Dataset object containing the features and labels.
    """
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)







#Get Your Data

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

train.head()

"""
And now we are ready to choo se amodel. For classification tasls there are a variety of models to choose from.
Some of the most common ones are:
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVMs)
- Neural Networks
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Gradient Boosting Machines (GBM)
- XGBoost
- LightGBM
- CatBoost
- Ensemble Methods (e.g., Voting Classifier, Stacking)
- Deep Learning Models (e.g., CNNs, RNNs)
- Transformers (for text classification)
- Bayesian Models (e.g., Bayesian Logistic Regression)
- Rule-Based Classifiers (e.g., RIPPER, OneR)
- Gaussian Processes

We can choose any of these but for this example we will use a DNNClassifier. 
This is a deep neural network classifier that is used for classification tasks. 
It is a part of the TensorFlow Estimator API and is designed to work with large 
datasets and high-dimensional feature spaces.
"""
classifier = tf.estimator.DNNClassifier(
    feature_columns=[tf.feature_column.numeric_column(key) for key in CSV_COLUMN_NAMES[:-1]],
    hidden_units=[30, 10],
    n_classes=3,
    model_dir='iris_model'
)