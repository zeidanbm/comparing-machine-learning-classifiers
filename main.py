from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# Define some constants to be used later
SEED = 72
YELLOW = '\033[93m'
RED = '\033[91m'
END = '\033[0m'

# Import the data
data = pd.read_csv("tripadvisor_reviews.csv")
X = np.array(data['Review'])
y = np.array(data['Rating'])


# ###########################################################################
# Data Pre-processing
print(f"{RED}--------Data Pre-processing--------{END}")

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

# Term Frequency times Inverse Document Frequency
tfidf = TfidfVectorizer(min_df=5, max_df=0.8, use_idf=True)

X_train_tf = tfidf.fit_transform(X_train)
print(f'Training Data Shape: {X_train_tf.shape}')
X_test_tf = tfidf.transform(X_test)
print(f'Testing Data Shape: {X_test_tf.shape}')

# Sort the tf-idf vectors by descending order of scores
df = pd.DataFrame(X_train_tf[0].T.todense(), index=tfidf.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)

# Plot top 10 words with highest Tf-idf score
df.iloc[:10].plot(kind='bar')
plt.show()


# ###########################################################################
# Helper Functions
def run_grid_search(classifier, param_grid, cv=5, verbose=0):
    print(f"Running Grid Search with {cv}-fold cross validation")
    t0 = time()
    model = GridSearchCV(
        classifier, param_grid, cv=cv, n_jobs=-1, verbose=verbose
    )
    model = model.fit(X_train_tf, y_train)
    print(f"{YELLOW}Grid Search done{END} in %0.3fs" % (time() - t0))
    print(f"{YELLOW}Best Params{END} found by grid search:")
    print(model.best_params_)
    return model


def predict_and_evaluate(model, test_data):
    t0 = time()
    y_pred = model.predict(test_data)
    print(f"{YELLOW}Testing done{END} in %0.3fs" % (time() - t0))
    print(f'{YELLOW}Confusion Matrix{END}')
    print(confusion_matrix(y_test, y_pred))
    total_errs = np.sum(y_test != y_pred)
    error_percent = (total_errs / len(y_test)) * 100
    print(f"{YELLOW}Classification Error Percentage{END}: %0.2f%%" % error_percent)
    return y_pred


def final_training(model):
    t0 = time()
    model.best_estimator_.fit(X_train_tf, y_train)
    print(f"{YELLOW}Final Training done{END} in %0.3fs" % (time() - t0))
    return model


def plot_roc(model, test_labels, title):
    probs = model.predict_proba(test_labels)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1], pos_label=1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title(title)
    plt.show()


# ###########################################################################
# KNN Classifier
# print(f"{RED}--------KNN Classifier--------{END}")
# search_params = {
#     "n_neighbors": [3, 5, 7, 9, 11]
# }
# model = run_grid_search(
#     KNeighborsClassifier(), search_params
# )
#
# # Final Training
# model = final_training(model)
#
# # Prediction & Evaluation on the test data
# predict_and_evaluate(model, X_test_tf)
#
# # Plot ROC Curve
# plot_roc(model, X_test_tf, title='ROC for KNN')


# ###########################################################################
# AdaBoost Classifier
print(f"{RED}--------AdaBoost Classifier--------{END}")
search_params = {
    "n_estimators": [20, 30, 50, 70, 90],
    "learning_rate": [0.2, 0.4, 0.6, 0.8, 1]
}
model = run_grid_search(
    AdaBoostClassifier(), search_params
)

# Final Training
model = final_training(model)

# Prediction & Evaluation on the test data
predict_and_evaluate(model, X_test_tf)

# Plot ROC Curve
plot_roc(model, X_test_tf, title='ROC for AdaBoost')


# ###########################################################################
# SVM Classifier
# print(f"{RED}--------SVM Classifier--------{END}")
# search_params = {
#     'gamma': [0.0001, 0.01, 0.1, 0.5, 1]
# }
# model = run_grid_search(
#     SVC(kernel='rbf', class_weight='balanced', probability=True, C=1000), search_params
# )
#
# # Final Training
# model = final_training(model)
#
# # Prediction & Evaluation on the test data
# predict_and_evaluate(model, X_test_tf)
#
# # Plot ROC Curve
# plot_roc(model, X_test_tf, title='ROC for SVM')
#
# print(f'{RED}Done{END}')
