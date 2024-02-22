'''
    File name: mlrd_learn.py
    Author: Callum Lock
    Date created: 31/03/2018
    Date last modified: 31/03/2018
    Python Version: 3.6
'''
import joblib
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import model_selection
import sklearn.ensemble as ske
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.tree import DecisionTreeClassifier

# Main code function that trains the random forest algorithm on dataset.
def main():
    print('\n[+] Training MLRD using Random Forest Algorithm, Support Vector Machine, and Decision Tree...')

    # Creates pandas dataframe and reads in csv file.
    df = pd.read_csv('data_file.csv', sep=',')

    # Drops FileName, md5Hash and Label from data.
    X = df.drop(['FileName', 'md5Hash', 'Benign'], axis=1).values

    # Assigns y to label
    y = df['Benign'].values

    # Splitting data into training and test data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the number of training and testing samples.
    print("\n\t[*] Training samples: ", len(X_train))
    print("\t[*] Testing samples: ", len(X_test))

    # Train Random forest algorithm on training dataset.
    clf_rf = ske.RandomForestClassifier(n_estimators=50)
    clf_rf.fit(X_train, y_train)

    # Train Support Vector Machine classifier on training dataset.
    clf_svm = SVC(kernel='linear')
    clf_svm.fit(X_train, y_train)

    # Train Decision Tree classifier on training dataset.
    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X_train, y_train)

    # Perform cross validation and print out accuracy for Random Forest.
    score_rf = model_selection.cross_val_score(clf_rf, X_test, y_test, cv=10)
    print("\n\t[*] Random Forest Cross Validation Score: ", round(score_rf.mean()*100, 2), '%')

    # Calculate f1 score for Random Forest.
    y_train_pred_rf = model_selection.cross_val_predict(clf_rf, X_train, y_train, cv=3)
    f_rf = f1_score(y_train, y_train_pred_rf)
    print("\t[*] Random Forest F1 Score: ", round(f_rf*100, 2), '%')

    # Perform cross validation and print out accuracy for Support Vector Machine.
    score_svm = model_selection.cross_val_score(clf_svm, X_test, y_test, cv=10)
    print("\n\t[*] SVM Cross Validation Score: ", round(score_svm.mean()*100, 2), '%')

    # Calculate f1 score for Support Vector Machine.
    y_train_pred_svm = model_selection.cross_val_predict(clf_svm, X_train, y_train, cv=3)
    f_svm = f1_score(y_train, y_train_pred_svm)
    print("\t[*] SVM F1 Score: ", round(f_svm*100, 2), '%')

    # Perform cross validation and print out accuracy for Decision Tree.
    score_dt = model_selection.cross_val_score(clf_dt, X_test, y_test, cv=10)
    print("\n\t[*] Decision Tree Cross Validation Score: ", round(score_dt.mean()*100, 2), '%')

    # Calculate f1 score for Decision Tree.
    y_train_pred_dt = model_selection.cross_val_predict(clf_dt, X_train, y_train, cv=3)
    f_dt = f1_score(y_train, y_train_pred_dt)
    print("\t[*] Decision Tree F1 Score: ", round(f_dt*100, 2), '%')

    # Visualize accuracy and loss over number of samples for Random Forest
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf_rf, X_train, y_train, cv=10, n_jobs=-1,
                       train_sizes=np.linspace(.1, 1.0, 5),
                       return_times=True)

    # Plot learning curve for Random Forest
    plt.figure()
    plt.title("Random Forest Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    # Save the configuration of the Random Forest classifier and features as a pickle file.
    all_features = X.shape[1]
    features = []

    for feature in range(all_features):
        features.append(df.columns[2+feature])

    try:
        print("\n[+] Saving Random Forest algorithm and feature list in classifier directory...")
        joblib.dump(clf_rf, 'classifier_rf/classifier_rf.pkl')
        open('classifier_rf/features_rf.pkl', 'wb').write(pickle.dumps(features))
        print("\n[*] Random Forest Classifier Saved.")
    except:
        print('\n[-] Error: Random Forest Algorithm and feature list not saved correctly.\n')

    plt.show()

if __name__ == '__main__':
    main()

