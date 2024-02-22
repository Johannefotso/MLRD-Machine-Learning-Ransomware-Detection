'''
    File name: mlrd_learn.py
    Author: Callum Lock
    Date created: 31/03/2018
    Date last modified: 31/03/2018
    Python Version: 3.6
'''
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import model_selection
import sklearn.ensemble as ske
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
import joblib

# Main code function that trains the random forest algorithm on dataset.
def main():
    print('\n[+] Training MLRD using Random Forest Algorithm...')

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
    clf = ske.RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)

    # Perform cross validation and print out accuracy.
    score = model_selection.cross_val_score(clf, X_test, y_test, cv=10)
    print("\n\t[*] Cross Validation Score: ", round(score.mean()*100, 2), '%')

    # Calculate f1 score.
    y_train_pred = model_selection.cross_val_predict(clf, X_train, y_train, cv=3)
    f = f1_score(y_train, y_train_pred)
    print("\t[*] F1 Score: ", round(f*100, 2), '%')

    # Plot learning curve
    plt.figure()
    plt.title("Learning Curve")
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

    # Save the configuration of the classifier and features as a pickle file.
    all_features = X.shape[1]
    features = []

    for feature in range(all_features):
        features.append(df.columns[2+feature])

    try:
        print("\n[+] Saving algorithm and feature list in classifier directory...")
        joblib.dump(clf, 'classifier/classifier.pkl')
        open('classifier/features.pkl', 'wb').write(pickle.dumps(features))
        print("\n[*] Saved.")
    except:
        print('\n[-] Error: Algorithm and feature list not saved correctly.\n')

    plt.show()

if __name__ == '__main__':
    main()
