import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from preprocessing import multi_collinearity_check
import matplotlib.pyplot as plt
import seaborn as sns

# Random forest can over fit if the number of features exceeds the number of observations
# To remedy this, use a dimensionality reduction technique like PCA
# For gene expression set, 20k vars --> 184 vars


def rf_rando_best_params(data_df, labels, know_params):
    data = np.array(data_df)
    labels = LabelEncoder().fit_transform(labels)
    train_features, test_features, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=3, stratify=labels)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=3)
    if know_params:
        best = RandomForestClassifier(n_estimators=700, max_depth=41, min_samples_split=2, min_samples_leaf=1,
                                      bootstrap=False)
        scores = cross_val_score(best, train_features, train_labels, cv=cv, scoring='accuracy')
        print('The cross validated training score is %.2f with a spread of %.2f' % (np.mean(scores), np.std(scores)))
        best.fit(train_features, train_labels)
    else:
        rf = RandomForestClassifier()
        # optimizing hyperparameters
        # number of trees ranging from 50 to 1000
        n_estimators = [int(x) for x in np.linspace(500, 900, 7)]
        # max number of features being sqrt(n_features), log2(n_features), or n_features
        # max tree depth ranging from 3 to 50
        max_depth = [int(x) for x in np.linspace(30, 50, 8)]
        max_depth.append(None)
        # minimum number of samples required to split a node
        min_samples_split = [2, 4, 6]
        # minimum number of samples for a leaf node to exist
        min_samples_leaf = [1, 2]
        # bootstrapping or no
        bootstrap = [True, False]
        # instantiating random grid
        random_grid = {'n_estimators': n_estimators, 'max_depth': max_depth,
                       'min_samples_split':
                           min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
        rf_hyper = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=cv, verbose=2,
                                      random_state=3, n_jobs=-1)
        rf_hyper.fit(train_features, train_labels)
        print('The best parameters from cross validated random search are:', rf_hyper.best_params_)
        print('with a training score of %.2f' % rf_hyper.best_score_)
        best = rf_hyper.best_estimator_

    # evaluate the performance of the hyperparameter tuned model
    test_score = best.score(test_features, test_labels)*100
    print('The test score of this RF classifier model is %.2f' % test_score)
    y_pred = best.predict(test_features)
    print(classification_report(test_labels, y_pred))
    return


# LDA PERFORMS POORLY WHEN D >> N, that is why the RF model performs so much better
# Use LDA/QDA when n > 5 x D (lol wtf)

def lda(data, labels, classifier, mcheck):
    # scaling data before modeling, as LDA assumes each input variable has the same variance (identical covar matrix)
    #  We also need to remove multi collinearity from the dataset before proceeding
    # Predictive power can DECREASE with an increase in correlation between variables
    # SO if you are not using PCA transformed data for this, then remove features w VIF > 10
    if mcheck:
        data = multi_collinearity_check(data, clean=True)

    if classifier:
        data = StandardScaler().fit_transform(data)
        labels = LabelEncoder().fit_transform(labels)
    else:
        data = StandardScaler().fit_transform(data)

    # encoding categorical labels before computation, maybe will speed it up?

    categories = np.unique(labels)
    n = len(categories)

    if classifier:
        # creating train/class split, stratified sampling to make up for imbalance in classes

        xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, random_state=3, stratify=labels)

        # instantiating model then evaluating with repeated k fold validation
        # remember kf cv splits data into n parts, using one part as validation for each of n runs
        # this way, each fold is checked against at one point
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=3)
        lda1 = LinearDiscriminantAnalysis(n_components=n-1)
        # hyperparameter tuning with GridSearchCv
        grid = dict()
        grid['solver'] = ['svd', 'lsqr', 'eigen']
        search = GridSearchCV(lda1, grid, scoring='accuracy', n_jobs=-1, cv=cv, verbose=2)
        search.fit(xtrain, ytrain)
        lda1 = search.best_estimator_
        train_score = search.best_score_*100
        print('The training score of the LDA classifier is %.2f' % train_score)
        test_score = lda1.score(xtest, ytest)*100
        print('The testing score of this LDA classifier is %.2f' % test_score)
        y_pred = lda1.predict(xtest)
        print(classification_report(ytest, y_pred))
        return

    else:
        lda1 = LinearDiscriminantAnalysis(n_components=n-1)
        lda1.fit(data, labels)
        transformed = lda1.transform(data)
        return transformed


def logit_classifier(data, labels, know_params, mcheck):
    classes = np.unique(labels)
    # validate that the assumptions of logistic regression have been met, starting with no multi collinearity
    # calculate VIF for each variable, remove if greater than 10
    # turn this off if you have used PCA to make set uncorrelated

    if mcheck:
        data = multi_collinearity_check(data, clean=True)
        data = StandardScaler().fit_transform(data)

    # then create test train split and cross validation repeated k fold
    # but stratify it to ensure that class representation is equal

    xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.1, random_state=3, stratify=labels)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=3)

    # instantiating multinomial logit model
    if know_params:
        best = LogisticRegression(multi_class='multinomial', penalty='l2', solver='newton-cg', C=100)
        scores = cross_val_score(best, xtrain, ytrain, cv=cv, scoring='accuracy')
        print('The cross validated train score is %.2f with a spread of %.2f' % (np.mean(scores), np.std(scores)))
        best.fit(xtrain, ytrain)
    else:
        logit = LogisticRegression(multi_class='multinomial')
        # creating grid of parameters to tune
        grid = dict()
        grid['solver'] = ['newton-cg', 'lbfgs']
        grid['penalty'] = ['l2']
        grid['C'] = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
        # instantiate grid search and fit on training data
        search = GridSearchCV(estimator=logit, param_grid=grid, n_jobs=-1, scoring='accuracy', verbose=2, cv=cv)
        search.fit(xtrain, ytrain)
        # export the best parameters from the grid search to move forward with
        best = search.best_estimator_
        best_score = search.best_score_*100
        print('The tuned hyperparameters are:', best.get_params_,
              'Due to their training score of %.2f' % best_score)

    # evaluate best estimator
    test_score = best.score(xtest, ytest)*100
    print('The final test score of this logit classifier is %.2f' % test_score)
    y_pred = best.predict(xtest)
    print(classification_report(ytest, y_pred))

    # visualize confusion matrix

    cnf = confusion_matrix(ytest, y_pred)
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    sns.heatmap(pd.DataFrame(cnf), annot=True, cmap='Purples', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    return
