# Loan Prediction

<div align='center'>
    <img src="loan.jpeg" width=500/>
    <br/><br/>
</div>

A loan is a form of debt incurred by an individual or other entity. The lender - usually a corporation, financial institution, or government - advances a sum of money to the borrower. In return, the borrower agrees to a certain set of terms including any finance charges, interest, repayment date, and other conditions. It's better to know for the lender if how likely an individual or an entity will pay the loan back or not in advance. From this angle this is a binary classification problem. That's why our target is to build a Machine Learning model which can predict it with high accuracy.

In this study, the given dataset preprocess was started by removing the null values. Then, all categorical features were encoded. After that, outliers were detected based on the KDE plots of the numerical features and then, outliers were removed from the dataset. Then the dataset splited into train and test sets by using `train_test_split` function of `sklearn`. We've scaled the numerical and the ordinal encoded categorical features by using `StandardScaler`. We've fitted only on the trainin set and transformed both the training and the test sets by using that fit not to cause any data leakage. When we check the distribution of the target in the training set, we've noticed that the ratio of the label 1 (loan was paid) was 69.8% of the overal outcomes. Which shows that the dataset was imbalanced. We've brought it to balance by reducing the number of samples with label 1 by using `RandomUnderSampler` from `imblearn` library. The `sampling_strategy` was set to `majority` which allowed us to undersample the majority class determined by the class with the largest number of examples.

We've evaluated below listed methods by using $10-fold$ cross validation and created a baseline to be used as comparison point:

Linear Algorithms

1. Logistic Regression (LR)
2. Linear Discriminant Analysis (LDA)

Non-linear Algorithms

3. Decision Tree Classifier (DT)
4. $k$-Neighbors Classifier (KNN)
5. Support Vector Classifier (SVC)
6. Gaussian Naive Bayes (GNB)

Ensemble Algorithms 

7. Random Forest Classifier (RFC)
8. AdaBoost Classifier (ABC)
9. Gradient Boosting Classifier (GBC)

And the metrics that were used to evaluate the methods:
- ROC curve and AUC score
- Confusion matrix
- Precision
- Recall
- F1 Score
- Accuracy score

The max accuracy score that we got was slightly above 70% which was given by Gaussian Naive Bayes (GNB) algorithm also provided. The score is below than we anticipated. One reason could be that the number of samples might not be enough for the algorithms to let them make enough generilization. That's why we decided to increase the number of samples by using `SMOTE` method. It allowed us to achieve more than 80% accuracy. The results of Random Forest (RFC) and Gradient Boosting (GBC) were very close to each other. If you check the [comparison plot](Loan_Prediction.ipynb#smoteresults) you may see that the whiskers of RFC spreaded out from the interquartile range and the mean value is very close to the lower quartile (Q1). On the other hand GBC provides more stable results. In anycase, I'd prefer to train and test both then use the one that provides the best test result. When we trained both methods by using entire training dataset RFC provided better better results than the GBC.

We've tuned the hyperparameters as a next step. We used `GridSearchCV` with $5-fold$ cross validation to find the hyperparameters that provides the best accuracy score. There were 270 candidates and the details can be seen below.

````
h_params = {'max_depth': [10, 30, 60, 70, 100],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 800, 1400]}
````

The prediction that was performed on the test set provided slightly more than 80% accuracy with the below hyperparameters which was enough for our original target.

`'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 800`

As a result, Random Forest Classifier is a useful tool to predict if a loan will be paid or not based on the given dataset. 