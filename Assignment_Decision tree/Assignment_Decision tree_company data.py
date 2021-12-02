    #!/usr/bin/env python
    # coding: utf-8


import pandas as pd
import sklearn
import scipy
import numpy as np

#seaborn visualization library
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

def decision_tree_model():
    df=pd.read_csv('Company_Data.csv')
    df.dropna(inplace = True)

    '''label encode shelf location'''
    label_encoder = preprocessing.LabelEncoder()
    df['ShelveLoc']= label_encoder.fit_transform(df['ShelveLoc','Urban','US'])
    print(df['ShelveLoc'].unique())


    print(df['US'].unique())

    print(df["Sales"].describe())


    median = np.median(df['Sales'])

    Q3 = np.quantile(df['Sales'], 0.75).round(2)
    Q1 = np.quantile(df['Sales'], 0.25).round(2)
    min=np.min(df['Sales'])
    max=np.max(df['Sales'])

    print(Q1,Q3,min,max)

    df['Sales_level'] = pd.cut(df.Sales,bins=[min,Q1,Q3,max],labels=(['low','medium','high']))

    print(df['Sales_level'].unique())

    df.dropna(inplace = True)


    df['Sales_level']= label_encoder.fit_transform(df['Sales_level'])

    df['Sales_level'].describe
    df.Sales_level.value_counts()

    # scatter plot and correlation analysis
    sns.pairplot(df)





    df.corr()

    x=df.iloc[:,1:-1]
    y=df.iloc[:,-1]

    # Splitting data into training and testing data set
    x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=40)


    #Fit the model
    model = DecisionTreeClassifier(criterion = 'gini',max_depth=6)
    model.fit(x_train,y_train)

    #PLot the decision tree
    tree.plot_tree(model);


    #Predicting on test data
    preds = model.predict(x_test) # predicting on test data set
    pd.Series(preds).value_counts() # getting the count of each category
    pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions

    # Accuracy
    return np.mean(preds==y_test)

if __name__ == "__main__":
    accuracy = decision_tree_model()
    print(accuracy)