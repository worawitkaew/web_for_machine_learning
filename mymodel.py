# Importing the libraries

import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier


name_model = 'mymodel.sav'

def train(data):
    
    y = data["Class"]
        
    x = data.drop(["Class"],axis=1)

    clf = RandomForestClassifier().fit(x,y)
    
    pickle.dump(clf, open(name_model, 'wb'))

def test(test):
    data = pd.read_csv(test)
    print("Hell")
    print(data)

    clf = pickle.load(open(name_model, 'rb'))

    y = data["Class"]
    x = data.drop(["Class"],axis=1)
    print("clf")
    score = clf.score(x, y)
    result = clf.predict(x)
    print(score)
    return result,score