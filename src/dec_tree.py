from sklearn import tree
import numpy as np
class dec_tree():
    def __init__(self, matrix):
        self.matrix=matrix

    def tree():
            
        clf = tree.DecisionTreeClassifier()
        clf=clf.fit(w,vals)
        
        #print(clf)
        #print(vmain.get_feature_names())

        valid,w1=parseval(validation[num])
        pred=[]
        w1=vmain.fit_transform(w1)

        #print(valid)
        for x in range(len(valid)):
            pred.append(clf.predict(w1)[0])

        #print(f1_score(valid, pred, average='micro'))
        #print(pred)
        if '1' in pred:
            print("yes")