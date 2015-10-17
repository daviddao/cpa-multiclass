import re
import dbconnect
import logging
import multiclasssql
import numpy as np
import matplotlib.pyplot as plt
from sys import stdin, stdout, argv, exit
from time import time
from sklearn import ensemble, naive_bayes, grid_search, svm, lda, qda, tree, multiclass, linear_model, neighbors
#from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
import cPickle, json
from sklearn.externals import joblib

'''
TODO missing functions for class algorithm
Show model
GetComplxTxt <- Get params
'''

class GeneralClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier = "lda.LDA()"):
        logging.info('Initialized New Classifier: ' + classifier)
        self.name = "" # Name for User Information
        self.classBins = []
        self.classifier = eval(classifier)
        self.trained = False


    # A converter function for labels
    def label2np(self, labels):
        ## Parsing original label_matrix into numpy format 
        ## Original [-1 1] -> [0 1] (take only second?)
        return np.nonzero(labels + 1)[1] + 1

    # Used to plot results in a seperate matlab lib
    def PlotResults(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn import svm
        #from sklearn.linear_model import SGDClassifier

        # we create 40 separable points
        rng = np.random.RandomState(0)
        n_samples_1 = 1000
        n_samples_2 = 100
        X = np.r_[1.5 * rng.randn(n_samples_1, 2),
                  0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
        y = [0] * (n_samples_1) + [1] * (n_samples_2)

        # fit the model and get the separating hyperplane
        clf = svm.SVC(kernel='linear', C=1.0)
        clf.fit(X, y)

        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - clf.intercept_[0] / w[1]


        # get the separating hyperplane using weighted classes
        wclf = svm.SVC(kernel='linear', class_weight={1: 10})
        wclf.fit(X, y)

        ww = wclf.coef_[0]
        wa = -ww[0] / ww[1]
        wyy = wa * xx - wclf.intercept_[0] / ww[1]

        # plot separating hyperplanes and samples
        h0 = plt.plot(xx, yy, 'k-', label='no weights')
        h1 = plt.plot(xx, wyy, 'k--', label='with weights')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.legend()

        plt.axis('tight')
        plt.show()

    def CheckProgress(self):
        import wx
        ''' Called when the Cross Validation Button is pressed. '''
        # get wells if available, otherwise use imagenumbers
        try:
            nRules = int(self.classifier.nRulesTxt.GetValue())
        except:
            logging.error('failed crossvalidation')
            return

        if not self.classifier.UpdateTrainingSet():
            self.PostMessage('Cross-validation canceled.')
            return

        db = dbconnect.DBConnect.getInstance()
        groups = [db.get_platewell_for_object(key) for key in self.classifier.trainingSet.get_object_keys()]

        t1 = time()
        dlg = wx.ProgressDialog('Nothing', '0% Complete', 100, self.classifier, wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_REMAINING_TIME | wx.PD_CAN_ABORT)

    def ClassificationReport(self, true_labels, predicted_labels, confusion_matrix=False):
        #metrics.confusion_matrix(true_labels, predicted_labels)
        #unique_labels = np.unique(true_labels)
        #n_classes = len(unique_labels)
        #scorePerClass = np.zeros(n_classes)

        #for label_count, i in enumerate(unique_labels):
        #    indices = (true_labels == (i))
        #    predicted_correct = (predicted_labels[indices] == (i))
        #    scorePerClass[label_count] = 1.0*sum(predicted_correct)/sum(indices)

        #print scorePerClass.mean()
        return metrics.classification_report(true_labels, predicted_labels)

    def ClearModel(self):
        self.classBins = []
        self.trained = False

    # Adapter to SciKit Learn
    def ComplexityTxt():
        return str(self.get_params())

    def CreatePerObjectClassTable(self, classNames):
        multiclasssql.create_perobject_class_table(self, classNames)

    def FilterObjectsFromClassN(self, obClass, obKeysToTry):
        return multiclasssql.FilterObjectsFromClassN(obClass, self, obKeysToTry)

    def IsTrained(self):
        return self.trained

    def LoadModel(self, model_filename):

        try:
            self.classifier, self.bin_labels, self.name = joblib.load(model_filename)
        except:
            self.classifier = None
            self.bin_labels = None
            logging.error('Model not correctly loaded')
            raise TypeError


    def LOOCV(self, labels, values, details=False):
        labels = self.label2np(labels);
        '''
        Performs leave one out cross validation.
        Takes a subset of the input data label_array and values to do the cross validation.
        RETURNS: array of length folds of cross validation scores,
        detailedResults is an array of length # of samples containing the predicted classes
        '''
        num_samples = values.shape[0]
        scores = np.zeros(num_samples)
        detailedResults = np.zeros(num_samples)
        # get training and testing set, train on training set, score on test set
        for train, test in cross_validation.LeaveOneOut(num_samples):
            values_test = values[test]
            label_test = labels[test]
            self.Train(labels[train], values[train], fout=None)
            scores[test] = self.classifier.score(values_test, label_test)
            if details:
                detailedResults[test] = self.Predict(values_test)
        if details:
            return scores, detailedResults
        return scores

    def PerImageCounts(self, number_of_classes, filter_name=None, cb=None):
        return multiclasssql.PerImageCounts(self, number_of_classes, filter_name, cb)

    def Predict(self, test_values, fout=None):
        '''RETURNS: np array of predicted classes of input data test_values '''
        predictions = self.classifier.predict(test_values)
        if fout:
            print predictions
        return np.array(predictions)

    def SaveModel(self, model_filename, bin_labels):
        joblib.dump((self.classifier, bin_labels, self.name), model_filename, compress=9)

    def ShowModel(self):#SKLEARN TODO
        '''
        Transforms the weak learners of the algorithm into a human readable
        representation
        '''

    def Train(self, labels, values, fout=None):
        '''Trains classifier using values and label_array '''

        labels = self.label2np(labels)

        self.classifier.fit(values, labels)
        self.trained = True
        if fout:
            print self.classifier

    def UpdateBins(self, classBins):
        self.classBins = classBins

    def Usage(self):
        print "usage :"
        print " classifier              - read from stdin, write to stdout"
        print " classifier file         - read from file, write to stdout"
        print " classifier file1 file2  - read from file1, write to file2"
        print ""
        print "Input files should be tab delimited."
        print "Example:"
        print "ClassLabel	Value1_name	Value2_name	Value3_name"
        print "2	0.1	0.3	1.5"
        print "1	0.5	-0.3	0.5"
        print "3	0.1	1.0	0.5"
        print ""
        print "Class labels should be integers > 0."
        exit(1)

    def XValidate(self, labels, values, folds, stratified=False, scoring=None):
        '''
        Performs K fold cross validation based on input folds.
        Takes a subset of the input data label_array and values to do the cross validation.
        RETURNS: array of length folds of cross validation scores
        '''

        labels = self.label2np(labels)


        num_samples = values.shape[0]
        if stratified:
            CV = folds
        else:
            CV = cross_validation.KFold(num_samples, folds)
        #scores = cross_validation.cross_val_score(self.classifier, scoring=scoring, X=values, y=labels, cv=CV, n_jobs=-1, verbose=1)
        scores = cross_validation.cross_val_score(self.classifier, X=values, y=labels, cv=folds, n_jobs=1)

        return np.array(scores)

    def XValidateBalancedClasses(self, labels, values, folds):
        '''
        :param labels: class of each sample
        :param values: feature values for each sample
        :param folds: number of folds
        :return: score for each fold
        '''

        labels = self.label2np(labels)


        n_samples = values.shape[0]
        unique_labels, indices = np.unique(labels, return_inverse=True)
        label_counts = np.bincount(indices) #count of each class
        min_labels = np.min(label_counts) #possibly make this flexible
        n_classes = len(unique_labels)
        cumSumLabelIndices = np.append(np.array(0), np.cumsum(label_counts))

        #make new data set
        #randomly choose min_labels samples from each class (the rest are thrown away)
        chosenIndices = [np.random.choice(range(cumSumLabelIndices[i],cumSumLabelIndices[i+1]), min_labels, replace=False) for i in range(n_classes)]

        labels_s = np.zeros(min_labels*n_classes)
        values_s = np.zeros((min_labels*n_classes, values.shape[1]))
        for c in reversed(range(n_classes)):
            labels_s[min_labels*c:min_labels*(c+1)] = labels[chosenIndices[c]]
            values_s[min_labels*c:min_labels*(c+1)] = values[chosenIndices[c]]

        #do k fold cross validation on this newly balanced data
        return self.XValidate(labels_s, values_s, folds, stratified=True)

    def XValidatePredict(self, labels, values, folds, stratified=False):
        '''
        :param labels: class of each sample
        :param values: feature values for each sample
        :param folds: number of folds
        :param stratified: boolean whether to use stratified K fold
        :return: cross-validated estimates for each input data point
        '''

        labels = self.label2np(labels)

        num_samples = values.shape[0]
        if stratified:
            CV = folds
        else:
            CV = cross_validation.KFold(num_samples, folds)

        predictions = cross_validation.cross_val_predict(self.classifier, X=values, y=labels, cv=CV, n_jobs=1)
        return np.array(predictions)


if __name__ == '__main__':

    classifier = GeneralClassifier(eval(argv[1]))
    if len(argv) == 2:
        fin = stdin
        fout = stdout
    elif len(argv) == 3:
        fin = open(argv[2])
        fout = stdout
    elif len(argv) == 4:
        fin = open(argv[2])
        fout = open(argv[3], 'w')
    elif len(argv) > 4:
        classifier.Usage()

    import csv
    reader = csv.reader(fin, delimiter='	')
    header = reader.next()
    label_to_labelidx = {}
    curlabel = 1

    def getNumlabel(strlabel):
        if strlabel in label_to_labelidx:
            return label_to_labelidx[strlabel]
        global curlabel
        print "LABEL: ", curlabel, strlabel
        label_to_labelidx[strlabel] = curlabel
        curlabel += 1
        return label_to_labelidx[strlabel]

    colnames = header[1:]
    labels = []
    values = []
    for vals in reader:
        values.append([0 if v == 'None' else float(v) for v in vals[1:]])
        numlabel = getNumlabel(vals[0])
        labels.append(numlabel)

    labels = np.array(labels).astype(np.int32)
    values = np.array(values).astype(np.float32)

    #scores = classifier.XValidate(labels, values, folds=20, stratified=True)
    #scores = classifier.XValidateBalancedClasses(labels, values, folds=5)
    scorePerClass = classifier.ScorePerClass(labels, classifier.XValidatePredict(labels,values,folds=20, stratified=True))