import re
import dbconnect
import logging
import multiclasssql
import numpy as np
import matplotlib.pyplot as plt
from sys import stdin, stdout, argv, exit
from time import time
from sklearn.ensemble import AdaBoostClassifier

class FastGentleBoosting(object):
    def __init__(self, classifier = None):
        logging.info('Initialized New Fast Gentle Boosting Classifier')
        self.model = None
        self.classBins = []
        self.classifier = classifier

    def CheckProgress(self):
        import wx
        ''' Called when the CheckProgress Button is pressed. '''
        # get wells if available, otherwise use imagenumbers
        try:
            nRules = int(self.classifier.nRulesTxt.GetValue())
        except:
            logging.error('Unable to parse number of rules')
            return

        if not self.classifier.UpdateTrainingSet():
            self.PostMessage('Cross-validation canceled.')
            return

        db = dbconnect.DBConnect.getInstance()
        groups = [db.get_platewell_for_object(key) for key in self.classifier.trainingSet.get_object_keys()]

        t1 = time()
        dlg = wx.ProgressDialog('Computing cross validation accuracy...', '0% Complete', 100, self.classifier, wx.PD_ELAPSED_TIME | wx.PD_ESTIMATED_TIME | wx.PD_REMAINING_TIME | wx.PD_CAN_ABORT)
        base = 0.0
        scale = 1.0

        class StopXValidation(Exception):
            pass

        def progress_callback(amount):
            pct = min(int(100 * (amount * scale + base)), 100)
            cont, skip = dlg.Update(pct, '%d%% Complete'%(pct))
            self.classifier.PostMessage('Computing cross validation accuracy... %s%% Complete'%(pct))
            if not cont:
                raise StopXValidation

        # each round of xvalidation takes about (numfolds * (1 - (1 / num_folds))) time
        step_time_1 = (2.0 * (1.0 - 1.0 / 2.0))
        step_time_2 = (20.0 * (1.0 - 1.0 / 20.0))
        scale = step_time_1 / (10 * step_time_1 + step_time_2)

        xvalid_50 = []

        try:
            for i in range(10):
                # JK - Start Modification
                xvalid_50 += self.XValidate(
                    self.classifier.trainingSet.colnames, nRules, self.classifier.trainingSet.label_matrix,
                    self.classifier.trainingSet.values, 2, groups, progress_callback
                )
                # JK - End Modification

                # each round makes one "scale" size step in progress
                base += scale
            xvalid_50 = sum(xvalid_50) / 10.0

            # only one more step
            scale = 1.0 - base
            # JK - Start Modification
            xvalid_95 = self.XValidate(
                self.classifier.trainingSet.colnames, nRules, self.classifier.trainingSet.label_matrix,
                self.classifier.trainingSet.values, 20, groups, progress_callback
            )
            # JK - End Modification

            dlg.Destroy()
            figure = plt.figure()
            plt.clf()
            plt.hold(True)
            plt.plot(range(1, nRules + 1), 1.0 - xvalid_50 / float(len(groups)), 'r', label='50% cross-validation accuracy')
            plt.plot(range(1, nRules + 1), 1.0 - xvalid_95[0] / float(len(groups)), 'b', label='95% cross-validation accuracy')
            chance_level = 1.0 / len(self.classifier.trainingSet.labels)
            plt.plot([1, nRules + 1], [chance_level, chance_level], 'k--', label='accuracy of random classifier')
            plt.legend(loc='lower right')
            plt.xlabel('Rule #')
            plt.ylabel('Accuracy')
            plt.xlim(1, max(nRules,2))
            plt.ylim(-0.05, 1.05)
            plt.title('Cross-validation accuracy')
            plt.show()
            self.classifier.PostMessage('Cross-validation complete in %.1fs.'%(time()-t1))
        except StopXValidation:
            dlg.Destroy()

    def ClearModel(self):
        self.classBins = []
        self.model = None

    def ComplexityTxt(self):
        return 'Max # of rules: '

    def CreatePerObjectClassTable(self, labels):
        multiclasssql.create_perobject_class_table(labels, self.model)

    def FilterObjectsFromClassN(self, obClass, obKeysToTry):
        return multiclasssql.FilterObjectsFromClassN(obClass, self.model, obKeysToTry)

    def IsTrained(self):
        return self.model is not None

    def LoadModel(self, model_filename):
        import cPickle
        fh = open(model_filename, 'r')
        try:
            self.model, self.bin_labels = cPickle.load(fh)
        except:
            self.model = None
            self.bin_labels = None
            logging.error('The loaded model was not a fast gentle boosting model')
            raise TypeError
        finally:
            fh.close()

    def ParseModel(self, string):
        self.model = []
        string = string.replace('\r\n', '\n')
        for line in string.split('\n'):
            if line.strip() == '':
                continue
            m = re.match('^IF \((\w+) > (-{0,1}\d+\.\d+), \[(-{0,1}\d+\.\d+(?:, -{0,1}\d+\.\d+)*)\], \[(-{0,1}\d+\.\d+(?:, -{0,1}\d+\.\d+)*)\]\)',
                         line, flags=re.IGNORECASE)
            if m is None:
                raise ValueError
            colname, thresh, a, b = m.groups()
            thresh = float(thresh)
            a = map(float, a.split(','))
            b = map(float, b.split(','))
            if len(a) != len(b):
                raise ValueError, 'Alpha and beta must have the same cardinality in "IF (column > threshold, alpha, beta)"'
            self.model.append((colname, thresh, a, b, None))
        n_classes = len(self.model[0][2])
        for wl in self.model:
            if len(wl[2]) != n_classes:
                raise ValueError, 'Number of classes must remain the same between rules.'
        return self.model

    def PerImageCounts(self, filter_name=None, cb=None):
        return multiclasssql.PerImageCounts(self.model, filter_name=filter_name, cb=cb)

    def SaveModel(self, model_filename, bin_labels):
        import cPickle
        fh = open(model_filename, 'w')
        cPickle.dump((self.model, bin_labels), fh)
        fh.close()

    def ShowModel(self):
        '''
        Transforms the weak learners of the algorithm into a human readable
        representation
        '''
        if self.model is not None and self.model is not []:
            return '\n'.join("IF (%s > %s, %s, %s)" %(colname, repr(thresh),
                             "[" + ", ".join([repr(v) for v in a]) + "]",
                             "[" + ", ".join([repr(v) for v in b]) + "]")
                        for colname, thresh, a, b, e_m in self.model)
        else:
            return ''

    def Train(self, colnames, num_learners, label_matrix, values, fout=None, do_prof=False, test_values=None, callback=None):
        self.model = AdaBoostClassifier(n_estimators=5)
        #SKLEARN convert label matrix to numerical class
        classes = np.nonzero(label_matrix == 1)[1] + 1
        self.model.fit(values, classes)
        print classes, self.model


    def UpdateBins(self, classBins):
        self.classBins = classBins

    def Usage(self, name):
        print "usage %s:" % (name)
        print "%s num_learners              - read from stdin, write to stdout" % (name)
        print "%s num_learners file         - read from file, write to stdout" % (name)
        print "%s num_learners file1 file2  - read from file1, write to file2" % (name)
        print ""
        print "Input files should be tab delimited."
        print "Example:"
        print "ClassLabel	Value1_name	Value2_name	Value3_name"
        print "2	0.1	0.3	1.5"
        print "1	0.5	-0.3	0.5"
        print "3	0.1	1.0	0.5"
        print ""
        print "Labels should be positive integers."
        print "Note that if one learner is sufficient, only one will be written."
        exit(1)

    def XValidate(self, colnames, num_learners, label_matrix, values, folds, group_labels, progress_callback):
        # if everything's in the same group, ignore the labels
        if all([g == group_labels[0] for g in group_labels]):
            group_labels = range(len(group_labels))

        # randomize the order of labels
        unique_labels = list(set(group_labels))
        np.random.shuffle(unique_labels)


        fold_min_size = len(group_labels) / float(folds)
        num_misclassifications = np.zeros(num_learners, int)

        # break into folds, randomly, but with all identical group_labels together
        for f in range(folds):
            current_holdout = [False] * len(group_labels)
            while unique_labels and (sum(current_holdout) < fold_min_size):
                to_add = unique_labels.pop()
                current_holdout = [(a or b) for a, b in zip(current_holdout, [g == to_add for g in group_labels])]

            if sum(current_holdout) == 0:
                print "no holdout"
                break

            holdout_idx = np.nonzero(current_holdout)[0]
            current_holdin = ~ np.array(current_holdout)
            holdin_idx = np.nonzero(current_holdin)[0]
            holdin_labels = label_matrix[holdin_idx, :]
            holdin_values = values[holdin_idx, :]
            holdout_values = values[holdout_idx, :]
            holdout_results = self.Train(colnames, num_learners, holdin_labels, holdin_values, test_values=holdout_values)
            if holdout_results is None:
                return None
            # pad the end of the holdout set with the last element
            if len(holdout_results) < num_learners:
                holdout_results += [holdout_results[-1]] * (num_learners - len(holdout_results))
            holdout_labels = label_matrix[holdout_idx, :].argmax(axis=1)
            num_misclassifications += [sum(hr != holdout_labels) for hr in holdout_results]
            if progress_callback:
                progress_callback(f / float(folds))

        return [num_misclassifications]

if __name__ == '__main__':
    fgb = FastGentleBoosting()

    if len(argv) == 2:
        fin = stdin
        fout = stdout
    elif len(argv) == 3:
        fin = open(argv[2])
        fout = stdout
    elif len(argv) == 4:
        fin = open(argv[2])
        fout = open(argv[3], 'w')
    else:
        fgb.usage(argv[0])

    num_learners = int(argv[1])
    assert num_learners > 0

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

    # convert labels to a matrix with +1/-1 values only (+1 in the column matching the label, 1-indexed)
    num_classes = max(labels)
    label_matrix = -np.ones((len(labels), num_classes), np.int32)
    for i, j in zip(range(len(labels)), np.array(labels)-1):
        label_matrix[i, j] = 1

    wl = fgb.Train(colnames, num_learners, label_matrix, values, fout)
    for w in wl:
        print w
    print label_matrix.shape, "groups"
    print fgb.xvalidate(colnames, num_learners, label_matrix, values, 20, range(1, label_matrix.shape[0]+1), None)
