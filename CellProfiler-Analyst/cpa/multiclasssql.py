import numpy as np
import sys
import cpa.sqltools
from dbconnect import DBConnect, UniqueObjectClause, UniqueImageClause, image_key_columns, GetWhereClauseForImages, GetWhereClauseForObjects
from properties import Properties
from datamodel import DataModel
from sklearn.ensemble import AdaBoostClassifier

db = DBConnect.getInstance()
p = Properties.getInstance()
dm = DataModel.getInstance()

temp_stump_table = "_stump"
temp_score_table = "_scores"
temp_class_table = "_class"
filter_table_prefix = '_filter_'

def FilterObjectsFromClassN(classNum, classifier, filterKeys):
    '''
    classNum: 1-based index of the class to retrieve obKeys from
    classifier: trained model of classifier
    filterKeys: (optional) A list of specific imKeys OR obKeys (NOT BOTH)
        to classify.
        * WARNING: If this list is too long, you may exceed the size limit to
          MySQL queries.
        * Useful when fetching N objects from a particular class. Use the
          DataModel to get batches of random objects, and sift through them
          here until N objects of the desired class have been accumulated.
        * Also useful for classifying a specific image or group of images.
    RETURNS: A list of object keys that fall in the specified class,
        if Properties.area_scoring_column is specified, area sums are also
        reported for each class
    '''

    if filterKeys != [] and filterKeys is not None:
        if isinstance(filterKeys, str):
            whereclause = filterKeys + " AND"
        else:
            isImKey = len(filterKeys[0]) == len(image_key_columns())
            if isImKey:
                whereclause = GetWhereClauseForImages(filterKeys) + " AND"
            else:
                whereclause = GetWhereClauseForObjects(filterKeys) + " AND"
    else:
        whereclause = "1=1"

    if p.area_scoring_column:
        data = db.execute('SELECT %s, %s FROM %s WHERE %s'%(UniqueObjectClause(p.object_table),
        ",".join(db.GetColnamesForClassifier()),
        _objectify(p, p.area_scoring_column), p.object_table, whereclause))
        area_score = data[-1] #separate area from data
        data = data[:-1]
    else:
        data = db.execute('SELECT %s, %s FROM %s WHERE %s'%(UniqueObjectClause(p.object_table),
        ",".join(db.GetColnamesForClassifier()), p.object_table, whereclause))

    number_of_features = len(db.GetColnamesForClassifier())

    cell_data = np.array([row[-number_of_features:] for row in data]) #last number_of_features columns in row
    object_keys = np.array([row[:-number_of_features] for row in data]) #all elements in row before last (number_of_features) elements

    predicted_classes = classifier.Predict(cell_data)
    return object_keys[predicted_classes == classNum * np.ones(predicted_classes.shape)]

def _objectify(p, field):
    return "%s.%s"%(p.object_table, field)

def _where_clauses(p, dm, filter_name):
    imkeys = dm.GetAllImageKeys(filter_name)
    imkeys.sort()
    stepsize = max(len(imkeys) / 100, 50)
    key_thresholds = imkeys[-1:1:-stepsize]
    key_thresholds.reverse()
    if len(key_thresholds) == 0:
        return ['(1 = 1)']
    if p.table_id:
        # split each table independently
        def splitter():
            yield "(%s = %d) AND (%s <= %d)"%(_objectify(p, p.table_id), key_thresholds[0][0],
                                              _objectify(p, p.image_id), key_thresholds[0][1])
            for lo, hi in zip(key_thresholds[:-1], key_thresholds[1:]):
                if lo[0] == hi[0]:
                    # block within one table
                    yield "(%s = %d) AND (%s > %d) AND (%s <= %d)"%(_objectify(p, p.table_id), lo[0],
                                                                    _objectify(p, p.image_id), lo[1],
                                                                    _objectify(p, p.image_id), hi[1])
                else:
                    # query spans a table boundary
                    yield "(%s >= %d) AND (%s > %d)"%(_objectify(p, p.table_id), lo[0],
                                                     _objectify(p, p.image_id), lo[1])
                    yield "(%s <= %d) AND (%s <= %d)"%(_objectify(p, p.table_id), hi[0],
                                                      _objectify(p, p.image_id), hi[1])
        return list(splitter())
    else:
        return (["(%s <= %d)"%(_objectify(p, p.image_id), key_thresholds[0][0])] +
                ["(%s > %d) AND (%s <= %d)"
                 %(_objectify(p, p.image_id), lo[0], _objectify(p, p.image_id), hi[0])
                 for lo, hi in zip(key_thresholds[:-1], key_thresholds[1:])])

def PerImageCounts(classifier, num_classes, filter_name=None, cb=None):
    '''
    classifier: trained model of classifier
    filter: name of filter, or None.
    cb: callback function to update with the fraction complete
    RETURNS: A list of lists of imKeys and respective object counts for each class:
        Note that the imKeys are exploded so each row is of the form:
        [TableNumber, ImageNumber, Class1_ObjectCount, Class2_ObjectCount,...]
        where TableNumber is only present if table_id is defined in Properties.
        If p.area_scoring_column is set, then area scores will be appended to
        the object scores.
    '''

    # I'm pretty sure this would be even faster if we were to run two
    # or more parallel threads and split the work between them.
    # For each image clause, classify the cells using the model
    # then for each image key, count the number in each class (and maybe area)
    def do_by_steps(tables, filter_name, area_score=False):
        filter_clause = '1 = 1'
        join_clause = ''
        if filter_name is not None:
            filter = p._filters[filter_name]
            if isinstance(filter, cpa.sqltools.OldFilter):
                join_table = '(%s) as filter' % str(filter)
            else:
                if p.object_table in tables:
                    join_table = None
                else:
                    join_table = p.object_table
                    filter_clause = str(filter)
            if join_table:
                join_clause = 'JOIN %s USING (%s)' % (join_table, ','.join(image_key_columns()))

        wheres = _where_clauses(p, dm, filter_name)
        num_clauses = len(wheres)
        counts = {}

        # iterate over where clauses to go through whole set
        for idx, where_clause in enumerate(wheres):
            if filter_clause is not None:
                where_clause += ' AND ' + filter_clause
            if area_score:
                data = db.execute('SELECT %s, %s, %s FROM %s '
                                  '%s WHERE %s'
                                  %(UniqueImageClause(p.object_table),
                                    ",".join(db.GetColnamesForClassifier()),
                                    _objectify(p, p.area_scoring_column), tables,
                                    join_clause, where_clause),
                                  silent=(idx > 10))
                area_score = data[-1] #separate area from data
                data = data[:-1]
            else:
                data = db.execute('SELECT %s, %s FROM %s '
                                  '%s WHERE %s'
                                  %(UniqueImageClause(p.object_table),
                                    ",".join(db.GetColnamesForClassifier()), tables,
                                    join_clause, where_clause),
                                  silent=(idx > 10))

            number_of_features = len(db.GetColnamesForClassifier())

            cell_data = np.array([row[-number_of_features:] for row in data]) #last number_of_features columns in row
            image_keys = np.array([row[:-number_of_features] for row in data]) #all elements in row before last (number_of_features) elements

            predicted_classes = classifier.Predict(cell_data)
            for i in range(0, len(predicted_classes)):
                row_cls = tuple(np.append(image_keys[i], predicted_classes[i]))
                oneCount = np.array([1])
                if area_score:
                    oneCount = np.append(oneCount, area_score[i])
                if row_cls in counts:
                    counts[row_cls] += oneCount
                else:
                    counts[row_cls] = oneCount

            if cb:
                cb(min(1, idx/float(num_clauses))) #progress
        return counts

    counts = do_by_steps(p.object_table, filter_name, p.area_scoring_column)

    def get_count(im_key, classnum):
        return counts.get(im_key + (classnum, ), np.array([0]))[0]

    def get_area(im_key, classnum):
        return counts.get(im_key + (classnum, ), np.array([0, 0]))[1]

    def get_results():
        for imkey in dm.GetImageKeysAndObjectCounts(filter_name):
            if p.area_scoring_column is None:
                yield list(imkey[0]) + [get_count(imkey[0], cl) for cl in range(1, num_classes+1)]
            else:
                yield list(imkey[0]) + [get_count(imkey[0], cl) for cl in range(1, num_classes+1)] + [get_area(imkey[0], cl) for cl in range(1, num_classes+1)]

    return list(get_results())


if __name__ == "__main__":
    from trainingset import TrainingSet
    from StringIO import StringIO
    import generalclassifier
    from datatable import DataGrid
    import wx
    p = Properties.getInstance()
    db = DBConnect.getInstance()
    dm = DataModel.getInstance()

    props = '/Users/jyhung/work/projects/subpop/analysis/input/2013-01-16_Handtraining_using_only_the_DNA_stain/training_sets/2013_01_17_Anne_AZ_DNA_training/az-dnaonly.properties'
    ts = '/Users/jyhung/work/projects/subpop/analysis/input/2013-01-16_Handtraining_using_only_the_DNA_stain/training_sets/2013_01_17_Anne_AZ_DNA_training/Anne_DNA_66.txt'
    nRules = 5
    filter = None

    classifier = AdaBoostClassifier(n_estimators=nRules)
    GC = generalclassifier.GeneralClassifier(classifier)

    p.LoadFile(props)
    trainingSet = TrainingSet(p)
    trainingSet.Load(ts)
    print trainingSet.labels
    print len(trainingSet.colnames)
    print trainingSet.values.shape
    output = StringIO()
    print 'Training classifier with '+str(nRules)+' rules...'

    GC.Train(trainingSet.labels,trainingSet.values, output)
    num_classes = trainingSet.label_matrix.shape[1]

    '''
    table = PerImageCounts(GC.classifier, num_classes, filter_name=filter)
    table.sort()
    labels = ['table', 'image'] + list(trainingSet.labels) + list(trainingSet.labels)
    for row in table:
        print row'''

    obkey_list = FilterObjectsFromClassN(2, GC.classifier, filterKeys=None)
    for row in obkey_list:
        print row
    #object_scores()
    #create_perobject_class_table()
    #_objectify()
