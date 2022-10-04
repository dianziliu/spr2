from RBPR import *


if __name__ == '__main__':
    trainDataFile = 'run/train.csv'
    testDataFile = 'run/test.csv'
    resultSaveFile = 'Baseline/RBPR_result.txt'
    # read_file_without_scores(trainDataFile)
    iterations = 2000
    cf = RBPR(trainDataFile, resultSaveFile, ',', 10, iterations)
    cf.train_rbpr()