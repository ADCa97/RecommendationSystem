import scipy.sparse as sp 
import numpy as np 
import pandas as pd 

class Dataset(object):
    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path)
        self.testRatings = self.load_rating_file_as_list(path + '_test.csv')
        self.testNegatives = self.load_negative_file(path + '_negative.csv')
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape
        print("Dataset init")
    def load_rating_file_as_list(self, filename):
        ratingList = []
        df = pd.read_csv(filename, names = ['userid', 'itemid', 'rating'])
        for line in df.itertuples():
            user, item = line[1], line[2]
            ratingList.append([user, item])
        return ratingList
        
    def load_rating_file_as_matrix(self, filename):
        df = pd.read_csv(filename + '.csv', names = ['userid', 'itemid', 'rating'])
        n_users = df.userid.unique().shape[0]
        n_items = df.itemid.unique().shape[0]
        
        train_df = pd.read_csv(filename + '_train.csv', names = ['userid', 'itemid', 'rating'])
        mat = sp.dok_matrix((n_users, n_items), dtype = np.float32)
        for line in train_df.itertuples():
            userid, itemid, rating = line[1], line[2], line[3]
            if rating > 0:
                mat[userid, itemid] = 1.0
        return mat


    def load_negative_file(self, filename):
        negativeList = []
        df = pd.read_csv(filename,header = None)
        for line in df.itertuples():
            negative = list(line[2:])
            negativeList.append(negative)
        return negativeList

if __name__ == '__main__':
    data = Dataset('../../data/dataset_1/Office_Products')
        