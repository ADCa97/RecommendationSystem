import argparse
import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    print("ratings", i)
    return pd.DataFrame.from_dict(df, orient = 'index')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type = str, default = 'Movies_and_TV')

    return parser.parse_args()

# 将*.json.gz的数据存储为*.csv形式
if __name__ == '__main__':
    args = parse_args()
    src = '../../data/ori_data_5_core/reviews_' + args.filename + '_5.json.gz'
    dst = '../../data/ori_data/ratings_' + args.filename + '.csv'
    df = getDF(src)
    df.to_csv(dst, columns = ['reviewerID', 'asin', 'overall'], header = False, index = False)