import pandas as pd
from scipy.sparse import csr_matrix
filename = '/home/zhanchao/RecommendationSystem/data/dataset_4/Automotive'
df = pd.read_csv(filename + '.csv', names = ['userid', 'itemid', 'ratings'])
# 用户数和项目数
n_users = df.userid.unique().shape[0]
n_items = df.itemid.unique().shape[0]
print(n_users, n_items)

# 遍历数据，生成稀疏矩阵形式
row, col, rating = [], [], []
for line in df.itertuples():
    row.append(line[1])
    col.append(line[2])
    rating.append(line[3])
print(len(row))
matrix = csr_matrix((rating, (row, col)), shape = (n_users, n_items))
print(matrix.nnz)