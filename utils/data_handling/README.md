# 数据处理

`python getDF.py --filename Movies_and_TV`
将../../data/ori_data_5_core/reviews_Movies_and_TV_5.json.gz转储为../../data/ori_data/ratings_Movies_and_TV.csv


`python datahandle.py --src Office_Products --dst Movies_and_TV --savepath ../../data/dataset_1`
从../../data/ori_data/ratings_Office_Products.csv和../../data/ori_data/ratings_Movies_and_TV.csv中提取共享用户，并重新编码userid和itemid

