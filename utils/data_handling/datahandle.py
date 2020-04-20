import argparse

import numpy as np 
import pandas as pd 
import csv
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type = str, default = 'Office_Products')
    parser.add_argument('--dst', type = str, default = 'Movies_and_TV')
    parser.add_argument('--savepath', type = str, default = '../../data/dataset_1')
    
    return parser.parse_args()

# 提取共享用户的评分信息
def draw_shareuser(src, dst):
    if(os.path.exists('src_temp1.csv')):
        os.remove('src_temp1.csv')
    if(os.path.exists('dst_temp1.csv')):
        os.remove('dst_temp1.csv')
    
    src_set = set()
    dst_set = set()

# 首先获得共享用户集合
    # 源数据集的用户集合
    src_read = csv.reader(open(src, 'r'))
    for row in src_read:
        src_set.add(row[0])

    # 目的数据集的用户集合
    dst_read = csv.reader(open(dst, 'r'))
    for row in dst_read:
        dst_set.add(row[0])

    # 源数据和目的数据集的用户集合的交集，即共享用户集合
    union_set = src_set & dst_set
    print("Src_data:%s\nDst_data:%s\nShared Users#%d" %(src, dst, len(union_set)))
# 然后根据共享用户集合，筛选出所有共享用户的评分信息，即只保留共享用户的评分信息，写入src_temp1.csv和dst_temp1.csv文件中
    src_read = csv.reader(open(src, 'r'))
    with open('src_temp1.csv', 'a', newline = '') as src_out:
        src_write = csv.writer(src_out, dialect = 'excel')
        i = 0
        for row in src_read:
            if row[0] in union_set:
                src_write.writerow(row)
                i += 1
        print('Src Ratings#%d' %i)

    dst_read = csv.reader(open(dst, 'r'))
    with open('dst_temp1.csv', 'a', newline = '') as dst_out:
        dst_write = csv.writer(dst_out, dialect = 'excel')
        i = 0
        for row in dst_read:
            if row[0] in union_set:
                dst_write.writerow(row)
                i += 1
        print('Dst Ratings#%d' %i)

# 重新编码userid和itemid
def recode_userid_itemid():
    if(os.path.exists('src_temp2.csv')):
        os.remove('src_temp2.csv')
    if(os.path.exists('dst_temp2.csv')):
        os.remove('dst_temp2.csv')
    
    userid = 0
    itemid = 0

    userdict = {}
    itemdict = {}

    src_read = csv.reader(open('src_temp1.csv', 'r'))
    with open('src_temp2.csv', 'a', newline = '') as src_out:
        src_write = csv.writer(src_out, dialect = 'excel')
        i = 0
        for row in src_read:
            ori_userid = row[0]
            ori_itemid = row[1]
            if ori_userid not in userdict:
                userdict[ori_userid] = userid
                userid += 1
            if ori_itemid not in itemdict:
                itemdict[ori_itemid] = itemid
                itemid += 1
            i += 1
            row[0] = userdict[ori_userid]
            row[1] = itemdict[ori_itemid]
            src_write.writerow(row)
    print('Src_save Ratings#%d Src_Users#%d Src_Items#%d' %(i, len(userdict), len(itemdict)))

    itemid = 0
    itemdict.clear()

    dst_read = csv.reader(open('dst_temp1.csv', 'r'))
    with open('dst_temp2.csv', 'a', newline = '') as dst_out:
        dst_write = csv.writer(dst_out, dialect = 'excel')
        i = 0
        for row in dst_read:
            ori_userid = row[0]
            ori_itemid = row[1]
            if ori_userid in userdict:
                row[0] = userdict[ori_userid]
                if ori_itemid not in itemdict:
                    itemdict[ori_itemid] = itemid
                    itemid += 1
                row[1] = itemdict[ori_itemid]
                dst_write.writerow(row)
                i += 1
        print('Dst_save Ratings#%d Dst_Users#%d Dst_Items%d' %(i, len(userdict), len(itemdict)))

# 按userid的大小重新排序
def sort_data(src_save, dst_save):
    df = pd.read_csv('src_temp2.csv', names = ['userid', 'itemid', 'ratings'])
    df.sort_values('userid').to_csv(src_save, index = False, header = False)
    df = pd.read_csv('dst_temp2.csv', names = ['userid', 'itemid', 'ratings'])
    df.sort_values('userid').to_csv(dst_save, index = False, header = False)




if __name__ == '__main__':
    ori_datapath = '../../data/ori_data/'

    args = parse_args()
    src = ori_datapath + 'ratings_' + args.src + '.csv'
    dst = ori_datapath + 'ratings_' + args.dst + '.csv'

    src_save = args.savepath + '/' + args.src + '.csv'
    dst_save = args.savepath + '/' + args.dst + '.csv'


    draw_shareuser(src, dst)
    recode_userid_itemid()
    sort_data(src_save, dst_save)

    os.remove('src_temp1.csv')
    os.remove('dst_temp1.csv')
    os.remove('src_temp2.csv')
    os.remove('dst_temp2.csv')