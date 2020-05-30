import os
import pandas as pd

lr = [0.001, 0.01, 0.1]
reg = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
reg_v = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


src = '/home/zhanchao/RecommendationSystem/data/dataset_2/Sports_and_Outdoors'
dst = '/home/zhanchao/RecommendationSystem/data/dataset_2/CDs_and_Vinyl'
share = '/home/zhanchao/RecommendationSystem/data/dataset_2/share_info.csv'
savedir = './saver/Sports_and_Outdoors_CDs_and_Vinyl/'

src_minrmse = 3.098104
dst_minrmse = 2.546115

def runcmd():
    i = 0
    for u in lr:
        for v in reg:
            for w in reg_v:
                savepath = savedir + str(i) + '.csv'
                cmd = ' python umf_v1.py --src %s --dst %s --share %s --learning_rate %f --reg_rate [%f,%f,%f] --savepath %s' %(src, dst, share, u, v, w, w, savepath)
                os.system(cmd)
                i += 1
                if judge(savepath):
                    return
                

def judge(savepath):
    df = pd.read_csv(savepath, names = ['cost', 'src_rmse', 'dst_rmse', 'all_rmse'])
    df = df[(df.src_rmse < src_minrmse) & (df.dst_rmse < dst_minrmse)]
    if not df.empty:
        print(savepath)
        print(df)
        print(df['all_rmse'].min())
        return True
    return False

if __name__ == '__main__':
    runcmd()
