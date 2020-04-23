import os

lr = [0.001, 0.01, 0.1]
reg = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
reg_v = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

i = 0
savedir = './saver/Video_Games/'
#os.system('mkdir ' + savedir)

for u in lr:
    for v in reg:
        for w in reg_v:
            savepath = '--savepath ' + savedir + str(i) + '.csv'
            cmd = 'python mf.py --datapath /home/zhanchao/RecommendationSystem/data/dataset_3/Video_Games --learning_rate %f --reg_rate [%f,%f] %s' %(u, v, w, savepath)
            os.system(cmd)
            i += 1