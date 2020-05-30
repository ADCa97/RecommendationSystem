import pandas as pd 

src_minrmse = []
dst_minrmse = []
csv_list = []
all_minrmse = []

for i in range(147):
    datapath = './saver/Software_Video_Games/' + str(i) + '.csv'
    df = pd.read_csv(datapath, names = ['cost', 'src_rmse', 'dst_rmse', 'all_rmse'])
    df = df[(df.src_rmse < 2.119884) & (df.dst_rmse < 3.172037)]
    if not df.empty:
        print(datapath)
        print(df)
        csv_list.append(i)
        all_minrmse.append(df['all_rmse'].min())
print("Index#%d, all_minrmse#%f" %(csv_list[all_minrmse.index(min(all_minrmse))], min(all_minrmse)))
'''
    src_minrmse.append(df['src_rmse'].min())
    dst_minrmse.append(df['dst_rmse'].min())
    
    
print("Index#%d, src_minrmse#%f" %(src_minrmse.index(min(src_minrmse)), min(src_minrmse)))
print("Index#%d, dst_minrmse#%f" %(dst_minrmse.index(min(dst_minrmse)), min(dst_minrmse)))

'''
