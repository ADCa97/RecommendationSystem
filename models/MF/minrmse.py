import pandas as pd 

minrmse = []

for i in range(147):
    datapath = './saver/CDs_and_Vinyl/' + str(i) + '.csv'
    df = pd.read_csv(datapath, names = ['cost', 'rmse'])
    minrmse.append(df['rmse'].min())
    
print("Index#%d, minrmse#%f" %(minrmse.index(min(minrmse)), min(minrmse)))
