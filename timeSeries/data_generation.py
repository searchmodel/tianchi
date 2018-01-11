#!/usr/bin/env python
import numpy as np
import pandas as pd

#简单统计每个class_id的历史销售量情况（每月的销量是一条记录）
df = pd.read_csv('../data/resource/yancheng_train_20171226.csv')
class_quantity = df.groupby(['class_id','sale_date'])['sale_quantity'].sum()
count = 0
file_name = class_quantity.index[0][0]
quantity  = []
for i in range(len(class_quantity)):
  if class_quantity.index[i][0] != file_name:
    quantity = np.asarray(quantity)
    np.save('../data/processed/ts/'+str(file_name)+'_'+str(len(quantity))+'.npy',quantity)
    print "len of "+str(file_name)+': ', len(quantity)
    if len(quantity)>36:
      count+=1
    file_name = class_quantity.index[i][0]
    quantity = []
  assert class_quantity.values[i]>0
  quantity.append(class_quantity.values[i])
np.save('../data/processed/ts/'+str(file_name)+'_'+str(len(quantity))+'.npy',quantity)
print str(count)+' classes have more than 36 records'
  
