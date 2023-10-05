import os

import pandas as pd

x_axis = [1,2,3]
loss_list = [4,5,6]
acclost = [2,6,8]

loss_acc_df = pd.DataFrame()
loss_acc_df['x_axis'] = x_axis
loss_acc_df['loss_list'] = loss_list
loss_acc_df['acclost'] = acclost
loss_acc_df.to_csv('loss_acc.csv')