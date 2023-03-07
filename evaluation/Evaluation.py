import os
import numpy as np
import pandas as pd

# 计算平均响应时间（P90）
def avg_response_time(path: str):
    df = pd.read_csv(path, index_col=0)
    df = df['frontend&p90']
    return df.mean()

# 判断SLA违反情况
def SLA_conflict(threshold: float, path: str):
    df = pd.read_csv(path, index_col=0).fillna(0)
    # conflict_num = 0
    # all_num = 0
    # for index in df.columns:
    #     all_num += df[index].count()
    #     con = df[index] >= threshold
    #     conflict_num += (df[con][index].shape[0])  
    # return conflict_num / all_num
    con = df['frontend&p90'] >= threshold
    conflict_num = df['frontend&p90'][con].count()
    return conflict_num / df['frontend&p90'].count()

def cal_pod_change(path: str):
    count = 0
    instance_df = pd.read_csv(path)
    instance_df = instance_df.drop(columns=['timestamp']) if 'timestamp' in instance_df.columns else instance_df
    for index, row in instance_df.iterrows():
        if index != 0:
            last_row = instance_df.iloc[index-1]
            a = last_row.eq(row)
            if False in a.values:
                count+=1
    return count

# AWS标准计算resource cost
def resource_cost(path: str):
    '''
        cpu price: 0.00003334 (vCPU/s)
        mem price: 0.00001389 (G/s)
    '''
    df = pd.read_csv(path)
    cpu_cost = (df['vCPU']*5*0.00003334).sum()
    mem_cost = (df['memory']*5*0.00001389).sum()
    return cpu_cost+mem_cost

# avaliable
# 95%<p<=100%    0
# 90%<p<=95%     20% service charge
# 80%<p<=90%     50% service charge
# p<=80%         100% service charge
def avaliable_cost(path: str):
    df = pd.read_csv(path, index_col = 0).fillna(0)
    df = df.iloc[:, 1:]
    # cost = 0
    # for index in df.columns:
    #     con1 = df[index] <= 1.00
    #     con2 = df[index] > 0.95
    #     con3 = df[index] <= 0.95
    #     con4 = df[index] > 0.90
    #     con5 = df[index] <= 0.90
    #     con6 = df[index] > 0.80
    #     con7 = df[index] <= 0.80
    #     cost += ((df[index][con1 & con2].sum() * 0) + (df[index][con3 & con4].sum() * 0.2) + (df[index][con5 & con6].sum() * 0.5) + (df[index][con7].sum() * 1))
    # return cost
    return df['frontend'].mean()

def pod_cost(path: str):
    df = pd.read_csv(path, index_col = 0)
    df = df.fillna(0)
    return df.sum(axis=1).mean()
    
    

# 评估入口
def evaluation(path: str):
    conflict = SLA_conflict(500, path + 'latency.csv')
    res_cost = resource_cost(path + 'resource.csv')
    av_cost = avaliable_cost(path + 'success_rate.csv')
    p_cost = pod_cost(path + 'instances.csv')
    pod_change_counts = cal_pod_change(path + 'instances.csv')

    return conflict, res_cost, av_cost, p_cost, pod_change_counts

#     print('total_cost: %.2f$'%(av_cost + res_cost))


# if __name__ == '__main__':
#     conflicts, res_costs, av_costs, p_costs, p_change_counts = [], [], [], [], []
#     folder_path = '../train_data/boutique/none/swl1/'
#     for f in os.listdir(folder_path):
#         try:
#             temp_c, temp_r, temp_a, temp_p, temp_pcc = evaluation(folder_path + f + '/')
#         except:
#             continue
#         conflicts.append(temp_c)
#         res_costs.append(temp_r)
#         av_costs.append(temp_a)
#         p_costs.append(temp_p)
#         p_change_counts.append(temp_pcc)
#     # max_index = conflicts.index(max(conflicts))
#     # min_index = conflicts.index(min(conflicts))
#     # del_indexes = [max_index, min_index]
#     # # remove the max and min metrics
#     # conflicts = [conflicts[i] for i in range(0, len(conflicts), 1) if i not in del_indexes]
#     # av_costs = [av_costs[i] for i in range(0, len(av_costs), 1) if i not in del_indexes]
#     # p_costs = [p_costs[i] for i in range(0, len(p_costs), 1) if i not in del_indexes]
#     # p_change_counts = [p_change_counts[i] for i in range(0, len(p_change_counts), 1) if i not in del_indexes]

#     # conflict, res_cost, av_cost, p_cost = evaluation(folder_path + '1/')

#     print('SLA CONFLICT: {:.2%}'.format(np.mean(conflicts)))
#     print('resource_cost: %.2f$' % np.mean(res_costs))
#     print('avaliability_cost: %.3f' % np.mean(av_costs))
#     print('pod_mean: %.2f' % np.mean(p_costs))
#     print('pod_change_counts: %.2f' % np.mean(p_change_counts))

