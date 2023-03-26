from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def latency_canvas(path, qps_path, SLA):
    latency = pd.read_csv(path + 'latency.csv')['frontend&p90'].tolist()
    qps = pd.read_csv(qps_path)['count'].tolist()
    SLA = [SLA] * len(qps)

    x = list(range(len(qps)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, latency)
    ax1.set_ylabel('Response time(ms)')
    ax1.set_xlabel('Time interal')
    ax1.set_title("Response time vs workload")
    ax1.plot(x, SLA, color = 'r',label="SLA Threshold")
    ax1.plot(x,latency,color = 'g',label="Response time(ms)")
    ax1.set_ylim([0, 4000])
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, qps, color='blue', label='users')
    ax2.set_ylim([0, 800])
    ax2.set_ylabel('Number of users')
    ax2.legend(loc='upper right')

    plt.xlim(-0.5,len(x))
    plt.savefig(path + 'latency.svg')
    plt.close()

def instance_canvas(path, qps_path):
    df = pd.read_csv(path + 'instances.csv')
    col_list= list(df)
    df['count'] = df[col_list].sum(axis=1)
    instances = df['count'].tolist()
    qps = pd.read_csv(qps_path)['count'].tolist()[0:len(instances)]
    x = list(range(len(qps)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Total # of instances')
    ax1.set_xlabel('Time interal')
    ax1.set_title("instances vs workload")
    ax1.plot(x,instances,color = 'g',label="Response time(ms)")
    ax1.set_ylim([0, 100])

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, qps, color='blue')
    ax2.set_ylim([0, 800])
    ax2.set_ylabel('Number of locust threads')

    plt.xlim(-0.5,len(x))
    plt.savefig(path + 'instances.svg')
    plt.close()


def success_canvas(path, qps_path):
    df = pd.read_csv(path + 'success_rate.csv')
    col_list= list(df)
    df['count'] = df[col_list].mean(axis=1)
    instances = df['count'].tolist()
    qps = pd.read_csv(qps_path)['count'].tolist()
    x = list(range(len(qps)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Success rate')
    ax1.set_xlabel('Time interal')
    ax1.set_title("availably vs workload")
    ax1.plot(x,instances,color = 'cyan',label="success rate")
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, qps, color='blue',label='user')
    ax2.set_ylim([0, 800])
    ax2.set_ylabel('Number of locust threads')
    ax2.legend(loc='upper right')

    plt.xlim(-0.5,len(x))
    plt.savefig(path + 'success_rate.svg')
    plt.close()


def instance_cdf(path):
    df = pd.read_csv(path + 'instances.csv')
    col_list= list(df)
    df['count'] = df[col_list].sum(axis=1)
    instances = df['count'].tolist()

    plt.hist(instances,color = 'g',cumulative=True,density=1,histtype='step',range=(11,max(instances)))
    plt.ylabel("CDF",fontsize=20)
    plt.xlabel("Number of Replicas",fontsize=20)
    plt.savefig(path + 'instances_cdf.svg')
    plt.close()


def resource_canvas(path, qps_path):
    vCPU = pd.read_csv(path + 'resource.csv')['vCPU'].tolist()
    memory = pd.read_csv(path + 'resource.csv')['memory'].tolist()

    qps = pd.read_csv(qps_path)['count'].tolist()

    x = list(range(len(qps)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, vCPU)
    ax1.set_ylabel('Resource')
    ax1.set_xlabel('Time interal')
    ax1.set_title("Resource vs workload")
    ax1.plot(x, memory, color = 'r',label="memory")
    ax1.plot(x,vCPU,color = 'g',label="vCPU")
    ax1.set_ylim([0, 2])
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, qps, color='blue',label='user')
    ax2.set_ylim([0, 800])
    ax2.set_ylabel('Number of locust threads')
    ax2.legend(loc='upper right')

    plt.xlim(-0.5,len(x))
    plt.savefig(path + 'resource.svg')
    plt.close()


def draw(path, qps_path):
    latency_canvas(path, qps_path, 500)
    instance_canvas(path, qps_path)
    instance_cdf(path)
    success_canvas(path, qps_path)
    resource_canvas(path, qps_path)
