# PBScaler

### PBScaler: A Bottleneck-aware Autoscaling Framework for Microservice-based Applications

PBScaler is a bottleneck-aware autoscaling framework designed to prevent performance
degradation in microservice-based application.

## Enviroment
| Software | Version|
|  ----  | ----  |
| Kubernetes  | 1.20.4 |
| Istio  | 1.13.4 |


<B>Deploy benchmarks</B>

1. [Train Ticket](https://github.com/FudanSELab/train-ticket.git)
```shell
cd benchmarks/train-ticket/deployment/kubernetes-manifests/quickstart-k8s

# Deploy the databases
kubectl apply -f quickstart-ts-deployment-part1.yml
# Deploy the services
kubectl apply -f quickstart-ts-deployment-part2.yml
# Deploy the UI Dashboard
kubectl apply -f quickstart-ts-deployment-part3.yml
```
2. [Online-boutique](https://github.com/GoogleCloudPlatform/microservices-demo)
```shell
cd benchmarks/microservices-demo/release/
kubectl apply -f kubernetes-manifests.yaml
```


## Getting Started
<B>Clone the Repo</B>
```
git clone --depth=1 https://github.com/WHU-AISE/PBScaler.git
```

<B>Install Dependencies</B>
```
pip install -r requirements.txt
```


<B>Write Configuration</B>

Make these changes based on your local configuration and environment
```python
import time

def getNowTime():
    return int(round(time.time()))

class Config():
    def __init__(self):

        self.namespace = 'default'
        # self.nodes = {
        #     'ubuntu-Precision-Tower-7810': '192.168.31.202:9100',
        #     'dell': '192.168.31.201:9100',
        #     'node1': '192.168.31.136:9100'
        # }

        self.SLO = 200
        # maximum and minimum number of pods for a microservice
        self.max_pod = 8
        self.min_pod = 1

        # k8s config path
        self.k8s_config = '/home/ubuntu/xsy/config'
        self.k8s_yaml = '/home/ubuntu/xsy/microservices-demo/release/kubernetes-manifests.yaml'
        
        # experiment duration
        self.duration = 1 * 20 * 60 # 20 min
        self.start = getNowTime()
        self.end = self.start + self.duration

        # prometheus
        self.prom_range_url = "http://192.168.31.202:32030/api/v1/query_range"
        self.prom_no_range_url = "http://192.168.31.202:32030/api/v1/query"
        self.step = 5
```

<B>Train a SLO violation preditor</B>

Before doing this, you'll need to recollect historical data for the cluster. The data in the simulation folder is just the samples 
``` shell
cd simulation
python RandomForestClassify.py
```

<B>Run</B>

```python
python main.py
```
