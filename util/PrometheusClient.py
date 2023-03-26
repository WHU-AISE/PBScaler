import time
import requests
import networkx as nx
from config.Config import Config
from util.KubernetesClient import KubernetesClient
import pandas as pd
import numpy as np

class PrometheusClient:
    def __init__(self, config: Config):
        self.config = config
        self.namespace = config.namespace
        self.prom_no_range_url = config.prom_no_range_url
        self.prom_range_url = config.prom_range_url
        self.start = config.start
        self.end = config.end
        self.step = config.step

    def set_time_range(self, start, end):
        self.start = start
        self.end = end

    # execute prom_sql
    def execute_prom(self, prom_url, prom_sql):
        response = requests.get(prom_url,
                                params={'query': prom_sql,
                                        'start': self.start,
                                        'end': self.end,
                                        'step': self.step})
        return response.json()['data']['result']

    def p90(self, svc):
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload=\"%s\"}[1m])) by (destination_workload, le))' % svc
        responses_90 = self.execute_prom(self.prom_no_range_url, prom_90_sql)
        return float(responses_90[0].get('value')[1]) if len(responses_90) > 0 else 0

    def p50(self, svc):
        prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload=\"%s\"}[1m])) by (destination_workload, le))' % svc
        responses_50 = self.execute_prom(self.prom_no_range_url, prom_50_sql)
        return float(responses_50[0].get('value')[1]) if len(responses_50) > 0 else 0

    def get_call(self):
        DG = nx.DiGraph()
        k8s_util = KubernetesClient(self.config)
        svcs = [svc for svc in k8s_util.get_svcs() if 'redis' not in svc and 'mongo' not in svc and 'unknown' not in svc and 'mysql' not in svc and 'rabbitmq' not in svc]
        DG.add_nodes_from(svcs)
        edges = []
        prom_sql = 'sum(istio_tcp_received_bytes_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % self.namespace
        results = self.execute_prom(self.prom_no_range_url, prom_sql)

        prom_sql = 'sum(istio_requests_total{destination_workload_namespace=\"%s\"}) by (source_workload, destination_workload)' % self.namespace
        results = results + self.execute_prom(self.prom_no_range_url, prom_sql)

        for result in results:
            metric = result['metric']
            source = metric['source_workload']
            destination = metric['destination_workload']
            if source in svcs and destination in svcs:
                edges.append((source, destination))

        DG.add_edges_from(edges)
        return DG

    def get_call_latency(self):
        prom_url = self.prom_no_range_url
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, source_workload, le))' % self.namespace
        responses_90 = self.execute_prom(prom_url, prom_90_sql)

        call_latencies = {}

        if len(responses_90) > 0:
            for res in responses_90:
                name = res['metric']['source_workload'] + '_' + res['metric']['destination_workload']
                p90 = float(res.get('value')[1])
                call_latencies[name] = p90

        return call_latencies

    def get_svc_latency(self):
        # p50,p90,p99
        prom_url = self.prom_no_range_url
        prom_50_sql = 'histogram_quantile(0.50, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, le))' % self.namespace
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, le))' % self.namespace
        prom_99_sql = 'histogram_quantile(0.99, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, le))' % self.namespace
        
        responses_50 = self.execute_prom(prom_url, prom_50_sql)
        responses_90 = self.execute_prom(prom_url, prom_90_sql)
        responses_99 = self.execute_prom(prom_url, prom_99_sql)


        svc_latencies = {}

        if len(responses_50) > 0:
            for res in responses_50:
                name = res['metric']['destination_workload'] + '&p50'
                p50 = float(res.get('value')[1])
                svc_latencies[name] = p50
        if len(responses_90) > 0:
            for res in responses_90:
                name = res['metric']['destination_workload'] + '&p90'
                p90 = float(res.get('value')[1])
                svc_latencies[name] = p90
        if len(responses_99) > 0:
            for res in responses_99:
                name = res['metric']['destination_workload'] + '&p99'
                p99 = float(res.get('value')[1])
                svc_latencies[name] = p99

        return svc_latencies

    def get_svc_qps(self):
        qps_dic = {}
        qps_sql = 'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload)' % self.namespace
        response = self.execute_prom(self.prom_no_range_url, qps_sql)

        if len(response) > 0:
            for res in response:
                name = res['metric']['destination_workload'] + '&qps'
                qps = float(res.get('value')[1])
                qps_dic[name] = qps

        return qps_dic

    def get_call_p90_latency_range(self):
        call_df = pd.DataFrame()
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, source_workload, le))' % self.namespace
        responses_90 = self.execute_prom(self.prom_range_url, prom_90_sql)

        def handle(result, call_df, type):
            name = result['metric']['source_workload'] + '_' + result['metric']['destination_workload']
            values = result['values']
            values = list(zip(*values))
            if 'timestamp' not in call_df:
                timestamp = values[0]
                call_df['timestamp'] = timestamp
                call_df['timestamp'] = call_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            key = name
            call_df[key] = pd.Series(metric)
            call_df[key] = call_df[key].astype('float64')

        [handle(result, call_df, 'p90') for result in responses_90]

        return call_df

    def get_svc_p90_latency_range(self):
        latency_df = pd.DataFrame()
        prom_90_sql = 'histogram_quantile(0.90, sum(irate(istio_request_duration_milliseconds_bucket{reporter=\"destination\", destination_workload_namespace=\"%s\"}[1m])) by (destination_workload, destination_workload_namespace, le))' % self.namespace
        responses_90 = self.execute_prom(self.prom_range_url, prom_90_sql)

        def handle(result, latency_df):
            name = result['metric']['destination_workload']
            values = result['values']
            values = list(zip(*values))
            if 'timestamp' not in latency_df:
                timestamp = values[0]
                latency_df['timestamp'] = timestamp
                latency_df['timestamp'] = latency_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            key = name
            latency_df[key] = pd.Series(metric)
            latency_df[key] = latency_df[key].astype('float64')

        [handle(result, latency_df) for result in responses_90]

        return latency_df

    def get_resource_metric_range(self):
        metric_df = pd.DataFrame()
        vCPU_sql = 'sum(rate(container_cpu_usage_seconds_total{job="kubernetes-cadvisor",container!~\'POD|istio-proxy|\',namespace="%s"}[1m]))' % self.namespace
        mem_sql = 'sum(rate(container_memory_usage_bytes{job="kubernetes-cadvisor",container!~\'POD|istio-proxy|\', namespace="%s"}[1m])) / (1024*1024)' % self.namespace
        vCPU = self.execute_prom(self.prom_range_url, vCPU_sql)
        mem = self.execute_prom(self.prom_range_url, mem_sql)

        def handle(result, metric_df, col):
            values = result['values']
            values = list(zip(*values))
            if 'timestamp' not in metric_df:
                timestamp = values[0]
                metric_df['timestamp'] = timestamp
                metric_df['timestamp'] = metric_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            metric_df[col] = pd.Series(metric)
            metric_df[col] = metric_df[col].fillna(0)
            metric_df[col] = metric_df[col].astype('float64')

        [handle(result, metric_df, 'vCPU') for result in vCPU]
        [handle(result, metric_df, 'memory') for result in mem]

        return metric_df

    # Get qps for microservices
    def get_svc_qps_range(self):
        qps_df = pd.DataFrame()
        qps_sql = 'sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload)' % self.namespace
        response = self.execute_prom(self.prom_range_url, qps_sql)

        def handle(result, qps_df):
            name = result['metric']['destination_workload']
            values = result['values']
            values = list(zip(*values))
            if 'timestamp' not in qps_df:
                timestamp = values[0]
                qps_df['timestamp'] = timestamp
                qps_df['timestamp'] = qps_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            qps_df[name] = pd.Series(metric)
            qps_df[name] = qps_df[name].astype('float64')

        [handle(result, qps_df) for result in response]

        return qps_df

    # Get CPU,memory,fs,network for microservices
    def get_svc_metric_range(self):
        df = pd.DataFrame()
        k8s_util = KubernetesClient(self.config)
        cpu_usage_sql = '(sum(rate(container_cpu_usage_seconds_total{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod))' % (
            self.namespace)
        cpu_limit_sql = '(sum(container_spec_cpu_quota{namespace="%s",container!~\'POD|istio-proxy|\'}) by(pod) /100000)' % (
            self.namespace)
        # memory usage MB
        mem_usage_rate_sql = 'sum(rate(container_memory_usage_bytes{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod) / (1024*1024)' % (
            self.namespace)
        mem_usage_sql = 'sum(container_memory_usage_bytes{namespace="%s",container!=\'POD|istio-proxy|\'}) by(pod) / (1024*1024)' % (
            self.namespace)
        mem_limit_sql = 'sum(container_spec_memory_limit_bytes{namespace="%s",container_name!=\'POD|istio-proxy|\'}) by(pod) / (1024*1024)' % (
            self.namespace)
        # file IO usage MB
        fs_usage_sql = '(sum(rate(container_fs_usage_bytes{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod)) / (1024*1024)' % (
            self.namespace)
        fs_write_sql = '(sum(rate(container_fs_write_seconds_total{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod))' % (
            self.namespace)
        fs_read_sql = '(sum(rate(container_fs_read_seconds_total{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod))' % (
            self.namespace)
        # network KB/s
        net_receive_sql = 'sum(rate(container_network_receive_bytes_total{namespace="%s"}[1m])) by (pod) / 1024' % (
            self.namespace)
        net_trainsmit_sql = 'sum(rate(container_network_transmit_bytes_total{namespace="%s"}[1m])) by (pod) / 1024' % (
            self.namespace)

        cpu_usage = self.execute_prom(self.prom_range_url, cpu_usage_sql)
        cpu_limit = self.execute_prom(self.prom_range_url, cpu_limit_sql)
        mem_usage_rate = self.execute_prom(self.prom_range_url, mem_usage_rate_sql)
        mem_usage = self.execute_prom(self.prom_range_url, mem_usage_sql)
        mem_limit = self.execute_prom(self.prom_range_url, mem_limit_sql)
        fs_usage = self.execute_prom(self.prom_range_url, fs_usage_sql)
        fs_write = self.execute_prom(self.prom_range_url, fs_write_sql)
        fs_read = self.execute_prom(self.prom_range_url, fs_read_sql)
        net_receive = self.execute_prom(self.prom_range_url, net_receive_sql)
        net_trainsmit = self.execute_prom(self.prom_range_url, net_trainsmit_sql)

        # indexs = []
        # for i in range(self.config.start, self.config.end + 1):
        #     indexs.append(i)
        # df['timestamp'] = indexs
        # df.set_index('timestamp',inplace=True)

        tempdfs = []

        def handle(result, tempdfs, col):
            name = result['metric']['pod'] + '&' + col
            values = result['values']
            values = list(zip(*values))
            metric = values[1]
            tempdf = pd.DataFrame()
            tempdf['timestamp'] = values[0]
            tempdf[name] = pd.Series(metric)
            tempdf[name] = tempdf[name].astype('float64')
            tempdf.set_index('timestamp',inplace=True)
            tempdfs.append(tempdf)
            # pd.merge(df, tempdf, on='timestamp', how='left')


        [handle(result, tempdfs, 'cpu_usage') for result in cpu_usage]
        [handle(result, tempdfs, 'cpu_limit') for result in cpu_limit]
        [handle(result, tempdfs, 'mem_usage') for result in mem_usage]
        [handle(result, tempdfs, 'mem_usage_rate') for result in mem_usage_rate]
        [handle(result, tempdfs, 'mem_limit') for result in mem_limit]
        [handle(result, tempdfs, 'fs_usage') for result in fs_usage]
        [handle(result, tempdfs, 'fs_write') for result in fs_write]
        [handle(result, tempdfs, 'fs_read') for result in fs_read]
        [handle(result, tempdfs, 'net_receive') for result in net_receive]
        [handle(result, tempdfs, 'net_trainsmit') for result in net_trainsmit]

        df = pd.concat(tempdfs, axis=1)
        df = df.fillna(0)


        # 聚合
        final_df = pd.DataFrame()
        final_df['timestamp'] = df.index
        final_df.set_index('timestamp',inplace=True)
        svcs = k8s_util.get_svcs()
        col_list = list(df)
        for svc in svcs:
            if svc == 'loadgenerator':
                continue
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('cpu_usage')]
            final_df[svc + '&cpu_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('cpu_limit')]
            final_df[svc + '&cpu_limit'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_usage')]
            final_df[svc + '&mem_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_usage_rate')]
            final_df[svc + '&mem_usage_rate'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_limit')]
            final_df[svc + '&mem_limit'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('fs_usage')]
            final_df[svc + '&fs_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('fs_write')]
            final_df[svc + '&fs_write'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('fs_read')]
            final_df[svc + '&fs_read'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('net_receive')]
            final_df[svc + '&net_receive'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('net_trainsmit')]
            final_df[svc + '&net_trainsmit'] = df[svc_cols].sum(axis=1)
        final_df['timestamp'] = final_df.index
        final_df['timestamp'] = final_df['timestamp'].astype('datetime64[s]')
        return final_df

    # Get success rate for microservices
    def get_success_rate_range(self):
        success_df = pd.DataFrame()
        success_rate_sql = '(sum(rate(istio_requests_total{reporter="destination", response_code!~"5.*",namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace) / sum(rate(istio_requests_total{reporter="destination",namespace="%s"}[1m])) by (destination_workload, destination_workload_namespace))' % (
            self.namespace, self.namespace)
        response = self.execute_prom(self.prom_range_url, success_rate_sql)

        def handle(result, success_df):
            name = result['metric']['destination_workload']
            values = result['values']
            values = list(zip(*values))
            if 'timestamp' not in success_df:
                timestamp = values[0]
                success_df['timestamp'] = timestamp
                success_df['timestamp'] = success_df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            success_df[name] = pd.Series(metric)
            success_df[name] = success_df[name].astype('float64')

        [handle(result, success_df) for result in response]

        return success_df

    def get_svc_metric(self):
        df = pd.DataFrame()
        k8s_util = KubernetesClient(self.config)
        cpu_usage_sql = '(sum(rate(container_cpu_usage_seconds_total{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod))' % (
            self.namespace)
        cpu_limit_sql = '(sum(container_spec_cpu_quota{namespace="%s",container!~\'POD|istio-proxy|\'}) by(pod) /100000)' % (
            self.namespace)
        # memory usage MB
        mem_usage_rate_sql = 'sum(rate(container_memory_usage_bytes{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod) / (1024*1024)' % (
            self.namespace)
        mem_usage_sql = 'sum(container_memory_usage_bytes{namespace="%s",container!=\'POD|istio-proxy|\'}) by(pod) / (1024*1024)' % (
            self.namespace)
        mem_limit_sql = 'sum(container_spec_memory_limit_bytes{namespace="%s",container_name!=\'POD|istio-proxy|\'}) by(pod) / (1024*1024)' % (
            self.namespace)
        # file IO usage MB
        fs_usage_sql = '(sum(rate(container_fs_usage_bytes{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod)) / (1024*1024)' % (
            self.namespace)
        fs_write_sql = '(sum(rate(container_fs_write_seconds_total{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod))' % (
            self.namespace)
        fs_read_sql = '(sum(rate(container_fs_read_seconds_total{namespace="%s",container!=\'POD|istio-proxy|\'}[1m])) by(pod))' % (
            self.namespace)
        # network KB/s
        net_receive_sql = 'sum(rate(container_network_receive_bytes_total{namespace="%s"}[1m])) by (pod) / 1024' % (
            self.namespace)
        net_trainsmit_sql = 'sum(rate(container_network_transmit_bytes_total{namespace="%s"}[1m])) by (pod) / 1024' % (
            self.namespace)

        cpu_usage = self.execute_prom(self.prom_no_range_url, cpu_usage_sql)
        cpu_limit = self.execute_prom(self.prom_no_range_url, cpu_limit_sql)
        mem_usage_rate = self.execute_prom(self.prom_no_range_url, mem_usage_rate_sql)
        mem_usage = self.execute_prom(self.prom_no_range_url, mem_usage_sql)
        mem_limit = self.execute_prom(self.prom_no_range_url, mem_limit_sql)
        fs_usage = self.execute_prom(self.prom_no_range_url, fs_usage_sql)
        fs_write = self.execute_prom(self.prom_no_range_url, fs_write_sql)
        fs_read = self.execute_prom(self.prom_no_range_url, fs_read_sql)
        net_receive = self.execute_prom(self.prom_no_range_url, net_receive_sql)
        net_trainsmit = self.execute_prom(self.prom_no_range_url, net_trainsmit_sql)

        def handle(result, df, col):
            name = result['metric']['pod'] + '&' + col
            values = result['value']
            if 'timestamp' not in df:
                timestamp = values[0]
                df['timestamp'] = timestamp
                df['timestamp'] = df['timestamp'].astype('datetime64[s]')
            metric = values[1]
            df[name] = pd.Series(metric)
            df[name] = df[name].astype('float64')

        [handle(result, df, 'cpu_usage') for result in cpu_usage]
        [handle(result, df, 'cpu_limit') for result in cpu_limit]
        [handle(result, df, 'mem_usage') for result in mem_usage]
        [handle(result, df, 'mem_usage_rate') for result in mem_usage_rate]
        [handle(result, df, 'mem_limit') for result in mem_limit]
        [handle(result, df, 'fs_usage') for result in fs_usage]
        [handle(result, df, 'fs_write') for result in fs_write]
        [handle(result, df, 'fs_read') for result in fs_read]
        [handle(result, df, 'net_receive') for result in net_receive]
        [handle(result, df, 'net_trainsmit') for result in net_trainsmit]

        df = df.fillna(0)

        # aggregation
        final_df = pd.DataFrame()
        final_df['timestamp'] = df['timestamp']
        svcs = k8s_util.get_svcs()
        col_list = list(df)
        for svc in svcs:
            if svc == 'loadgenerator':
                continue
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('cpu_usage')]
            final_df[svc + '&cpu_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('cpu_limit')]
            final_df[svc + '&cpu_limit'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_usage')]
            final_df[svc + '&mem_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_usage_rate')]
            final_df[svc + '&mem_usage_rate'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('mem_limit')]
            final_df[svc + '&mem_limit'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('fs_usage')]
            final_df[svc + '&fs_usage'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('fs_write')]
            final_df[svc + '&fs_write'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('fs_read')]
            final_df[svc + '&fs_read'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('net_receive')]
            final_df[svc + '&net_receive'] = df[svc_cols].sum(axis=1)
            svc_cols = [col for col in col_list if col.startswith(svc) and col.endswith('net_trainsmit')]
            final_df[svc + '&net_trainsmit'] = df[svc_cols].sum(axis=1)

        return final_df

    def cal_slo_violation(self, slo: float, apis: list, duration):
        '''
        eg:
            slo: 500
            apis: ['unknown_frontend']
            duration: 30 (seconds)
        '''
        self.end = int(round(time.time()))
        self.start = self.end - duration
        cols = [api+'&p90' for api in apis]
        df = self.get_call_p90_latency_range()[cols]
        con = df >= slo
        conflict_num = df[con].count().sum()
        return conflict_num / df.count().sum()

    def get_edge_index(self):
        # get the dependence
        DG = self.get_call()
        adj = nx.to_scipy_sparse_matrix(DG).tocoo()
        row = adj.row.astype(np.int64)
        col = adj.col.astype(np.int64)
        edge_index = np.stack([row, col], axis=0)
        return edge_index