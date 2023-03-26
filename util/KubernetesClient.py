import os
from kubernetes import client, config

from config.Config import Config


class KubernetesClient():
    def __init__(self, project_config: Config):
        self.namespace = project_config.namespace
        self.k8s_yaml = project_config.k8s_yaml
        config.kube_config.load_kube_config(config_file=project_config.k8s_config)
        self.core_api = client.CoreV1Api()  # namespace,pod,service,pv,pvc
        self.apps_api = client.AppsV1Api()  # deployment

    # Get all microservices
    def get_svcs(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        svcs = [i.metadata.name for i in ret.items if i.metadata.name != 'loadgenerator']
        svcs.sort()
        return svcs

    # Get stateless microservices（exclude redis，mq，mongo，db）
    def get_svcs_without_state(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        def judge_state_svc(svc):
            state_svcs = ['redis', 'rabbitmq', 'mongo', 'mysql']
            for state_svc in state_svcs:
                if state_svc in svc:
                    return True
            return False
        svcs = [i.metadata.name for i in ret.items if not judge_state_svc(i.metadata.name)]
        svcs.sort()
        return svcs


    def get_svcs_counts(self):
        dic = {}
        pod_ret=self.core_api.list_namespaced_pod(self.namespace, watch=False)
        svcs = self.get_svcs()
        for svc in svcs:
            dic[svc] = 0
            for i in pod_ret.items:
                if i.metadata.name.find(svc)!=-1:
                    dic[svc] = dic[svc] + 1
        return dic

    def get_svc_count(self, svc):
        ret_deployment = self.apps_api.read_namespaced_deployment_scale(svc, self.namespace)
        return ret_deployment.spec.replicas

    def all_avaliable(self):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        for item in ret.items:
            if item.status.ready_replicas != item.spec.replicas:
                return False
        return True

    # Determine the status of the service (avaliable?)
    def svcs_avaliable(self, svcs):
        ret = self.apps_api.list_namespaced_deployment(self.namespace)
        items = [item for item in ret.items if item.metadata.name == 'svc']
        for item in ret.items:
            if item.metadata.name in svcs and item.status.ready_replicas != item.spec.replicas:
                return False
        return True

    def patch_scale(self, svc, count):
        body = {'spec': {'replicas': count}}
        self.apps_api.patch_namespaced_deployment_scale(svc, self.namespace, body)

    def update_yaml(self):
        os.system('kubectl apply -f %s > temp.log' % self.k8s_yaml)
