# !/usr/bin/python
# -*- coding: UTF-8 -*-
import yaml, os, time


class Metadata(yaml.YAMLObject):
    yaml_tag = u'!Metadata'

    def __init__(self, name):
        self.name = name


class Spec(yaml.YAMLObject):
    yaml_tag = u'!Spec'

    def __init__(self, hosts, http):
        self.hosts = hosts
        self.http = http


class Http(yaml.YAMLObject):
    yaml_tag = u'!Http'

    def __init__(self, routes):
        self.route = routes


class Route(yaml.YAMLObject):
    yaml_tag = u'!Route'

    def __init__(self, destination, weight):
        self.destination = destination
        self.weight = weight


class Destination(yaml.YAMLObject):
    yaml_tag = u'!Destination'

    def __init__(self, svcName, subset):
        self.host = svcName
        self.subset = subset


class VirtualService(yaml.YAMLObject):
    yaml_tag = u'!VirtualService'

    def __init__(self, svcName, sw):
        self.apiVersion = 'networking.istio.io/v1alpha3'
        self.kind = 'VirtualService'
        self.metadata = Metadata(svcName)
        routes = []
        for subset, weight in sw.items():
            dest = Destination(svcName, subset)
            route = Route(dest, weight)
            routes.append(route)
        http = Http(routes)
        hosts = [svcName]
        self.spec = Spec(hosts, http)


def noop(self, *args, **kw):
    pass


yaml.emitter.Emitter.process_tag = noop
dict = {'v1': 100, 'v2': 0}
while True:
    vs = VirtualService('ts-voucher-service', dict)
    f = open(r'virtual-services-fault.yaml', 'w')
    yaml.dump(vs, f)
    (status, output) = os.system('kubectl apply -f virtual-services-fault.yaml')
    #status = 0
    if status == 0:
        time.sleep(5)
        if dict['v1'] > 0:
            dict['v1'] = dict['v1'] - 10
            dict['v2'] = dict['v2'] + 10
        elif dict['v1'] == 0:
            dict['v1'] = 100
            dict['v2'] = 0
        else:
            pass
    else:
        raise RuntimeError('output')
