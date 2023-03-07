#!/usr/bin/env bash
kubectl delete -f ./virtual-services-fault.yaml
kubectl delete -f ./destination-rule-fault.yaml
kubectl delete -f ./fault-inject-deployment.yaml