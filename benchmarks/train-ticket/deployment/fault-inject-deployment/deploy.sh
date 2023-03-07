#!/usr/bin/env bash
kubectl apply -f ./fault-inject-deployment.yaml
kubectl apply -f ./destination-rule-fault.yaml
kubectl apply -f ./virtual-services-fault.yaml