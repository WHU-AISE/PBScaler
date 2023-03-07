> Deploy the Train-Ticket system on K8S with istio.

```
(1) kubectl create -f <(istioctl kube-inject -f ts-deployment-part1.yml)
(2) kubectl create -f <(istioctl kube-inject -f ts-deployment-part2.yml)
(3) kubectl create -f <(istioctl kube-inject -f ts-deployment-part3.yml)
(4) kubectl apply  -f trainticket-gateway.yaml
```

