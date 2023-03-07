# delete the label of all nodes in the cluster
echo '***Delete the label of all nodes in the cluster***'
kubectl label node `kubectl get node | awk 'NR == 1 {next}{print $1}'` beta.kubernetes.io/fluentd-ds-ready-

# delete all resource files in current directory
echo '***Delete EFK(elasticsearch、fluentd、kibana)***'
kubectl delete -f .
