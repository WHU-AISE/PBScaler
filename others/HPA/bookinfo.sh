kubectl autoscale deployment details-v1 --min=1 --max=8 --cpu-percent=90
kubectl autoscale deployment productpage-v1 --min=1 --max=8 --cpu-percent=90
kubectl autoscale deployment ratings-v1 --min=1 --max=8 --cpu-percent=90
kubectl autoscale deployment reviews-v1 --min=1 --max=8 --cpu-percent=90
kubectl autoscale deployment reviews-v2 --min=1 --max=8 --cpu-percent=90
kubectl autoscale deployment reviews-v3 --min=1 --max=8 --cpu-percent=90