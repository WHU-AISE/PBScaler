# Deployment

## 0. prepare your NFS SERVER
```bash
# ubuntu
$$ sudo apt install nfs-kernel-server

$$ sudo vi /etc/exports
# ADD
<YOUR-NFS-DATA-PATH> *(rw,sync,no_subtree_check,no_root_squash)

$$ sudo mkdir -p <YOUR-NFS-DATA-PATH>

$$ sudo service nfs-kernel-server restart

## check nfs staus
$$ sudo showmount -e <YOUR-NFS-SERVER>
YOUR-NFS-DATA-PATH
```

## 1. prepare NFS dir
> NOTE: Before running, need to create <YOUR-NFS-DATA-PATH>/pv-1g-{n} at <YOUR-NFS-SERVER>, otherwise, NFS will fail to mount with return code 32.

```bash
$$ pwd
<YOUR-NFS-DATA-PATH>
$$ sudo seq -f "<YOUR-NFS-DATA-PATH>/pv-1g-%01g" 1 22| xargs mkdir -p

```


## 2. deployment trainticket

```bash
$$ kubectl apply -f quickstart-ts-deployment-part1.yml

$$ kubectl apply -f quickstart-ts-deployment-part2.yml

$$ kubectl apply -f quickstart-ts-deployment-part3.yml
```
