
rest url:
http://train

build:
mvn clean package

docker:
注意更新DockerFile
cd mongo-cluster
docker run -d --name train-mongo mongo
cd target
docker build -t my/ts-train-service .
docker run -d -p 14567:14567  --name ts-train-service --link train-mongo:train-mongo my/ts-train-service
(mongo-local is in config file: resources/application.yml)

!!!!!notice: please add following lines into /etc/hosts to simulate the network access:
127.0.0.1	ts-train-service