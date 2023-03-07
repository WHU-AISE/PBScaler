
rest url:
http://pay

always return true
query in json:
{"money":324.50}


build:
mvn clean package

docker:
docker build -t my/payment-service
docker run -p 19001:19001 --link payment-service:payment-service my/payment-service

!!!!!notice: please add following lines into /etc/hosts to simulate the network access:
127.0.0.1	payment-service