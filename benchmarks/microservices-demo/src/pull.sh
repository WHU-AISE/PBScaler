docker build -t gcr.io/google-samples/microservices-demo/emailservice:v0.2.2 -f emailservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/checkoutservice:v0.2.2 -f checkoutservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/recommendationservice:v0.2.2 -f emailservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/frontendservice:v0.2.2 -f frontendservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/paymentservice:v0.2.2 -f paymentservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/productcatalogservice:v0.2.2 -f productcatalogservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/loadgenerator:v0.2.2 -f loadgenerator/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/cartservice:v0.2.2 -f cartservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/shippingservice:v0.2.2 -f shippingservice/Dockerfile .
docker build -t gcr.io/google-samples/microservices-demo/currencyservice:v0.2.2 -f currencyservice/Dockerfile .

