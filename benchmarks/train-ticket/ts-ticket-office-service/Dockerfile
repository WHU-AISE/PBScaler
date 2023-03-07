FROM node

RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' >/etc/timezone

RUN mkdir -p /app
WORKDIR /app

ADD ./node_modules /app/node_modules
ADD ./package.json /app/
ADD ./public /app/public
# RUN npm install

ADD ./bin /app/bin
ADD ./app.js /app/
CMD [ "npm", "start" ]

EXPOSE 16108