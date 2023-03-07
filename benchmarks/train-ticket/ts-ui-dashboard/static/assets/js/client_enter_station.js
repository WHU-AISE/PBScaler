var enterStation = new Vue({
    el: '#enterStation',
    data: {
        enter_order_id: '',
        myOrderList: [],
        tempOrderList: [],
        requestTime: 0
    },
    methods: {
        initPage() {
            this.checkLogin();
        },
        checkLogin() {
            var username = sessionStorage.getItem("client_name");
            if (username == null || username == "Not Login") {

                location.href = "client_login.html";
            } else {
                document.getElementById("client_name").innerHTML = username;
                this.queryMyOrderList()
            }
        },
        queryMyOrderList() {
            var myOrdersQueryInfo = new Object();

            myOrdersQueryInfo.loginId = sessionStorage.getItem("client_id");
            myOrdersQueryInfo.enableStateQuery = false;
            myOrdersQueryInfo.enableTravelDateQuery = false;
            myOrdersQueryInfo.enableBoughtDateQuery = false;
            myOrdersQueryInfo.travelDateStart = null;
            myOrdersQueryInfo.travelDateEnd = null;
            myOrdersQueryInfo.boughtDateStart = null;
            myOrdersQueryInfo.boughtDateEnd = null;

            this.myOrderList = [];
            var myOrdersQueryData = JSON.stringify(myOrdersQueryInfo);
            this.queryForMyOrderThree("/api/v1/orderservice/order/refresh", myOrdersQueryData);
            this.queryForMyOrderThree("/api/v1/orderOtherService/orderOther/refresh", myOrdersQueryData);
        },
        queryForMyOrderThree(path, data) {

            var that = this;
            $.ajax({
                type: "post",
                url: path,
                contentType: "application/json",
                dataType: "json",
                data: data,
                headers: {"Authorization": "Bearer " + sessionStorage.getItem("client_token")},
                xhrFields: {
                    withCredentials: true
                },
                success: function (result) {
                    console.log(result);
                    that.tempOrderList = [];
                    var size = result.data.length;
                    var tempIndex = 0;
                    for (var i = 0; i < size; i++) {
                        // filter
                        if (result.data[i].status == 2) {
                            that.tempOrderList[tempIndex] = result.data[i];

                            that.tempOrderList[tempIndex].boughtDate = that.convertNumberToDateTimeString(that.tempOrderList[tempIndex].boughtDate)
                            tempIndex++;
                        }
                    }
                    that.requestTime = that.requestTime + 1;
                    that.myOrderList = that.myOrderList.concat(that.tempOrderList);
                    if (that.requestTime == 2 && that.myOrderList.length == 0) {
                        alert("no ticket to enter statrion")
                    }
                }
            });
        },
        convertNumberToDateTimeString(timeNumber) {
            var date = new Date(Number(timeNumber));
            var year = date.getFullYear(),
                month = date.getMonth() + 1,//月份是从0开始的
                day = date.getDate(),
                hour = date.getHours(),
                min = date.getMinutes(),
                sec = date.getSeconds();

            var newTime = year + '-' +
                (month < 10 ? '0' + month : month) + '-' +
                (day < 10 ? '0' + day : day) + ' ' +
                (hour < 10 ? '0' + hour : hour) + ':' +
                (min < 10 ? '0' + min : min) + ':' +
                (sec < 10 ? '0' + sec : sec);
            return newTime;
        },
        enterStation(orderId) {
            if (orderId != '' && orderId != "") {
                $("#enter_reserve_execute_order_button").attr("disabled", true);
                var executeInfo = new Object();
                executeInfo.orderId = orderId;
                var data = JSON.stringify(executeInfo);
                $.ajax({
                    type: "get",
                    url: "/api/v1/executeservice/execute/execute/" + executeInfo.orderId,
                    contentType: "application/json",
                    dataType: "json",
                    headers: {"Authorization": "Bearer " + sessionStorage.getItem("client_token")},
                    xhrFields: {
                        withCredentials: true
                    },
                    success: function (result) {

                        if (result.status == 1) {
                            alert("server send message: " + result.msg);
                        } else {
                            alert(result.msg);
                        }
                        window.location.reload();
                    },error: function (e) {
                        var message = e.responseJSON.message;
                        console.log(message);
                        if (message.indexOf("Token") != -1) {
                            alert("Token is expired! please login first!");
                        }
                    },
                    complete: function () {
                        $("#enter_reserve_execute_order_button").attr("disabled", false);
                    }
                });
            } else {
                alert("please input your order id first !")
            }
        }
    },
    mounted() {
        this.initPage();
    }
});