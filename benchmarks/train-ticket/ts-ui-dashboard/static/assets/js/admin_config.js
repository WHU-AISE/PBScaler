var configModule = angular.module("myApp", []);

configModule.factory('loadDataService', function ($http, $q) {

    var service = {};

    service.loadAdminBasic = function (url) {
        var deferred = $q.defer();
        var promise = deferred.promise;
        //返回的数据对象
        var information = new Object();

        $http({
            method: "get",
            url: url,
            headers: {"Authorization": "Bearer " + sessionStorage.getItem("admin_token")},
            withCredentials: true
        }).success(function (data, status, headers, config) {
            if (data.status == 1) {
                information = data;
                deferred.resolve(information);
            }
            else {
                alert("Request the configure list fail!" + data.msg);
            }
        }).error(function (data, header, config, status) {
            alert(data.message)
        });
        return promise;
    };

    return service;
});

configModule.controller("configCtrl", function ($scope, $http, loadDataService, $window) {

    //首次加载显示数据
    loadDataService.loadAdminBasic("/api/v1/adminbasicservice/adminbasic/configs").then(function (result) {
        console.log(result);
        $scope.configs = result.data;
    });

    $scope.deleteConfig = function (config) {
        $('#delete-config-confirm').modal({
            relatedTarget: this,
            onConfirm: function (options) {
                $http({
                    method: "delete",
                    url: "/api/v1/adminbasicservice/adminbasic/configs/" + config.name,
                    headers: {"Authorization": "Bearer " + sessionStorage.getItem("admin_token")},
                    withCredentials: true
                }).success(function (data, status, headers, config) {
                    if (data.status == 1) {
                        alert("Delete config successfully!");
                    } else {
                        alert("Update config failed!");
                    }
                    $window.location.reload();
                }).error(function (data, header, config, status) {
                    alert(data.message)
                });
            },
            // closeOnConfirm: false,
            onCancel: function () {

            }
        });
    };

    $scope.updateConfig = function (config) {
        $('#update-config-name').val(config.name);
        $('#update-config-value').val(config.value);
        $('#update-config-desc').val(config.description);

        $('#update-config-table').modal({
            relatedTarget: this,
            onConfirm: function (options) {
                var data = new Object();
                data.name = $('#update-config-name').val();
                data.value = $('#update-config-value').val();
                data.description = $('#update-config-desc').val();
                // alert(JSON.stringify(data));
                $http({
                    method: "put",
                    url: "/api/v1/adminbasicservice/adminbasic/configs",
                    headers: {"Authorization": "Bearer " + sessionStorage.getItem("admin_token")},
                    withCredentials: true,
                    data: data
                }).success(function (data, status, headers, config) {
                    if (data.status == 1) {
                        alert("Update configure successfully!");
                    } else {
                        alert("Update configure failed!");
                    }
                    $window.location.reload();
                }).error(function (data, header, config, status) {
                    alert(data.message)
                });

            },
            onCancel: function () {

            }
        });
    };

    $scope.addConfig = function () {
        $('#add-config-name').val("");
        $('#add-config-value').val("");
        $('#add-config-desc').val("");

        $('#add-config-table').modal({
            relatedTarget: this,
            onConfirm: function (options) {
                var data = new Object();
                data.name = $('#add-config-name').val();
                data.value = $('#add-config-value').val();
                data.description = $('#add-config-desc').val();
                // alert(JSON.stringify(data));
                $http({
                    method: "post",
                    url: "/api/v1/adminbasicservice/adminbasic/configs",
                    headers: {"Authorization": "Bearer " + sessionStorage.getItem("admin_token")},
                    withCredentials: true,
                    data: data
                }).success(function (data, status, headers, config) {
                    if (data.status == 1) {
                        alert("Add Configure successfully!");
                    } else {
                        alert("Add Configure failed!");
                    }
                    $window.location.reload();
                }).error(function (data, header, config, status) {
                    alert(data.message)
                });

            },
            onCancel: function () {

            }
        });
    };
});