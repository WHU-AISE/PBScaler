
/*************************************************************************/
/********************Function For Station Service*************************/
/********************Used For Station Service Single Microservice Test****/

$("#station_update_button").click(function(){
    var stationInfo = new Object();
    stationInfo.id = $("#station_update_id").val();
    if(stationInfo.id == null || stationInfo.id == ""){
        alert("Please input the station id.");
        return;
    }
    stationInfo.name = $("#station_update_name").val();
    if(stationInfo.name == null || stationInfo.name == ""){
        alert("Please input the station name.");
        return;
    }
    stationInfo.stayTime = $("#station_update_stayTime").val();
    if(stationInfo.stayTime == null || stationInfo.stayTime == ""){
        alert("Please input the stay time.");
        return;
    }
    var data = JSON.stringify(stationInfo);
    $("#station_update_button").attr("disabled",true);
    $("#station_update_status").text("false");
    $.ajax({
        type: "post",
        url: "/station/update",
        contentType: "application/json",
        dataType: "text",
        data:data,
        xhrFields: {
            withCredentials: true
        },
        success: function(result){
            if(result == "true"){
                //
            }
            $("#station_update_status").text("true");
        },
        complete: function(){
            $("#station_update_button").attr("disabled",false);
        }
    });
});

$("#station_query_button").click(function(){
    $("#station_query_button").attr("disabled",true);
    $("#single_list_station_status").text("false");
    $.ajax({
        type: "get",
        url: "/station/query",
        contentType: "application/json",
        dataType: "json",
        xhrFields: {
            withCredentials: true
        },
        success: function(result){
            var size = result.length;
            $("#query_station_list_table").find("tbody").html("");
            for(var i = 0;i < size;i++){
                $("#query_station_list_table").find("tbody").append(
                    "<tr>" +
                    "<td>" + result[i]["id"]     + "</td>" +
                    "<td>" + result[i]["name"]     + "</td>" +
                    "<td>" + result[i]["stayTime"]     + "</td>" +
                    "</tr>"
                );
            }
            $("#single_list_station_status").text("true");
        },
        complete: function(){
            $("#station_query_button").attr("disabled",false);
        }
    });
});

//------For Station delete------------
// document.getElementById("station_delete_button").onclick = function post_station_delete(){
//     var stationInfo = new Object();
//     stationInfo.name = $("#station_delete_name").val();
//     var data = JSON.stringify(stationInfo);
//     $.ajax({
//         type: "post",
//         url: "/station/delete",
//         contentType: "application/json",
//         dataType: "json",
//         data:data,
//         xhrFields: {
//             withCredentials: true
//         },
//         success: function(result){
//             $("#station_result").html(JSON.stringify(result));
//         }
//     });
// }
