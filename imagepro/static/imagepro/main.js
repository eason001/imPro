$(document).ready(function(){
    $("#terminal-btn").click(function(){
        $("#terminal").toggle();
    });
    $(window).keydown(function(event){
     if(event.keyCode == 13) {
       event.preventDefault();
       return false;
     }
    });
/*    
    $(function(){
      refresh();
    });
*/
    $("input:radio[name=filter]").click(function() {
            var value = $(this).val();
            var image_name;
            if(value == 'grayscale'){
                image_name = "sample_gray.png";
            }
            if(value == 'sobel'){
                image_name = "sample_sobel.png";
            }
            if(value == 'roberts'){
                image_name = "sample_roberts.png";
            }
            if(value == 'canny'){
                image_name = "sample_canny.png";
            }
            if(value == 'contour'){
                image_name = "sample_contour.png";
            }
            if(value == 'skeleton'){
                image_name = "sample_skeleton.png";
            }
	    image_name = 'http://ec2-107-22-17-1.compute-1.amazonaws.com:8000/static/imagepro/' + image_name
             $('#transformed').attr('src', image_name);
        });


});
/*
window.setInterval(function() {
  var elem = document.getElementById('terminal-content');
  elem.scrollTop = elem.scrollHeight;
}, 1000);
*/
//var client = new Faye.Client('http://eda3367a.fanoutcdn.com/bayeux');
//client.subscribe('/test', function (data) {
    //alert('got data: ' + data);
//    $('#output').text(data);
//});
function refresh() {
    $.ajax({
        url: 'http://ec2-107-22-17-1.compute-1.amazonaws.com:8000/imagepro/terminal/',
        success: function(data) {
            $('#terminal-content').html(data);
        }
    });
    setInterval("refresh()", 5000);
}

var app = angular.module("myApp", []).config(function($interpolateProvider) {
  $interpolateProvider.startSymbol('{$');
  $interpolateProvider.endSymbol('$}');
});

app.controller('appCtrl', function($scope, $http) {
    var config = {
                headers : {
                    'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8;'
                },
                xsrfHeaderName: 'X-CSRFToken',
                xsrfCookieName: 'csrftoken'
            }

    $scope.change = function($event) {
        var data = {
            img_path: $scope.img_path
        };
        var config = {
                params: data
            };
        $http.get('http://ec2-107-22-17-1.compute-1.amazonaws.com:8000/?img_path=' + $scope.img_path)
        .success(function(response) {
            $scope.img_info = response.img_info;
        })
        .error(function (data, status, header, config) {
            $scope.state = 'Please select an App or a Batch.';
        });
       // alert($scope.appID);
    };
});
