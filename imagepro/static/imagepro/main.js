$(document).ready(function(){
    $("#terminal-btn").click(function(){
        $("#terminal").toggle(500);
    });

    $("#terminal-close").click(function(){
        $("#terminal").hide(500);
    });

    $(window).keydown(function(event){
     if(event.keyCode == 13) {
       event.preventDefault();
       return false;
     }
    });

  var elem = document.getElementById('terminal-body');
  elem.scrollTop = elem.scrollHeight;
  max_bg = 4;
  bg_1 = getRandomInt(0,max_bg);
  $('.bg-1').css('background-image', 'url(' + '/static/imagepro/main_bg' + bg_1 + ')');
/*  bg_2 = getRandomInt(0,max_bg);
  $('.bg-2').css('background-image', 'url(' + '/static/imagepro/main_bg' + bg_2 + ')');
  bg_3 = getRandomInt(0,max_bg);
  $('.bg-3').css('background-image', 'url(' + '/static/imagepro/main_bg' + bg_3 + ')');
*/
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
                $("#canny_sigma").show();
            }else{
                $("#canny_sigma").hide();		
		}
            if(value == 'contour'){
                image_name = "sample_contour.png";
            }
            if(value == 'skeleton'){
                image_name = "sample_skeleton.png";
            }
	    image_name = "static/imagepro/" + image_name
             $('#transformed').attr('src', image_name);
        });


$("#imgrun-btn").click(function(){  
if($('#img_path').val()!='/' 
	&& $('#img_info').html()!='invalid directory' 
	&& $('#img_path').val()!=''
    ){
//    $('#loadingA').show();
    $('#loading').modal({backdrop: 'static', keyboard: false});
    var radios = document.getElementsByName('filter'), 
        value  = '';

    for (var i = radios.length; i--;) {
        if (radios[i].checked) {
            value = radios[i].value;
            break;
        }
    }
    $.ajax({
	type: "GET",
        url: '/?img_path=' + $('#img_path').val() + "&filter=" + value + "&canny_sigma=" + $('#canny_sigma_input').val() + "&option=" + $("#imgrun-btn").val(),
        dataType: "json",
	async: true,
	success: function(data) {
	     var now = new Date($.now());
	     var log_data = $('#terminal-body').html() + "</br></br>"
	     log_data = log_data + now + " </br> " + data.result
	     $('#terminal-body').html(log_data);
	     $('#imgproResult').html(data.result); 
	     $('#loading').modal('hide');
//	     $('#loadingA').hide();
//	     $('#img-output').modal('show');
 	     var elem = document.getElementById('terminal-body');
	     elem.scrollTop = elem.scrollHeight;
	     $('#terminal').show(500);
        }
    });
}

});


$("#dimrun-btn").click(function(){  
if($('#dim_path').val()!='/' 
	&& $('#dim_info').html()!='invalid data file'
	&& $('#dim_path').val()!=''
    ){
//    $('#loadingB').show();
    $('#loading').modal({backdrop: 'static', keyboard: false});
    var radios = document.getElementsByName('dim_red'), 
        value  = '';

    for (var i = radios.length; i--;) {
        if (radios[i].checked) {
            value = radios[i].value;
            break;
        }
    }
    $.ajax({
	type: "GET",
        url: '/?dim_path=' + $('#dim_path').val() + "&dim_red=" + value + "&dim_k=" + $('#dim_k').val() + "&option=" + $("#dimrun-btn").val(),
        dataType: "json",
	async: true,
	success: function(data) {
	     var now = new Date($.now());
	     var log_data = $('#terminal-body').html() + "</br></br>"
	     log_data = log_data + now + " </br> " + data.result
	     $('#terminal-body').html(log_data);
	     $('#imgproResult').html(data.result); 
	     $('#loading').modal('hide');
	//   $('#loadingB').hide();
//	     $('#img-output').modal('show');
 	     var elem = document.getElementById('terminal-body');
	     elem.scrollTop = elem.scrollHeight;
	     $('#terminal').show(500);
        }
    });
}
});


$("#clurun-btn").click(function(){  
if($('#clu_path').val()!='/' 
	&& $('#clu_info').html()!='invalid data file'
	&& $('#clu_path').val()!=''
    ){
//    $('#loadingC').show();
    $('#loading').modal({backdrop: 'static', keyboard: false});
    var radios = document.getElementsByName('clu_alg'), 
        value  = '';

    for (var i = radios.length; i--;) {
        if (radios[i].checked) {
            value = radios[i].value;
            break;
        }
    }
    $.ajax({
	type: "GET",
        url: '/?clu_path=' + $('#clu_path').val() + "&clu_alg=" + value + "&clu_k=" + $('#clu_k').val() + "&option=" + $("#clurun-btn").val(),
        dataType: "json",
	async: true,
	success: function(data) {
	     var now = new Date($.now());
	     var log_data = $('#terminal-body').html() + "</br></br>"
	     log_data = log_data + now + " </br> " + data.result
	     $('#terminal-body').html(log_data);
	     $('#imgproResult').html(data.result); 
	     $('#loading').modal('hide');
	//   $('#loadingC').hide();
//	     $('#img-output').modal('show');
 	     var elem = document.getElementById('terminal-body');
	     elem.scrollTop = elem.scrollHeight;
	     $('#terminal').show(500);
        }
    });
}
});



});
/*
window.setInterval(function() {
  var elem = document.getElementById('terminal-content');
  elem.scrollTop = elem.scrollHeight;
}, 1000);
*/
/*
var client = new Faye.Client('http://eda3367a.fanoutcdn.com/bayeux');
client.subscribe('/test', function (data) {
    alert('got data: ' + data);
    $('#output').text(data);
});
*/
function refresh() {
    $.ajax({
        url: '/imagepro/terminal/',
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

    $scope.change_img = function($event) {
        var data = {
            img_path: $scope.img_path
        };
        var config = {
                params: data
            };
        $http.get('/?img_path=' + $scope.img_path)
        .success(function(response) {
            $scope.img_info = response.img_info;
        })
        .error(function (data, status, header, config) {
            $scope.img_info = 'Please enter a valid image directory';
        });
    };
    $scope.change_dim = function($event) {
        var data = {
            dim_path: $scope.dim_path
        };
        var config = {
                params: data
            };
        $http.get('/?dim_path=' + $scope.dim_path)
        .success(function(response) {
            $scope.dim_info = response.dim_info;
        })
        .error(function (data, status, header, config) {
            $scope.dim_info = 'Please enter a valid data file';
        });
    };
    $scope.change_clu = function($event) {
        var data = {
            clu_path: $scope.clu_path
        };
        var config = {
                params: data
            };
        $http.get('/?clu_path=' + $scope.clu_path)
        .success(function(response) {
            $scope.clu_info = response.clu_info;
        })
        .error(function (data, status, header, config) {
            $scope.clu_info = 'Please enter a valid data file';
        });
    };
});



function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}
