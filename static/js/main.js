let canvas = document.getElementById("drawing-board");
let ctx = canvas.getContext("2d");
let eraser = document.getElementById("eraser");
let brush = document.getElementById("brush");
let reSetCanvas = document.getElementById("clear");
let aColorBtn = document.getElementsByClassName("color-item");
let save = document.getElementById("save");
let undo = document.getElementById("undo");
let range = document.getElementById("range");
let scan = document.getElementById("scan");
let isRequesting = false

let clear = false;
let activeColor = 'black';
let lWidth = 4;

let historyDeta = [];


autoSetSize(canvas);

setCanvasBg('white');

listenToUser(canvas);

getColor();

//window.onbeforeunload = function(){
//    return "Reload site?";
//};

function autoSetSize(canvas) {
    canvasSetSize();

    function canvasSetSize() {
        let pageWidth = document.documentElement.clientWidth;
        let pageHeight = document.documentElement.clientHeight;

        canvas.width = pageWidth;
        canvas.height = pageHeight;
    }

    window.onresize = function () {
        canvasSetSize();
    }
}

function setCanvasBg(color) {
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black";
}

function listenToUser(canvas) {
    console.log("listenToUser")
    let painting = false;
    let lastPoint = {x: undefined, y: undefined};

    if (document.body.ontouchstart !== undefined) {
        canvas.ontouchstart = function (e) {
            this.firstDot = ctx.getImageData(0, 0, canvas.width, canvas.height);//在这里储存绘图表面
            saveData(this.firstDot);
            painting = true;
            let x = e.touches[0].clientX;
            let y = e.touches[0].clientY;
            lastPoint = {"x": x, "y": y};
            ctx.save();
            drawCircle(x, y, 0);
        };
        canvas.ontouchmove = function (e) {
            if (painting) {
                let x = e.touches[0].clientX;
                let y = e.touches[0].clientY;
                let newPoint = {"x": x, "y": y};
                drawLine(lastPoint.x, lastPoint.y, newPoint.x, newPoint.y);
                lastPoint = newPoint;
            }
        };

        canvas.ontouchend = function () {
            painting = false;
        }
    } else {
        canvas.onmousedown = function (e) {
            this.firstDot = ctx.getImageData(0, 0, canvas.width, canvas.height);//在这里储存绘图表面
            saveData(this.firstDot);
            painting = true;
            let x = e.clientX;
            let y = e.clientY;
            lastPoint = {"x": x, "y": y};
            ctx.save();
            drawCircle(x, y, 0);
        };
        canvas.onmousemove = function (e) {
            if (painting) {
                let x = e.clientX;
                let y = e.clientY;
                let newPoint = {"x": x, "y": y};
                drawLine(lastPoint.x, lastPoint.y, newPoint.x, newPoint.y, clear);
                lastPoint = newPoint;
            }
        };

        canvas.onmouseup = function () {
            painting = false;
        };

        canvas.mouseleave = function () {
            painting = false;
        }
    }
}

function drawCircle(x, y, radius) {
    console.log("drawCircle")
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    if (clear) {
        ctx.clip();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    }
}

function drawLine(x1, y1, x2, y2) {
    console.log("drawLine")
    ctx.lineWidth = lWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (clear) {
        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.closePath();
        ctx.clip();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    } else {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.closePath();
    }
}

range.onchange = function () {
    lWidth = this.value;
};

eraser.onclick = function () {
    clear = true;
    this.classList.add("active");
    brush.classList.remove("active");
};

brush.onclick = function () {
    clear = false;
    this.classList.add("active");
    eraser.classList.remove("active");
};

reSetCanvas.onclick = function () {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setCanvasBg('white');
    var result_text = document.getElementById("info");
    result_text.innerHTML = "识别结果"
};
/*
save.onclick = function () {
    let imgUrl = canvas.toDataURL("image/png");
//    let saveA = document.createElement("a");
//    document.body.appendChild(saveA);
//    saveA.href = imgUrl;
//    saveA.download = "sketch" + (new Date).getTime();
//    saveA.target = "_blank";
//    saveA.click();
    console.log(imgUrl)
    document.getElementById("sketchUpload").value=imgUrl;

};

 */
/*
upload.onclick = function () {
    uploadFlag.val = "1";
    uploadSketch.click();
};

 */

scan.onclick = function () {
    //无正在进行的请求
    if (!isRequesting) {
        isRequesting = true
        var loadingImgObj = new loadingImg()
        loadingImgObj.show()

        let imgUrl = canvas.toDataURL("image/png");
        console.log(imgUrl)
        var data = {
            "img": imgUrl,
            "token": "THISISAFUCKINGTOKEN"
        }
        /*{#        $.getJSON($SCRIPT_ROOT + '/_add_numbers',data, function(data) {#}
        {#          $('#result').text(data.result);#}
        {#          $('input[name=a]').focus().select();#}
        {#        });#}
         */

        $.ajax({
            type: 'post',
            url: '/digit_rec',
            data: JSON.stringify(data),
            contentType: 'application/json; charset=UTF-8',
            dataType: 'json', success: function (data) {
                if (data.code == "200") {
                    var rootObj = JSON.parse(data.result)
                    showResult(rootObj.numbers, rootObj.marked_img)
                    //popPicWindow(rootObj.numbers,rootObj.marked_img)
                    //console.log(rootObj.marked_img)
                } else {
                    alert('请求错误:'+data.code+" "+data.message)
                }
                loadingImgObj.hide()
                scan.innerHTML = "<i class=\"iconfont icon-canvas-ranlysaoma\"></i>"
                isRequesting = false
            }, error: function (xhr, type, xxx) {
                alert('请求错误：未知错误')
                loadingImgObj.hide()
                scan.innerHTML = "<i class=\"iconfont icon-canvas-ranlysaoma\"></i>"
                isRequesting = false
            }
        });
    } else {
        alert("正在识别中，请等待");
    }
}

/*
uploadSketch.onchange = function () {
    upload_form.submit();
};

 */

function getColor() {
    for (let i = 0; i < aColorBtn.length; i++) {
        aColorBtn[i].onclick = function () {
            for (let i = 0; i < aColorBtn.length; i++) {
                aColorBtn[i].classList.remove("active");
                this.classList.add("active");
                activeColor = this.style.backgroundColor;
                ctx.fillStyle = activeColor;
                ctx.strokeStyle = activeColor;
            }
        }
    }
}

// let historyDeta = [];

function saveData(data) {
    (historyDeta.length === 10) && (historyDeta.shift());// 上限为储存10步，太多了怕挂掉
    historyDeta.push(data);
}

undo.onclick = function () {
    if (historyDeta.length < 1) return false;
    ctx.putImageData(historyDeta[historyDeta.length - 1], 0, 0);
    historyDeta.pop()
    var result_text = document.getElementById("info");
    result_text.innerHTML = "识别结果"
};

//展示结果
function showResult(text, imgSrc) {
    popPicWindow(text, imgSrc)
    //result_text.innerHTML="识别结果："+text
    //result_img.src=imgSrc
}

//加载图片方法（对象）
function loadingImg(mySetting) {
    var that = this;
    if (mySetting == "" || mySetting == undefined || typeof mySetting != "object") {
        mySetting = {};
    }
    //使用时间戳作为空间的ID
    var targetID = new Date().getTime();
    this.setting = {
        //插入图片的容器,使用jquery的查询方式传入参数
        targetConater: scan,
        //使用图片的地址
        imgUrl: "/static/img/loading1.gif",
        //图片显示的 宽度
        imgWidth: "",
        //图片的默认样式
        imgClass: "",
        //生成控件的ID
        "targetID": targetID,
        //显示之前的回调函数
        beforeShow: function (plugin) {
        },
        //显示之后的回调函数
        afterShow: function (plugin, targetID) {
        }
    }
    this.setting = $.extend(this.setting, mySetting);
    //获取屏幕的宽度
    this.getScreenWidth = function () {
        return document.documentElement.clientWidth;
    }
    //获取屏幕的高度
    this.getScreenHeight = function () {
        return document.documentElement.clientHeight;
    }
    //显示控件
    this.show = function () {
        $("#" + that.setting.targetID).show();
    }
    //隐藏控件
    this.hide = function () {
        $("#" + that.setting.targetID).hide();
    }
    this.init = function () {
        //显示之前执行回调函数
        if (typeof that.setting.beforeShow == "function") {
            that.setting.beforeShow(that);
        }
        //存放字符串的变量
        var targetHTML = '';
        //将内容存放到指定的容器中，默认存放到body最底部
        if (that.setting.targetConater != "" && this.setting.targetConater != undefined) {
            targetHTML = '<img src="' + that.setting.imgUrl + '" class="' + that.setting.imgClass + '" id="' + that.setting.targetID + '" style="display:none;vertical-align: middle">';
            $(that.setting.targetConater).html(targetHTML);
        } else {
            targetHTML = '<img src="' + that.setting.imgUrl + '" class="' + that.setting.imgClass + '" style="margin: 0 auto;">';

            targetHTML = '<div id="' + that.setting.targetID + '" style="display:none;position: absolute;top:50%;left: 50%;height: ' + that.getScreenHeight() + ';width:' + that.getScreenWidth() + '">' + targetHTML + '</div>';
            $("body").append(targetHTML);
        }
        //判断用户是否自定义了图片的宽度
        if (that.setting.imgWidth != "" && that.setting.imgWidth.indexOf("px") > 0) {
            $("#" + targetID).css("width", that.setting.imgWidth);
        }
        //显示之后执行回调函数
        if (typeof that.setting.afterShow == "function") {
            that.setting.afterShow(that, targetID);
        }
    }
    this.init();
}

function popPicWindow(text, imgSrc) {
    // 获取弹窗
    var modal = document.getElementById('myModal');

    // 获取图片插入到弹窗 - 使用 "alt" 属性作为文本部分的内容
    //var img = document.getElementById('myImg');
    var modalImg = document.getElementById("img01");
    var captionText = document.getElementById("caption");
    modal.style.display = "block";
    modalImg.src = imgSrc;
    captionText.innerHTML = "识别结果：" + text;

    // 获取 <span> 元素，设置关闭按钮
    var span = document.getElementsByClassName("close")[0];

    // 当点击 (x), 关闭弹窗
    span.onclick = function () {
        modal.style.display = "none";
    }
    /*
    $('#item').popup({
        time: 500,
        classAnimateShow: 'flipInX',
        classAnimateHide: 'hinge',
        onPopupClose: function e() {
            // on window close
        },
        onPopupInit: function e() {
            // on window init

            var result_text = document.getElementById("info")
            var result_img = document.getElementById("result_img")
            result_text.innerHTML = "识别结果：" + text
            result_img.src = imgSrc
        }
    });

     */
}

