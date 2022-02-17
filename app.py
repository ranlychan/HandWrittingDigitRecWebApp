# coding:utf-8
import numpy as np
from flask import Flask, render_template, request
import json

from datetime import timedelta

# sketchseg network------------------------
# from SketchSeg.SketchSegDataset import *
# from SketchSeg.fcn_stn_s800_v2 import *
# from SketchSeg.seg import seg_sketch
# # sketchtriplet network--------------------
# from SketchTriplet.SketchTriplet_half_sharing import BranchNet
# from SketchTriplet.SketchTriplet_half_sharing import SketchTriplet as SketchTriplet_hs
# from SketchTriplet.flickr15k_dataset import flickr15k_dataset_lite
# from SketchTriplet.retrieval import retrieval

# def load_model_seg():
#     net_dict_path = '../fcn_stn/output/model_s800_v2/fcn_stnv2_b5_w_f_1811151659/seg/seg_model_49.pth'
#     net = fcn(len(myPara.classes))
#     net.load_state_dict(torch.load(net_dict_path))
#     net = net.eval()
#     net = net.cuda()
#     return net

# def load_model_retrieval():
#     net_dict_path = '../SketchTriplet/out/flickr15k_1904041458/500.pth'
#     branch_net = BranchNet()  # for photography edge
#     net = SketchTriplet_hs(branch_net)
#     net.load_state_dict(torch.load(net_dict_path))
#     net = net.cuda()
#     net.eval()
#     return net

#-----------------------------------------
from mnist_predict import mnist_digit_rec

DEFAULT_TOKEN = "THISISAFUCKINGTOKEN"

app = Flask(__name__,
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/')
def hello():
    return render_template('canva.html')

# post方式http请求与js交互
@app.route('/digit_rec',methods=["POST"])
def digit_rec():
    # 默认返回内容
    return_dict = {'code': '200', 'message': '处理成功', 'result': False}

    print(request.get_json())

    # 错误处理
    if request.get_data() is None:
        return_dict['code'] = '5002'
        return_dict['message'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)

    # 获取传入的参数
    get_data = request.get_json()

    # 使用简单的token校验
    token = get_data.get("token")
    if token != DEFAULT_TOKEN:
        return_dict['code'] = '5001'
        return_dict['message'] = 'TOKEN错误'
        return json.dumps(return_dict, ensure_ascii=False)

    img_base64 = get_data.get("img")

    # 调用模型识别
    result_dict = mnist_digit_rec(img_base64)
    print(result_dict)

    # 结果打包为JSON格式
    json_encode_result = json.dumps(result_dict, cls=NpEncoder)
    print(json_encode_result)

    return_dict['result'] = json_encode_result

    return_str = json.dumps(return_dict, ensure_ascii=False)
    print(return_str)

    return return_str

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            print(obj.tolist())
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True,host='0.0.0.0')