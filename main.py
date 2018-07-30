#coding=utf-8
import os

from flask import Flask, flash, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return '''
        <!doctype html>
        <title>Image Detect</title>
        <h2>API:</h2>
        
        <h1>1.GET请求</h1>
        URL:<a href="/detect">http://ip/detect</a>
        <br>说明：<br>
        打开上传页面
        
        <h1>2.POST请求</h1>
        URL:<a href="/detect?debugger=true">http://ip/detect?debugger=false</a>
        <br>说明：<br>
        比如： curl -F "file=@/home/cc/ai/tmp/text-detection-ctpn/data/demo/3.png" http://localhost:8080/detect\?debugger\=true > 3.marked.png
        <br>debugger取值：<br>
        false：则返回识别文字后的坐标数据（默认）.比如 [276,406,698,448,92,55,329,109,144,175,487,257] 其中每4个值为一个标记矩形在图片上的像素坐标<br>
        true：则下载标记后的图片<br>
        </html>
        '''


@app.route('/detect', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        upload_f = request.files['file']  # type: object
        if upload_f.filename == '':
            return 'No selected file'
        if upload_f and allowed_file(upload_f.filename):
            filename = secure_filename(upload_f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, "result."+filename)
            upload_f.save(save_path)
            boxes, scale, img = detect(save_path)
            if 'true' == request.args.get('debugger'):
                draw_boxes(img, save_path, boxes, scale)
                return send_from_directory(UPLOAD_FOLDER, "result."+filename, as_attachment=True)
            os.remove(save_path)
            return get_boxes_info(boxes, scale)
        else:
            return 'No allowed file'
    return '''
        <!doctype html>
        <title>Upload Image File ['png', 'jpg', 'jpeg']</title>
        <h1>Upload Image File ['png', 'jpg', 'jpeg']</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        </html>
        '''



def draw_boxes(img, image_name, boxes, scale):
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(image_name, img)


def get_boxes_info(boxes, scale):
    line = ""
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        line = line + ','.join([str(min_x), str(min_y), str(max_x), str(max_y)])
    return "[" + line.lstrip(",") + "]"


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


sess = None
input_img = None
output_cls_prob = None
output_box_pred = None
text_detector = None


def init_session():
    global sess, input_img, output_cls_prob, output_box_pred, text_detector

    cfg_from_file('ctpn/text.yml')
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    text_detector = TextDetector()


def detect(im_name):
    print('detect for {:s}'.format(im_name))
    img = cv2.imread(im_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)
    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    return boxes, scale, img


if "__main__" == __name__:
    init_session()
    app.run(host='0.0.0.0', port=8080)
