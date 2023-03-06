#Flaskとrender_template（HTMLを表示させるための関数）をインポート
from flask import Flask,render_template,request,url_for,redirect,send_from_directory
from datetime import datetime
import os
import numpy as np
import cv2
from app.ssd_test import *
from app.make_background import *

current_generation = []
coordinate = []
size = []
color = []

# 入力画像, 出力画像および背景画像の候補を保存するフォルダ
SAVE_DIR = "./app/static/input"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

app = Flask(__name__)

# ファイルを変更した時に毎回更新処理をする(この処理がないと同じ名前の別ファイルに変更した時に変更が適用されない)
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/')
@app.route('/index')
def index():
    # index.html をレンダリングする
    return render_template('./index.html')

#「/input」へアクセスがあった場合に, next.html と画像を返す
@app.route('/input')
def input():
    return render_template('./next.html', images=os.listdir(SAVE_DIR)[::-1])

#「/nextpage」へアクセスがあった場合に, next.htmlを返す
@app.route("/nextpage", methods=["GET"])
def nextpage():
    return render_template("./next.html")

@app.route('/images/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

#「/upload」へアクセスがあった場合の処理
@app.route('/upload', methods=["POST"])
def upload():
    global current_generation
    global coordinate
    global size
    global color
    if request.files['image']:
        # 画像として読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img1 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        # チェックボックスにチェックが入っているかどうか
        check = request.form.getlist('checkbox')

        # 保存
        save_path = os.path.join(SAVE_DIR, "input.png")
        cv2.imwrite(save_path, img1)

        # 編集 & 保存
        coordinate, size, color = test(img2, check)
        # 背景画像の初期個体群の生成 & 保存
        current_generation = make_first_background()

        return redirect('/input')

@app.route("/select_image1", methods=["GET"])
def select_img1():
    global current_generation
    global coordinate
    global size
    global color
    new_img = new_image(coordinate, size, color, 1)
    current_generation = iGA(current_generation, 1)

    return redirect('/input')

@app.route("/select_image2", methods=["GET"])
def select_img2():
    global current_generation
    global coordinate
    global size
    global color
    new_img = new_image(coordinate, size, color, 2)
    current_generation = iGA(current_generation, 2)

    return redirect('/input')

@app.route("/select_image3", methods=["GET"])
def select_img3():
    global current_generation
    global coordinate
    global size
    global color
    new_img = new_image(coordinate, size, color, 3)
    current_generation = iGA(current_generation, 3)

    return redirect('/input')

@app.route("/select_image4", methods=["GET"])
def select_img4():
    global current_generation
    global coordinate
    global size
    global color
    new_img = new_image(coordinate, size, color, 4)
    current_generation = iGA(current_generation, 4)

    return redirect('/input')

@app.route("/reset", methods=["GET"])
def reset():
    global current_generation
    current_generation = make_first_background()

    return redirect('/input')

if __name__ == "__main__":
    app.run(debug=True)