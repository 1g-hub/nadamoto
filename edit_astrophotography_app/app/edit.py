import cv2
import numpy as np
import math
import itertools
from app.make_coordinate_data import *

# 星のスペクトル型から色の特定
def star_color(color):
    if color[0] == 'O':
        return (255, 176, 155)
    elif color[0] == 'B':
        return (255, 191, 170)
    elif color[0] == 'A':
        return (255, 215, 202)
    elif color[0] == 'F':
        return (255, 247, 248)
    elif color[0] == 'G':
        return (234, 244, 255)
    elif color[0] == 'K':
        return (161, 210, 255)
    elif color[0] == 'M':
        return (111, 204, 255)
    else:
        return (0, 0, 0)

def multMatrix(a, b):
    matrix = [a[0]*b[0]+a[1]*b[3]+a[2]*b[6], a[0]*b[1]+a[1]*b[4]+a[2]*b[7], a[0]*b[2]+a[1]*b[5]+a[2]*b[8]]
    return matrix

def matrix(x, y, ratio, dif_rot, Coordinate):
    a = np.array([x, y, 1])
    b = np.array([ratio*math.cos(dif_rot), ratio*math.sin(dif_rot), 0,
                  -ratio*math.sin(dif_rot), ratio*math.cos(dif_rot), 0,
                  Coordinate[0], Coordinate[1], 1])

    new_coordinate = multMatrix(a, b)

    return new_coordinate[0], new_coordinate[1]

# 点マッチング
def matching(coordinate_r, coordinate_r_com, size_r, height, width, Coordinate, d, rot):
    D = []
    ratio_rot = []
    for i in range(len(coordinate_r_com)):
        a = coordinate_r_com[i][0]
        b = coordinate_r_com[i][1]
        coordinate_r_o = coordinate_r.copy()
        coordinate_r_o.remove(a)
        coordinate_r_o.remove(b)
        
        # 実データにおける距離および角度の算出
        r_d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        r_rot = math.atan2((a[1] - b[1]), (a[0] - b[0]))

        # 実データと写真における距離比と角度差の算出
        ratio = d / r_d
        dif_rot = rot - r_rot

        dis_min1 = 1000
        dis_min2 = 1000
        dis_min3 = 1000

        # 抽出した 5 点のうち残りの 3 点とそれらに最も近い各星との各距離和が最小になる組み合わせを探索
        for j in range(len(coordinate_r_o)):

            x, y = matrix((coordinate_r_o[j][0] - a[0]), (coordinate_r_o[j][1] - a[1]), ratio, dif_rot, Coordinate[0])

            dis1 = math.sqrt((Coordinate[2][0] - x) ** 2 + (Coordinate[2][1] - y) ** 2)
            if dis1 < dis_min1 and size_r[j]:
                dis_min1 = dis1

            dis2 = math.sqrt((Coordinate[3][0] - x) ** 2 + (Coordinate[3][1] - y) ** 2)
            if dis2 < dis_min2:
                dis_min2 = dis2

            dis3 = math.sqrt((Coordinate[4][0] - x) ** 2 + (Coordinate[4][1] - y) ** 2)
            if dis3 < dis_min3:
                dis_min3 = dis3

        D.append(dis_min1 + dis_min2 + dis_min3)
        ratio_rot.append([ratio, dif_rot])

    return D.index(min(D)), ratio_rot[D.index(min(D))][0], ratio_rot[D.index(min(D))][1]

# 編集処理
def edit(img, name, back_check):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    print(type(img))
    height, width = img.shape[:2]

    contours_check, hierarchy_check = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count_check = 0
    for c in range(len(contours_check)):
        M_check = cv2.moments(contours_check[c])
        if M_check['m00'] != 0.0:
            count_check += 1

    kernel = np.ones((7, 7), np.uint8)
    # 星が 5 つ以上抽出されるまで膨張処理を繰り返し適用
    while count_check < 5:
        img = cv2.dilate(img, kernel, iterations=1)
        ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
        contours_check, hierarchy_check = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count_check = 0
        for co in range(len(contours_check)):
            M_check = cv2.moments(contours_check[co])
            if M_check['m00'] != 0.0:
                count_check += 1
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 各星の重心と面積を求める
    cx = []
    cy = []
    area = []
    for j in range(len(contours)):
        M = cv2.moments(contours[j])
        if M['m00'] != 0.0:
            cx.append(int(M['m10'] / M['m00']))
            cy.append(int(M['m01'] / M['m00']))
            area.append(cv2.contourArea(contours[j]))

    # 明るさが上位 5 つの星の情報を獲得
    area = np.array(area)
    area_sort = area.argsort()[::-1]
    Coordinate = []
    for l in range(5):
        coordinate = []
        coordinate.append(cx[area_sort[l]])
        coordinate.append(cy[area_sort[l]])
        Coordinate.append(coordinate)

    # 明るさが上位 2 つの星の距離および角度を計算
    d = math.sqrt((Coordinate[0][0] - Coordinate[1][0]) ** 2 + (Coordinate[0][1] - Coordinate[1][1]) ** 2)
    rot = math.atan2((Coordinate[0][1] - Coordinate[1][1]), (Coordinate[0][0] - Coordinate[1][0]))

    # 実データ内における星の二次元座標情報とサイズおよび色情報を獲得
    coordinate_r, size_r, color_r = make_coordinate(name, height, width, 0, 0)
    coordinate_r_com = list(itertools.permutations(coordinate_r, 2))

    R_D, R_ratio, R_rot = matching(coordinate_r, coordinate_r_com, size_r, height, width, Coordinate, d, rot)

    # 実データにおける 1 番目と 2 番目に明るい星の写真内での座標の計算
    R_ic_x1, R_ic_y1 = matrix((coordinate_r[0][0] - coordinate_r_com[R_D][0][0]), 
                              (coordinate_r[0][1] - coordinate_r_com[R_D][0][1]),
                              R_ratio, R_rot, Coordinate[0])
    R_ic_x2, R_ic_y2 = matrix((coordinate_r[1][0] - coordinate_r_com[R_D][1][0]),
                              (coordinate_r[1][1] - coordinate_r_com[R_D][1][1]),
                              R_ratio, R_rot, Coordinate[0])


    # 画像内における座標に変換
    r_ic_x1 = R_ic_x1 - width / 2
    r_ic_y1 = height / 2 - R_ic_y1
    r_ic_x1 /= R_ratio
    r_ic_y1 /= R_ratio

    r_ic_x2 = R_ic_x2 - width / 2
    r_ic_y2 = height / 2 - R_ic_y2
    r_ic_x2 /= R_ratio
    r_ic_y2 /= R_ratio


    # 2 点間距離を基に切り出す範囲を決定し, 原点を中心にして座標を設定する
    point_1, point_2, point_3 = [width / (R_ratio * 2), 1000,
                                 -(height / (R_ratio * 2))], \
                                [-(width / (R_ratio * 2)), 1000,
                                 -(height / (R_ratio * 2))], \
                                [-(width / (R_ratio * 2)), 1000,
                                 height / (R_ratio * 2)]

    # 実データ内における星の三次元座標情報とサイズおよおび色情報を獲得
    Coordinate_r_3, Size_r_3, color_r_3 = make_coordinate(name, height, width, -np.rad2deg(R_rot), 1)

    # 明るさが上位 2 つの星のサイズが全く同じであった場合両者の違いを色で判断
    if size_r[0] == Size_r_3[0] and color_r[0] == color_r_3[0]:
        Coordinate_r_3[0], Coordinate_r_3[1] = Coordinate_r_3[0], Coordinate_r_3[1]
    else:
        Coordinate_r_3[0], Coordinate_r_3[1] = Coordinate_r_3[1], Coordinate_r_3[0]

    # 仰俯角 (横方向) を計算し, その角度分回した後の座標を計算
    a_x = math.asin(r_ic_y1 / math.sqrt(Coordinate_r_3[0][1] ** 2 + Coordinate_r_3[0][2] ** 2)) - math.asin(Coordinate_r_3[0][2] / math.sqrt(Coordinate_r_3[0][1] ** 2 + Coordinate_r_3[0][2] ** 2))
    Coordinate_r_3_x = Coordinate_r_3[0][0]
    Coordinate_r_3_y = Coordinate_r_3[0][1] * math.cos(a_x) - Coordinate_r_3[0][2] * math.sin(a_x)

    # 方位角 (縦方向) を計算
    a_z = math.acos(r_ic_x1 / math.sqrt(Coordinate_r_3_x ** 2 + Coordinate_r_3_y ** 2)) - math.acos(Coordinate_r_3_x / math.sqrt(Coordinate_r_3_x ** 2 + Coordinate_r_3_y ** 2))

    # 検出星座が原点にくる際の x 軸, z 軸における移動距離を取得
    arg_X, arg_Z = take_rota(name)

    # 特定した画角部分が原点を中心とした部分にくるまで回転し, 回転後の各星の情報を獲得する
    coordinate_p, size_p, color_p = make_starfigure_main(point_1, point_2, point_3, arg_X, arg_Z, np.rad2deg(a_x), np.rad2deg(a_z), 2, -np.rad2deg(R_rot))

    """
    ここからは背景画像の処理. 木や影などがある場合には sky が, ない場合には back が使われる. 
    """

    back = cv2.imread("app/static/input/input.png")
    
    sky = cv2.imread("app/static/input/input.png")
    # 画像内に写っている星をオープニング処理で削除
    kernel2 = np.ones((7, 7), np.uint8)
    sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, kernel2)

    # 画像のグレースケール化
    gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

    # 画像の白黒2値化
    ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 輪郭を抽出する
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black = np.zeros((height, width, 3), np.uint8)
    # 輪郭を画像に書き込む
    output = cv2.drawContours(black, contours, -1, (255, 255, 255), 20)
    output2 = cv2.drawContours(back, contours, -1, (255, 255, 255), 20)
    mask = cv2.bitwise_not(output)

    output3 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(output3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cnt[0][0][1] < (height) / 2:
            output = cv2.drawContours(output, [cnt], 0, (255, 255, 255), -1)
            output2 = cv2.drawContours(output2, [cnt], 0, (255, 255, 255), -1)

    mask = cv2.bitwise_and(output, sky)
    target_color = (0, 0, 0)
    change_color = (255, 255, 255)
    mask[np.where((mask == target_color).all(axis=2))] = change_color
    mask = np.asarray(mask, np.uint8)
    if back_check == []:
        mask = sky

    # 星のぼやけた部分を描画
    for i in range(len(coordinate_p)):
        x = coordinate_p[i][0] * R_ratio
        y = coordinate_p[i][1] * R_ratio
        if ((max(height, width)/2) // size_p[i]) / 2 > 150:
            mask2 = np.zeros((int(height), int(width), 3), np.uint8)
            mask2 = cv2.circle(mask2, (int(x), int(y)), int(((max(height, width)/80) // size_p[i]) / 2), star_color(color_p[i]), -1)
            mask2 = cv2.blur(mask2, (20, 20))
            mask = cv2.addWeighted(src1=mask, alpha=1.0, src2=mask2, beta=1.0, gamma=0)

    # 星の中心部を描画
    for i in range(len(coordinate_p)):
        x = coordinate_p[i][0] * R_ratio
        y = coordinate_p[i][1] * R_ratio
        coordinate_p[i][0] = coordinate_p[i][0] * R_ratio
        coordinate_p[i][1] = coordinate_p[i][1] * R_ratio
        mask = cv2.circle(mask, (int(x), int(y)), int(((max(height, width)/160) // size_p[i]) / 2), (255, 255, 255), -1)
    mask = mask.clip(0, 255)

    img_AND2 = cv2.bitwise_and(mask, output2)
    if back_check == ['1']:
        return img_AND2, coordinate_p, size_p, color_p
    else:
        return mask, coordinate_p, size_p, color_p
