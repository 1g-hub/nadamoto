import cv2
import numpy as np
import math
import csv

# x 軸周りに回転
def all_rota_coordinate_x(x, y, z, arg):
    Y_NEW = []
    Z_NEW = []
    for i in range(len(y)):
        Y_NEW.append(y[i]*math.cos(arg) - z[i]*math.sin(arg))
        Z_NEW.append(y[i]*math.sin(arg) + z[i]*math.cos(arg))

    return x, Y_NEW, Z_NEW

# z 軸周りに回転
def all_rota_coordinate_z(x, y, z, arg):
    X_NEW = []
    Y_NEW = []
    for i in range(len(x)):
        X_NEW.append(x[i]*math.cos(arg) - y[i]*math.sin(arg))
        Y_NEW.append(x[i]*math.sin(arg) + y[i]*math.cos(arg))

    return X_NEW, Y_NEW, z

# 各星座が原点付近にくるように回転させる角度 (自分で設定)
def take_rota(name):
    if name == "Orion":
        arg_x = 1
        arg_z = 6
    elif name == "Canis Major":
        arg_x = 23
        arg_z = -12
    elif name == "Taurus":
        arg_x = -20
        arg_z = 22
    else:
        arg_x = -24
        arg_z = -13

    return arg_x, arg_z

def make_starfigure_main(point_1_org, point_2_org, point_3_org, arg_X, arg_Z, arg_x, arg_z, version, arg):
    num = []
    ra = []
    dec = []
    color = []
    size = []
    # 516個
    # マッチングには 5.0 等星までの星を, 星の補完には 7.0 等星までの星を使用
    if version == 0 or version == 1:
        file_name = 'app/static/starcatalog/browse_results_5.0.csv'
    else:
        file_name = 'app/static/starcatalog/starcatalog_hpc.csv'
    with open(file_name, "r") as f:
        for rows in csv.reader(f):
            num.append(rows[0])
            ra.append(rows[1])
            dec.append(rows[2])
            color.append(rows[3])
            size.append(rows[4])

    # ヒッパルコス番号
    NUM = []
    for i in range(len(num)-1):
        NUM.append(num[i+1])
    # 赤経
    RA = []
    for i in range(len(ra)-1):
        r = ra[i+1]
        r = float(r)
        r = (r * math.pi) / 180
        RA.append(r)
    # 赤緯
    DEC = []
    for i in range(len(dec)-1):
        d = dec[i+1]
        d = float(d)
        d = (d * math.pi) / 180
        DEC.append(d)

    # 地平座標に変換
    X = []
    Y = []
    Z = []
    # 投影天球の半径
    R = 1000
    for i in range(len(RA)):
        X.append(R * math.cos(RA[i]) * math.cos(DEC[i]))
        Y.append(R * math.sin(RA[i]) * math.cos(DEC[i]))
        Z.append(R * math.sin(DEC[i]))

    COLOR = []
    for i in range(len(color)-1):
        if color[i+1] == '':
            COLOR.append('0')
        else:
            COLOR.append(color[i+1])

    # 等級
    SIZE = []
    for i in range(len(size)-1):
        SIZE.append(size[i+1])
    SIZE = [float(s) for s in SIZE]
    # 等級が 1 未満のときは 1 として扱う
    SIZE = [float(s) for s in SIZE]
    for i in range(len(SIZE)):
        if SIZE[i] < 1:
            SIZE[i] = 1

    # 切り取る平面の指定(point_1 は右下の点, point_2 は左下の点, point_3 は左上の点に相当)
    point_1 = point_1_org
    point_2 = point_2_org
    point_3 = point_3_org

    width = math.sqrt((point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) +
                      (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]) +
                      (point_1[2] - point_2[2]) * (point_1[2] - point_2[2]))
    height = math.sqrt((point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) +
                       (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]) +
                       (point_2[2] - point_3[2]) * (point_2[2] - point_3[2]))
    black = np.zeros((int(height), int(width), 3))

    Coordinate, Size, Color = rotate_image(point_1, point_2, point_3, width, height, X, Y, Z, arg_X, arg_Z, arg_x, arg_z, SIZE, COLOR, black, version, arg)
    return Coordinate, Size, Color

# 各星を平面上に投影
def projection(point_1, point_2, point_3, X, Y, Z, SIZE, COLOR):
    X_new = []
    Y_new = []
    Z_new = []
    SIZE_new = []
    COLOR_new = []
    for i in range(len(X)):
        # 平面の y 座標が正であるならば y 座標が正の部分のみ投影する
        if Y[i] > 0:
            t = point_1[1] / 500
            x_new = X[i]
            y_new = point_1[1]
            z_new = Z[i]
        else:
            continue

        X_new.append(x_new)
        Y_new.append(y_new)
        Z_new.append(z_new)
        SIZE_new.append(SIZE[i])
        COLOR_new.append(COLOR[i])

    return X_new, Y_new, Z_new, SIZE_new, COLOR_new

# 新しい軸に再び座標変換して描画
def make_starfigure(point_1, point_2, point_3, X_new, Y_new, Z_new, Size, Color, black, version):
    Coordinate = []
    Size_new = []
    Color_new = []
    for i in range(len(X_new)):
        x_axes = X_new[i] - point_2[0]
        y_axes = point_3[2] - Z_new[i]

        if version == 1:
            Coordinate.append([X_new[i], Y_new[i], Z_new[i]])
        else:
            Coordinate.append([x_axes, y_axes])
        Size_new.append(Size[i])
        Color_new.append(Color[i])

    return Coordinate, Size_new, Color_new

# 各星の回転座標を生成
def rotate_coordinate(cx, cy, Arg, X, Y, Z):
    X_new = []
    Y_new = []
    Z_new = []
    for i in range(len(X)):
        point_rotate = [(X[i]-cx)*math.cos(math.radians(Arg))-(Z[i]-cy)*math.sin(math.radians(Arg))+cx,
                        Y[i],
                        (X[i]-cx)*math.sin(math.radians(Arg))+(Z[i]-cy)*math.cos(math.radians(Arg))+cy]
        X_new.append(point_rotate[0])
        Y_new.append(point_rotate[1])
        Z_new.append(point_rotate[2])
    return X_new, Y_new, Z_new

# 切り出す範囲の座標, 回転角などの情報を用いて各星の座標を計算
def rotate_image(point_1, point_2, point_3, width, height, X, Y, Z, arg_X, arg_Z, arg_x, arg_z, SIZE, COLOR, black, version, arg):
    Arg = arg

    cx = point_2[0] + (width/2)
    cy = point_1[2] + (height/2)
    if version == 0 or version == 1:
        X_new = []
        Y_new = []
        Z_new = []
        SIZE_new = []
        COLOR_new = []
        if version == 0:
            X, Y, Z = all_rota_coordinate_x(X, Y, Z, math.radians(arg_X))
            X, Y, Z = all_rota_coordinate_z(X, Y, Z, math.radians(arg_Z))
            X, Y, Z = rotate_coordinate(cx, cy, Arg, X, Y, Z)
            for i in range(len(X)):
                if (X[i] >= point_2[0]) and (X[i] <= point_1[0]) and (Z[i] >= point_1[2]) and (Z[i] <= point_3[2]) and \
                        Y[i] > 0:
                    X_new.append(X[i])
                    Y_new.append(Y[i])
                    Z_new.append(Z[i])
                    SIZE_new.append(SIZE[i])
                    COLOR_new.append(COLOR[i])
        else:
            X_check, Y_check, Z_check = all_rota_coordinate_x(X, Y, Z, math.radians(arg_X))
            X_check, Y_check, Z_check = all_rota_coordinate_z(X_check, Y_check, Z_check, math.radians(arg_Z))
            for i in range(len(X)):
                if (X_check[i] >= point_2[0]) and (X_check[i] <= point_1[0]) and (Z_check[i] >= point_1[2]) and (Z_check[i] <= point_3[2]) and \
                        Y[i] > 0:
                    X_new.append(X[i])
                    Y_new.append(Y[i])
                    Z_new.append(Z[i])
                    SIZE_new.append(SIZE[i])
                    COLOR_new.append(COLOR[i])
            X_new, Y_new, Z_new = rotate_coordinate(cx, cy, Arg, X_new, Y_new, Z_new)
            X_new, Y_new, Z_new = all_rota_coordinate_x(X_new, Y_new, Z_new, math.radians(arg_X))
            X_new, Y_new, Z_new = all_rota_coordinate_z(X_new, Y_new, Z_new, math.radians(arg_Z))

    else:
        X_new, Y_new, Z_new = rotate_coordinate(cx, cy, Arg, X, Y, Z)
        X_new, Y_new, Z_new = all_rota_coordinate_x(X_new, Y_new, Z_new, math.radians(arg_X))
        X_new, Y_new, Z_new = all_rota_coordinate_z(X_new, Y_new, Z_new, math.radians(arg_Z))
        X_new, Y_new, Z_new = all_rota_coordinate_x(X_new, Y_new, Z_new, math.radians(arg_x))
        X_new, Y_new, Z_new = all_rota_coordinate_z(X_new, Y_new, Z_new, math.radians(arg_z))
        SIZE_new = SIZE
        COLOR_new = COLOR

    if version == 0 or version == 2:
        X_new, Y_new, Z_new, SIZE_new, COLOR_new = projection(point_1, point_2, point_3, X_new, Y_new, Z_new, SIZE_new, COLOR_new)
    Coordinate, SIZE_new, COLOR_new = make_starfigure(point_1, point_2, point_3, X_new, Y_new, Z_new, SIZE_new, COLOR_new, black, version)

    return Coordinate, SIZE_new, COLOR_new

# 切り出す範囲および星の座標を計算
def make_coordinate(name, height, width, arg, version):

    rate = height/width
    if name == "Orion":
        if height < width:
            Z = 180
            X = Z / rate
        else:
            X = 180
            Z = X * rate
        point1_org, point2_org, point3_org = [X, 1000, -Z], [-X, 1000, -Z], [-X, 1000, Z]
    elif name == "Canis Major":
        if height < width:
            Z = 140
            X = Z / rate
        else:
            X = 140
            Z = X * rate
        point1_org, point2_org, point3_org = [X, 1000, -Z], [-X, 1000, -Z], [-X, 1000, Z]
    elif name == "Taurus":
        if height < width:
            Z = 200
            X = Z / rate
        else:
            X = 200
            Z = X * rate
        point1_org, point2_org, point3_org = [X, 1000, -Z], [-X, 1000, -Z], [-X, 1000, Z]
    else:
        point1_org, point2_org, point3_org = [10, 1000, -110], [-110, 1000, -110], [-110, 1000, 110]

    arg_X, arg_Z = take_rota(name)

    Coordinate, Size, Color = make_starfigure_main(point1_org, point2_org, point3_org, arg_X, arg_Z, 0, 0, version, arg)

    Coordinate_new = []
    Color_new = []
    Size_new = sorted(Size)
    Size = np.array(Size)
    size_sort = Size.argsort()

    for i in range(len(Coordinate)):
        Coordinate_new.append(Coordinate[size_sort[i]])
        Color_new.append(Color[size_sort[i]])
    if version == 0:
        black = np.zeros((int(point3_org[2]-point1_org[2]), int(point1_org[0]-point2_org[0]), 3))
        for j in range(len(Coordinate_new)):
            black = cv2.circle(black, (int(Coordinate_new[j][0]), int(Coordinate_new[j][1])), int(5/Size_new[j]), (255, 255, 255), thickness=-1)


    return Coordinate_new, Size_new, Color_new

