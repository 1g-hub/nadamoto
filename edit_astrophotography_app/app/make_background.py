import app.GeneticAlgorithm as ga
import random
from decimal import Decimal
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
import cv2
import imgsim
import numpy as np
from PIL import Image
from app.edit import *

latent_dim = 100
# 遺伝子情報の長さ
GENOM_LENGTH = 100
# 遺伝子集団の大きさ
MAX_GENOM_LIST = 4
# 遺伝子選択数
SELECT_GENOM = 4
# 個体突然変異確率
INDIVIDUAL_MUTATION = 0.1
# 遺伝子突然変異確率
GENOM_MUTATION = 0.1
# 繰り返す世代数
MAX_GENERATION = 40

# GAN の Generator
class Generator(nn.Module):
    def __init__(self, input_dim, n_features=128):
        super().__init__()
        self.main = nn.Sequential(
            # fmt: off
            # conv1
            nn.ConvTranspose2d(input_dim, n_features * 16,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_features * 16),
            nn.ReLU(inplace=True),
            # conv2
            nn.ConvTranspose2d(n_features * 16, n_features * 8,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 8),
            nn.ReLU(inplace=True),
            # conv3
            nn.ConvTranspose2d(n_features * 8, n_features * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU(inplace=True),
            # conv4
            nn.ConvTranspose2d(n_features * 4, n_features * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(inplace=True),
            # conv5
            nn.ConvTranspose2d(n_features * 2, n_features,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            # conv6
            nn.ConvTranspose2d(n_features, 3,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # fmt: on
        )

    def forward(self, x):
        return self.main(x)

def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")

device = get_device(gpu_id=0)

def generate_img(G, fixed_z):
    with torch.no_grad():
        # 画像を生成する。
        x = G(fixed_z)

    # 画像を格子状に並べる。
    img = torchvision.utils.make_grid(x.cpu(), nrow=1, normalize=True, pad_value=1)
    # テンソルを PIL Image に変換する。
    img = transforms.functional.to_pil_image(img)
    # img = torchvision.transforms.functional.to_pil_image(np.squeeze(x))

    return img

def evaluation(select_num, i):
    """評価関数です。今回は全ての遺伝子が1となれば最適解となるので、
    合計して遺伝子と同じ長さの数となった場合を1として0.00~1.00で評価
    :param ga: 評価を行うgenomClass
    :return: 評価処理をしたgenomClassを返す
    """
    if select_num == i + 1:
        return 1.0
    else:
        return 0

def select(ga, elite):
    """選択関数です。エリート選択を行う
    評価が高い順番にソートを行った後、一定以上
    :param ga: 選択を行うgenomClassの配列
    :return: 選択処理をした一定のエリート、genomClassを返す
    """
    # 現行世代個体集団の評価を高い順番にソート
    sort_result = sorted(ga, reverse=True, key=lambda u: u.evaluation)
    # 一定の上位を抽出
    result = [sort_result.pop(0) for i in range(elite)]
    return result


def crossover(ga_one, ga_second):
    """交叉関数です。二点交叉を行う
    :param ga: 交叉させるgenomClassの配列
    :param ga_one:
    :param ga_second:
    :return: 二つの子孫genomClassを格納したリスト返す
    """
    # 子孫を格納するリストを生成
    genom_list = []
    # 入れ替える二点の点を設定→[1:25]
    cross_one = random.uniform(-1, 1)
    # 遺伝子を取り出す
    one = ga_one.getGenom()
    second = ga_second.getGenom()
    # 交叉
    fixed_z3 = one + (second - one) * cross_one
    # genomClassインスタンスを生成して子孫をリストに格納
    genom_list.append(ga.genom(fixed_z3, 0))
    return genom_list

def mutation(ga_one):
    """突然変異関数
    :param ga: genomClass
    :return: 突然変異処理をしたgenomClassを返す"""
    genom_list = []
    ga_m = ga_one.getGenom()
    ga_m = torch.randn(1, GENOM_LENGTH, 1, 1, device=device)
    genom_list.append(ga.genom(ga_m, 0))
    return genom_list

def make_first_background():
    latent_dim = 100
    G = Generator(latent_dim)
    G.eval()
    G.to(device)
    G.load_state_dict(torch.load("app/static/model/G_model_128.pth", map_location=torch.device('cpu')))
    x_tensor = torch.randn(1, latent_dim, 1, 1, device=device)
    # 一番最初の現行世代個体集団を生成
    current_generation_individual_group = []
    for i in range(MAX_GENOM_LIST):
        gene = torch.randn(1, latent_dim, 1, 1, device=device)
        img = generate_img(G, gene)
        img.save('app/static/input/background_{0:04d}.png'.format(i))
        current_generation_individual_group.append(ga.genom(gene, 0))

    return current_generation_individual_group

# 選択された背景画像に星を描画
def new_image(coordinate, size, color, num):
    org_img = cv2.imread("app/static/input/output.png")
    height, width, _ = org_img.shape[:3]

    mask = cv2.imread("app/static/input/background_{0:04d}.png".format(num-1))
    mask = cv2.resize(mask, (int(width), int(height)))
    for i in range(len(coordinate)):
        x = coordinate[i][0]
        y = coordinate[i][1]
        if ((max(height, width)/2) // size[i]) / 2 > 150:
            mask2 = np.zeros((int(height), int(width), 3), np.uint8)
            mask2 = cv2.circle(mask2, (int(x), int(y)), int(((max(height, width)/80) // size[i]) / 2), star_color(color[i]), -1)
            mask2 = cv2.blur(mask2, (20, 20))
            mask = cv2.addWeighted(src1=mask, alpha=1.0, src2=mask2, beta=1.0, gamma=0)

    for i in range(len(coordinate)):
        x = coordinate[i][0]
        y = coordinate[i][1]
        mask = cv2.circle(mask, (int(x), int(y)), int(((max(height, width)/160) // size[i]) / 2), (255, 255, 255), -1)
    mask = mask.clip(0, 255)

    mask = np.array(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(mask)
    mask.save('app/static/input/output.png')

# 対話型遺伝的アルゴリズムを用いて新しい背景画像の候補を生成
def iGA(current_generation_individual_group, select_num):
    G = Generator(latent_dim)
    G.eval()
    G.to(device)
    G.load_state_dict(torch.load("app/static/model/G_model_128.pth", map_location=torch.device('cpu')))

    vtr = imgsim.Vectorizer()

    for i in range(MAX_GENOM_LIST):
        vec0 = vtr.vectorize(cv2.imread("app/static/input/background_{0:04d}.png".format(select_num-1)))
        vec1 = vtr.vectorize(cv2.imread("app/static/input/background_{0:04d}.png".format(i)))
        dist = imgsim.distance(vec0, vec1)
        current_generation_individual_group[i].setEvaluation(dist)

    sort_current_gene = sorted(current_generation_individual_group, reverse=False, key=lambda u: u.evaluation)
    next_gene = []

    next_gene.extend(crossover(sort_current_gene[0], sort_current_gene[1]))
    next_gene.extend(crossover(sort_current_gene[0], sort_current_gene[2]))
    next_gene.extend(mutation(sort_current_gene[2]))
    next_gene.extend(mutation(sort_current_gene[3]))

    c = 0
    for k in next_gene:
        g = k.getGenom()
        img = generate_img(G, g)
        img.save("app/static/input/background_{0:04d}.png".format(c))
        c += 1

    return next_gene

if __name__ == '__main__':
    iGA()

