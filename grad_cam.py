#-------------------------------------------
# import
#-------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing import image
import json
from model_utils import get_model, get_model_inputsize
#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))
JSON_PATH = os.path.join(CUR_PATH, 'args.json')


#-------------------------------------------
# private functions
#-------------------------------------------


def get_args():
    with open(JSON_PATH, "r") as f:
        j = json.load(f)
    return j


def load_img(img_path, img_size):
    img = image.load_img(img_path, target_size=img_size)  # load img as PIL
    x = image.img_to_array(img)  # PIL to ndarray
    x = np.expand_dims(x, axis=0)
    return x


def make_heatmap(model, preprocess_input, decode_predictions, layer_name, img_path):

    #----------------------
    # load and preprocess img
    #----------------------
    input_img = load_img(img_path, get_model_inputsize(model))
    input_img = preprocess_input(input_img)

    #----------------------
    # model predict
    #----------------------
    preds = model.predict(input_img)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    pred_max_idx = np.argmax(preds[0])

    #----------------------
    # 最も確率の高い出力を取得
    # model.output[bacthsize, classes]
    #----------------------
    output = model.output[:, pred_max_idx]

    #----------------------
    # heatmapを確認したいレイヤーを選択
    #----------------------
    target_layer = model.get_layer(layer_name)

    #----------------------
    # 勾配を計算
    # grads.shape = (batchsize, h, w, c)
    # grads.shape == last_conv_layer.output.shape
    #----------------------
    grads = K.gradients(output, target_layer.output)[0]

    #----------------------
    # 特徴マップのチャネル方向の平均勾配
    #----------------------
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    #----------------------
    # 平均勾配と特徴マップを出力する関数を生成
    #----------------------
    iterate = K.function([model.input], [pooled_grads, target_layer.output[0]])

    #----------------------
    # 平均勾配と特徴マップを出力
    #----------------------
    pooled_grads_value, layer_output_value = iterate([input_img])

    _, _, layer_out_ch = layer_output_value.shape

    #----------------------
    # チャネルに平均勾配を掛け算
    #----------------------
    for i in range(layer_out_ch):
        layer_output_value[:, :, i] *= pooled_grads_value[i]

    #----------------------
    # 正規化
    #----------------------
    heatmap = np.mean(layer_output_value, axis=-1)  # (h,w,c) -> (h,w)
    heatmap = np.maximum(heatmap, 0)  # 　全てのピクセルに対してmax(pix, 0)を実行
    heatmap /= np.max(heatmap)  # 0-1に正規化

    return heatmap


def superimpose(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)  # 0-1 -> 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        src1=img, alpha=0.6, src2=heatmap, beta=0.3, gamma=0)

    return superimposed_img, heatmap, img


def main(args):
    target_model_name = args["model"]
    target_layer_name = args["layer"]
    input_img_path = args["img_path"]

    model, preprocess_input, decode_predictions = get_model(
        target_model_name)

    model.summary()

    heatmap = make_heatmap(model, preprocess_input, decode_predictions,
                           target_layer_name, input_img_path)

    superimposed_img, heatmap, img = superimpose(input_img_path, heatmap)

    #----------------------
    # plot imgs
    #----------------------
    plt.figure(figsize=(10, 5))  # figsize=(w,h)
    plt.subplot(1, 3, 1)

    # input img
    plt.title('img')
    plt.imshow(img)
    plt.axis('off')

    # heatmap
    plt.subplot(1, 3, 2)
    plt.title('heatmap')
    plt.imshow(heatmap, cmap='hot')
    plt.axis('off')
    # plt.colorbar()

    # heatmap + img
    plt.subplot(1, 3, 3)
    plt.title('heatmap+img')
    plt.imshow(np.uint8(superimposed_img))
    plt.axis('off')

    basename = os.path.basename(input_img_path)
    basename, _ = os.path.splitext(basename)
    plt.savefig(os.path.join(os.getcwd(), basename+"_gradcam.png"))

    plt.show()


if __name__ == '__main__':
    main(get_args())
