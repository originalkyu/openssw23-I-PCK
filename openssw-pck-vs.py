# pip install --upgrade scikit-image
# pip install --upgrade imutils

import cv2
import torch
import gc
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import time
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse

def psnr_test(ori_img, con_img):
    max_pixel = 255.0

    # MSE
    mse = np.mean((ori_img - con_img)**2)

    if mse == 0:
        return 100
    # PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

def makeDatas(filename):
    ori_img = cv2.imread('./datas/input/'+filename)
    noi_img1 = cv2.imread('./datas/noisy-output/noisy1-'+filename)
    noi_img2 = cv2.imread('./datas/noisy-output/noisy2-'+filename)
    fil_img1 = cv2.imread('./datas/output/filtered-1-'+filename)
    fil_img2 = cv2.imread('./datas/output/filtered-2-'+filename)
    NAF_img1 = cv2.imread('./datas/output/NAFNet-1-'+filename)
    NAF_img2 = cv2.imread('./datas/output/NAFNet-2-'+filename)

    imglist = {
        filename: ori_img,
        'noisy1-'+filename: noi_img1,
        'noisy2-'+filename: noi_img2,
        'filtered1-'+filename: fil_img1,
        'filtered2-'+filename: fil_img2,
        'NAFNet1-'+filename: NAF_img1,
        'NAFNet2-'+filename: NAF_img2
    }

    datas = []
    for name,img in imglist.items():
        datas.append(psnr_test(ori_img,img))
    
    for name,img in imglist.items():
        datas.append(structural_similarity(ori_img, img, channel_axis=2))
    
    return datas


def get_filtered_img1(frame):
    ksize = 3 # 3 5 7
    kernel = np.ones((ksize, ksize), np.float32)/(ksize*ksize)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    filtered_Y = cv2.filter2D(Y,-1,kernel)
    filtered_img = cv2.cvtColor(cv2.merge((filtered_Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
    return filtered_img

def get_filtered_img2(frame):
    ksize = 3 # 3 5 7

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    filtered_Y = cv2.medianBlur(Y, ksize)
    filtered_img = cv2.cvtColor(cv2.merge((filtered_Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
    return filtered_img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img):
    inp = img2tensor(img)
    model.feed_data(data={'lq': inp.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    return sr_img

def main():
    # filenames = os.listdir(os.path.abspath("./datas/input/"))
    filenames = os.listdir("./datas/input")
    # step1 데이터 구하기
    datas_list = []
    for name in filenames:
        datas_list.append(makeDatas(name))
    noisy1p = list(map(lambda datas: datas[1], datas_list))
    noisy2p = list(map(lambda datas: datas[2], datas_list))
    filtered1p = list(map(lambda datas: datas[3], datas_list))
    filtered2p = list(map(lambda datas: datas[4], datas_list))
    NAFNet1p = list(map(lambda datas: datas[5], datas_list))
    NAFNet2p = list(map(lambda datas: datas[6], datas_list))
    noisy1s = list(map(lambda datas: datas[7], datas_list))
    noisy2s = list(map(lambda datas: datas[8], datas_list))
    filtered1s = list(map(lambda datas: datas[9], datas_list))
    filtered2s = list(map(lambda datas: datas[10], datas_list))
    NAFNet1s = list(map(lambda datas: datas[11], datas_list))
    NAFNet2s = list(map(lambda datas: datas[12], datas_list))


    # speed1
    ori_img = cv2.imread('./datas/input/'+filenames[0])
    starttime = time.time()
    for i in range(10):
        get_filtered_img1(ori_img)
    t1 = time.time() - starttime
    
    # speed2
    starttime = time.time()
    for i in range(10):
        get_filtered_img2(ori_img)
    t2 = time.time() - starttime

    # speed3
    gc.collect()
    torch.cuda.empty_cache()
    opt_path = './NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)
    starttime = time.time()
    for i in range(10):
        single_image_inference(NAFNet, ori_img)
    t3 = time.time() - starttime

    # step2 
    fig = plt.figure(figsize=(12,12))
    plt.subplots_adjust(hspace=0.99, top=0.95, bottom=0.15, wspace=0.4)
    gs = fig.add_gridspec(4,1)

    # graph1. cv2.filter2D() vs NAFNet
    ax0 = fig.add_subplot(gs[0])
    ax0.set_title("Gaussian Noise:\ncv2.filter2D() vs NAFNet")
    ax0.set_xlabel("Inputs")
    ax0.set_ylabel("psnr")

    ax0.plot(filenames, noisy1p, label='noisy img')
    ax0.plot(filenames, filtered1p, label = 'cv2.filter2D()')
    ax0.plot(filenames, NAFNet1p, label='NAFNet')
    ax0.set_xticks(range(0,len(filenames),1), labels=filenames, rotation=15)
    ax0.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=6)
    # graph2. cv2.MedianBlur() vs NAFNet
    ax1 = fig.add_subplot(gs[1])
    ax1.set_title("Pepper and salt Noise:\ncv2.MedianBlur() vs NAFNet")
    ax1.set_xlabel("Inputs")
    ax1.set_ylabel("psnr")

    ax1.plot(filenames, noisy2p, label='noisy img')
    ax1.plot(filenames, filtered2p,label = 'cv2.MedianBlur()')
    ax1.plot(filenames, NAFNet2p,label='NAFNet')
    ax1.set_xticks(range(0,len(filenames),1), labels=filenames, rotation=15)
    ax1.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=6)
    # graph3. cv2.filter2D() vs NAFNet
    ax2 = fig.add_subplot(gs[2])
    ax2.set_title("Gaussian Noise:\ncv2.filter2D() vs NAFNet")
    ax2.set_xlabel("Inputs")
    ax2.set_ylabel("ssim")

    ax2.plot(filenames, noisy1s, label='noisy img')
    ax2.plot(filenames, filtered1s,label = 'cv2.filter2D()')
    ax2.plot(filenames, NAFNet1s,label='NAFNet')
    ax2.set_xticks(range(0,len(filenames),1), labels=filenames, rotation=15)
    ax2.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=6)
    # graph4. cv2.MedianBlur() vs NAFNet
    ax3 = fig.add_subplot(gs[3])
    ax3.set_title("Pepper and Salt Noise:\ncv2.MedianBlur() vs NAFNet")
    ax3.set_xlabel("Inputs")
    ax3.set_ylabel("ssim")

    ax3.plot(filenames, noisy2s, label='noisy img')
    ax3.plot(filenames, filtered2s,label = 'cv2.MedianBlur()')
    ax3.plot(filenames, NAFNet2s,label='NAFNet')
    ax3.set_xticks(range(0,len(filenames),1), labels=filenames, rotation=15)
    ax3.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=6)

    # graph5 speed
    plt.figure()
    bars = plt.bar(range(3),[math.log10(t1),math.log10(t2),math.log10(t3)])

    plt.xticks([0,1,2], ['filter2D()', 'MedianBlur()', 'NAFNet'])
    plt.ylabel('log10 of time')
    plt.title("Time of Processing 10 Images")
    plt.text(bars[0].get_x(), bars[0].get_y(),str(round(math.log10(t1),4)))
    plt.text(bars[1].get_x(), bars[1].get_y(),str(round(math.log10(t2),4)))
    plt.text(bars[2].get_x(), bars[2].get_y(),str(round(math.log10(t3),4)))

    plt.show()

if __name__ == "__main__":
    main()