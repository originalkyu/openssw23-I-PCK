import cv2
import numpy as np
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import matplotlib.pyplot as plt
import gc
import random
import sys

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def display(img1, img2):
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 2, 1) 
    plt.title('Input image', fontsize=16)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.title('NAFNet output', fontsize=16)
    ax2.axis('off')
    ax1.imshow(img1)
    ax2.imshow(img2)

def single_image_inference(model, img, save_path):
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)


def get_noisy_img1(frame): # 가우시안 노이즈
    
    ngain = 50 # 1.0 - 100.0 가우시안 노이즈 정도
    height, width, _ = frame.shape

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    noisy = np.clip(Y + np.random.random((height, width))*ngain, 0, 255).astype(np.uint8)
    noisy_img = cv2.cvtColor(cv2.merge((noisy, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
    return noisy_img

def get_noisy_img2(frame): # pepper and salt noise

    ksize = 3 # 3 5 7
    rat_noise = 0.01 # 0.01 - 1.0
    height,width,_ = frame.shape
    num_noise = int(width*height*rat_noise)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    for i in range(num_noise):
        y = random.randint(0, height-1)
        x = random.randint(0, width-1)
        Y[y][x] = 255
    noisy_img = cv2.cvtColor(cv2.merge((Y, Cr, Cb)), cv2.COLOR_YCrCb2BGR)
    return noisy_img

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

def main():

    # step 0 path 설정
    inputpath = './datas/input/'
    outputpath = './datas/output/'
    noisyoutputpath = './datas/noisy-output/'
    filename = 'input10.png'

    opt_path = './NAFNet-width64.yml'

    filename = sys.argv[1]
    
    # step1 noisy image와 pepper and salt noist image 만들기
    input = inputpath + filename
    frame = cv2.imread(input) # frame : 원본 이미지
    
    noisy_img1 = get_noisy_img1(frame)
    noisy_img2 = get_noisy_img2(frame)
    cv2.imwrite(noisyoutputpath+'noisy1-'+filename,noisy_img1)
    cv2.imwrite(noisyoutputpath+'noisy2-'+filename,noisy_img2)

    # show
    # show_frame = np.hstack((frame,noisy_img2))
    # cv2.waitKey(0)
    # cv2.imshow('noisiy',show_frame)
    # cv2.waitKey(0)

    # step2 opencv의 필터를 이용한 디노이징
    
    filtered_img1 = get_filtered_img1(noisy_img1)
    filtered_img2 = get_filtered_img2(noisy_img2)
    cv2.imwrite(outputpath+'filtered-1-'+filename,filtered_img1)
    cv2.imwrite(outputpath+'filtered-2-'+filename,filtered_img2)
    # filtered_img = cv2.filter2D(noisy, -1, kernel)
    # cframe = np.hstack((img, noisy, filtered_img))
    # # cv2.imshow('Original, Noisy, Filetered', cframe)


    #
    # print((img-noisy).sum())
    # print((img-filtered_img).sum())
    # print((img-img).sum())

    # cv2.imwrite(outputpath+'noisy-'+filename,noisy)
    # cv2.imwrite(outputpath+'noisy-filtered-'+filename,filtered_img)



    ##########
    # step 3. NAFNet 적용
    gc.collect()
    torch.cuda.empty_cache()

    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)

    # # img_input = imread(input)
    # img_input = imread(outputpath+'noisy-'+filename)
    img_input1 = cv2.cvtColor(noisy_img1, cv2.COLOR_BGR2RGB)
    img_input2 = cv2.cvtColor(noisy_img2, cv2.COLOR_BGR2RGB)
    inp1 = img2tensor(img_input1)
    inp2 = img2tensor(img_input2)
    single_image_inference(NAFNet, inp1, outputpath+'NAFNet-1-'+filename)
    single_image_inference(NAFNet, inp2, outputpath+'NAFNet-2-'+filename)
    # img_output = imread(outputpath+'NAFNet-'+filename)
    # # display(img_input, img_output)


    ## step4. 비교 이미지 저장
    NAFNetimg1 = cv2.imread(outputpath+'NAFNet-1-'+filename)
    NAFNetimg2 = cv2.imread(outputpath+'NAFNet-2-'+filename)
    cframe1 = np.hstack((frame, noisy_img1, filtered_img1, NAFNetimg1))
    cframe2 = np.hstack((frame, noisy_img2, filtered_img2, NAFNetimg2))
    cv2.imwrite('./datas/total1-'+filename,cframe1)
    cv2.imwrite('./datas/total2-'+filename,cframe2)

    # img1 = imread(input)
    # img_input1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # inp = img2tensor(img_input1)
    # single_image_inference(NAFNet, inp, outputpath+'test-'+filename)
if __name__ == "__main__":
    main()