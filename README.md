# openssw23-I-PCK
2023년 1학기 '오픈소스SW입문' 개인 프로젝트입니다.

## Team Introduction
박찬규 201912325

## Topic Introduction
주제: NAFNet을 이용한 디노이징과 opencv의 이미지 필터링(cv2.filter2D, cv2.medianBlur)을 이용한 디노이징 결과를 비교.  
- 두 방식으로 처리한 이미지를 시각적으로 비교합니다.  
- 두 방식으로 처리한 이미지를 PSNR, SSIM을 이용해서 정량적으로 비교합니다.  
- NAFNet를 이용한 이미지 처리속도와 opencv의 이미지 필터링 함수를 이용한 이미지 처리속도를 비교합니다.  
- 참고 리포지토리: https://github.com/megvii-research/NAFNet  

* 설치 및 실행, input과 output 관련 사항은 아래의 Installation 참고  

## Results
예시1)  
![total1-input6](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/4760c87a-6609-48ed-abf6-174628d5458b)
  
예시2)  
![total1-input2](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/261ed282-6637-4846-b909-28d85aa5f3a0)
  
예시3)  
![total2-input6](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/6f6652e5-556b-4bee-8981-0a7938708e01)
  
예시4)  
![total2-input2](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/223ab2d9-1343-4ef3-853f-459dca7010ff)  

각 이미지의 구성은 왼쪽부터 차례대로  
  (1)원본 이미지   
  (2)원본에 노이즈를 추가한 이미지(가우시안 노이즈: 예시1, 예시2),(pepper and salt noise: 예시3, 예시4)   
  (3)opencv filter2D 함수로 필터링한 이미지(예시1, 예시2)  opencv medianBlur 함수로 필터링한 이미지(예시3, 예시4)   
  (4) NAFNet을 이용해서 디노이징한 이미지   

## Analysis/Visualization
[Graph1]  
![Figure_1](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/08d4f0aa-c8e7-45e0-8d6c-3b87f0f551ed)
[Graph2]  
![Figure_2](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/54ca2e3b-3065-4f5c-a9fe-2f8b5e5ed725)

#### 분석1: 시각적으로 비교하기
datas/total1-input1.png 부터 total1-input10.png 까지는 가우시안 노이즈가 추가된 10개의 이미지에서 opencv의 filter2D()함수와 NAFNet으로 디노이즈할 때의 결과물을 붙여둔 이미지이다. 10개의 이미지를 비교할 때 거의 대부분 NAFNet이 opencv의 filter2D() 함수보다 선명한 이미지를 만들어내는 것을 확인할 수 있었다.

datas/total2-input1.png 부터 total2-input10.png 까지는 Pepper and salt 노이즈가 추가된 10개의 이미지에서 opencv의 MedianBlur()와 NAFNet으로 디노이즈할 때의 결과물을 붙여둔 이미지이다. 10개의 이미지를 비교할 때 MedianBlur()함수는 거의 원본과 비슷한 결과물을 냈지만 NAFNet은 노이즈를 효과적으로 제거하지 못한 것을 확인할 수 있었다.

#### 분석2: [Graph1]의 첫 번째와 세 번째 그래프 해석
가우시안 노이즈가 추가된 11개의 이미지에서 opencv의 filter2D() 함수와 NAFNet를 이용하여 디노이징할 때 pnsr과 ssim을 기준으로 하여 비교하였다.
결과적으로 psnr과 ssim을 기준으로 본다면, filter2D() 함수와 NAFNet 두 가지 방법 중 어느 것이 낫다고 할 수 없었다. 이유는 다음과 같다. 먼저, 11개의 이미지를 디노이징할 때 두 방법을 이용한 결과물들의 각각의 psnr은 거의 유사하였다. 두 번째로, 두 방법을 이용한 결과물들의 각각의 ssim은 서로 엎치락뒤치락하여 5:6으로 NAFNet의 것이 높았지만 5:6 정도로는 유의미하게 더 낫다고 할 수 없다. 마지막으로 디노이징을 한 결과물들의 psnr 점수와 ssim 점수는 모두 디노이징을 하기 전보다 낮았다.   

#### 분석3: [Graph1]의 두 번째와 네 번째 그래프 해석
Pepper and salt 노이즈가 추가된 11개의 이미지에서 opencv의 MedianBlur() 함수와 NAFNet를 이용하여 디노이징할 때 psnr과 ssim을 기준으로 하여 비교하였다.
결과적으로 psnr 기준으로는 MedianBlur함수가 더 나았고, ssim 기준으로는 두 방식이 비슷한 성능을 냈다. 먼저, psnr 기준으로 11개 이미지에서 MedianBlur() 함수의 점수가 항상 더 높았다.
두 번째로, ssim 점수 기준으로 11개의 이미지에서 1개의 이미지를 제외하고 두 방식의 결과물의 점수는 거의 비슷했다.

#### 분석4: [Graph2] 해석
Graph2는 각각의 방식을 이용했을 때 10개의 이미지를 처리하는데 얼마나 걸리는지를 log10을 하여 표시한 것이다. filter2D()함수는 10^(-2) 초 정도가 걸렸고, MedianBlur() 함수는 10^(-2.4) 초 정도가 걸렸다. NAFNet 방식은 10^(0.75) 초 정도가 걸렸다. 즉, NAFNet을 이용해서 10개의 이미지를 처리하는 시간은 opencv의 함수를 이용할 때보다 약 100배가 넘게 오래 걸렸다.

#### 결론
(1) 시각적으로 비교할 때 가우시안 노이즈를 제거할 때는 NAFNet이 우수하고, Pepper and salt노이즈를 제거할 때는 opencv의 Median Blur 함수가 우수한 것을 명확하게 확인할 수 있었다.  
(2) 시각적으로 비교할 때와 다르게 psnr 점수로 비교할 때 Pepper and salt노이즈를 제거할 때 MedianBlur함수가 NAFNet을 이용할 때보다 낫다를 것을 확인할 수 있었다. 하지만 나머지의 비교에서는 psnr점수와 ssim점수 기준으로, 어느 방식이 유의미하게 낫다고하기 어려웠다.  
(3) 이미지를 처리하는 속도는 opencv가 NAFNet보다 압도적으로 빨랐다.  
(4) 결론 (1),(2)를 고려하면 psnr점수와 ssim점수는 시각적으로 확인했을 때의 차이를 잘 반영하지 못하는 지표이다.  
(5) 결론 (1),(2),(3)을 고려하면 Pepper and salt 노이즈를 제거할 때는 MedianBlur함수가 NAFNet보다 빠르고 효과가 좋다.  
(6) 결론 (1),(3)을 고려하면 가우시안 노이즈를 제거할 때 속도가 중요하다면 filter2D()함수를 이용하고, 시각적으로 나은 결과물이 필요하다면 NAFNet이 낫다.  

## Installation
* 이 리포지토리의 설치과정과 실행은 conda 4.12.0 환경에서 진행되었지만 conda 가상 환경이 아닌 환경에서도 진행이 가능함.  
#### 환경:
- Windows 10  
- python 3.9.16  
- PyTorch 2.0.1  
- CUDA 11.7  
  
### 실행 방법
다음의 절차를 따라 실행하면 "Result" 항목과 같은 결과를 얻을 수 있다.  
1. 현재 repository를 clone  
2. 아래의 "필요한 설치"의 과정을 따른다.  
3. 원하는 png 이미지를 ./datas/input 폴더에 넣는다.  
4. clone한 폴더로 이동하여  ./datas/input 폴더에 추가한 이미지의 이름과 확장자를 포함하여 openssw-pck.py를 다음과 같이 실행시킨다.   
```
python openssw-pck.py 추가한이미지이름.png
```
5. ./datas 폴더에서 total1-이미지이름.png 와 total2-이미지이름.png 파일을 확인한다.  
   total1-이미지이름.png : 가우시안 노이즈를 추가했을 때의 디노이징 결과  
   total2-이미지이름.png : pepper and salt 노이즈를 추가했을 때의 디노이징 결과  
   두 개의 결과 이미지는 README.md 파일의 Result 항목과 형식이 같다.  

* ./datas/input 폴더에서 input0.png ~ input10.png 파일을 샘플로 이용할 수 있다.  
* ./datas/input 폴더에서 input0.png ~ input10.png 파일을 샘플로 이용할 수 있다.    
  
#### 필요한 설치 및 설정

* PyTorch and CUDA 설치  
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
* 그 외의 설치
```
pip install -r requirements.txt
pip install --upgrade --no-cache-dir gdown
pip install matplotlib
pip install opencv-python
pip install numpy
pip install gdown
```

* 설정  
setup.py가 있는 위치에서 다음의 명령어를 실행시킨다.    
```
python3 setup.py develop --no_cuda_ext
```


* pretrained_model 다운로드   
다음의 코드를 프롬프트에 입력해서 파이썬을 실행시켜 "experiments" 폴더를 다운로드한다.
```
$ python
>>>import gdown
>>>gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
>>>exit()
```
"/experiments/pretrained_models"폴더 안에 있는 파일 중 "README.md"가 아닌 파일의 이름을 다음과 같이 바꾼다.  
```
"NAFNet-SIDD-width64.pth" 
```
이름을 바꾼 파일을 클론한 리포지토리의 "./experiments/pretrained_models" 폴더로 이동시킨다.  

#### Input과 output
* input  
  - ./datas/input 폴더에 사용자가 추가한 png 이미지 파일. 가로x세로가 256 x 256 정도의 작은 이미지가 권장됨
* output. 
  - ./datas/noisy-output/noisy1-추가한파일이름.png : 추가한 이미지에 가우시안 노이즈를 더한 이미지
  - ./datas/noisy-output/noisy2-추가한파일이름.png : 추가한 이미지에 pepper and salt 노이즈를 더한 이미지
  - ./datas/output/filtered-1-추가한파일이름.png : "noisy1-추가한파일이름.png" 파일을 cv2.filter2D() 함수로 필터링한 이미지
  - ./datas/output/filtered-2-추가한파일이름.png : "noisy2-추가한파일이름.png" 파일을 cv2.medianBlur() 함수로 필터링한 이미지
  - ./datas/output/NAFNet-1-추가한파일이름.png : "noisy1-추가한파일이름.png" 파일을 NAFNet으로 디노이징한 이미지
  - ./datas/output/NAFNet-2-추가한파일이름.png : "noisy2-추가한파일이름.png" 파일을 NAFNet으로 디노이징한 이미지
  - ./datas/total1-추가한파일이름.png : 위의 noisy1, filtered1, NAFNet-1 파일을 가로로 붙여둔 이미지
  - ./datas/total2-추가한파일이름.png : 위의 noisy2, filtered2, NAFNet-2 파일을 가로로 붙여둔 이미지

## Presentation
Empty
