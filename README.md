# openssw23-I-PCK
2023년 1학기 '오픈소스SW입문' 개인 프로젝트입니다.

## Team Introduction
박찬규 201912325

## Topic Introduction
주제: NAFNet을 이용한 디노이징과 opencv의 이미지 필터링(cv2.filter2D, cv2.medianBlur)을 이용한 디노이징 결과를 비교.  
- 두 방식으로 처리한 이미지를 시각적으로 비교합니다.  
- 두 방식으로 처리한 이미지를 PSNR을 이용해서 정량적으로 비교합니다.  
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
Empty

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
  
#### 필요한 설치

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
python3 setup.py develop --no_cuda_ext
```

* pretrained_model 다운로드   
다음의 코드를 프롬프트에 입력해서 "experiments" 폴더를 다운로드한다.
```
$ python
import gdown
gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
exit()
```
"/experiments/pretrained_models"폴더 안에 있는 파일 중 "README.md"가 아닌 파일의 이름을 다음과 같이 바꾼다.  
```
"NAFNet-SIDD-width64.pth" 
```
마지막파일을 클론한 리포지토리의 "experiments/pretrained_models" 폴더로 이동시킨다.  

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
