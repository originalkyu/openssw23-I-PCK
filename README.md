# openssw23-I-PCK
2023년 1학기 '오픈소스SW입문' 개인 프로젝트입니다.

## Team Introduction
박찬규 201912325

## Topic Introduction
NAFNet을 이용한 디노이징과 opencv의 이미지 필터링(cv2.filter2D, cv2.medianBlur)을 이용한 디노이징 결과 비교하기
- 참고 리포지토리: https://github.com/megvii-research/NAFNet

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
  (3)opencv filter2D 함수로 필터링한 이미지(예시1, 예시2)  opencv medianBlur(Y, ksize) 함수로 필터링한 이미지(예시3, 예시4)   
  (4) NAFNet을 이용해서 디노이징한 이미지   

## Analysis/Visualization
Empty

## Installation
#### 환경:
- Windows 10
- python 3.9.16
- PyTorch 2.0.1
- CUDA 11.7

#### 필요한 설치:
```
pip install -r requirements.txt
pip install --upgrade --no-cache-dir gdown
pip install matplotlib
pip install opencv-python
pip install numpy
pip install gdown
python3 setup.py develop --no_cuda_ext
```

```
import gdown
gdown.download('https://drive.google.com/uc?id=14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR', "./experiments/pretrained_models/", quiet=False)
```
다운받은 pth 파일을 experiments/pretrained_models 폴던에 넣는다.

### 실행 방법
1. 현재 repository를 clone
2. 위의 환경을 갖춘 후 "필요한 설치"의 과정을 따른다.
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
* ./datas/input 폴더에서 input0.png ~ input10.png 파일을 샘플로 이용할 수 있다

## Presentation
Empty
