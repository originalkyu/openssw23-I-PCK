# openssw23-I-PCK
2023년 1학기 '오픈소스SW입문' 개인 프로젝트입니다.

### Team Introduction
박찬규 201912325

### Topic Introduction
NAFNet을 이용한 디노이징과 opencv의 이미지 필터링(cv2.filter2D, cv2.medianBlur)을 이용한 디노이징 결과 비교하기
- 참고 리포지토리: https://github.com/megvii-research/NAFNet

### Results
![total1-input6](https://github.com/originalkyu/openssw23-I-PCK/assets/107669268/4760c87a-6609-48ed-abf6-174628d5458b)
- 왼쪽부터 차례대로 (1)원본 이미지, (2)원본에 노이즈를 추가한 이미지, (3)opencv filter2D 함수로 필터링한 이미지 (4) NAFNet을 이용해서 디노이징한 이미지

### Analysis/Visualization
Empty

### Installation
##### 환경:
- Windows 10
- python 3.9.16
- PyTorch 2.0.1
- CUDA 11.7

##### Installation:
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

##### Test:
```
python openssw-pck.py
```

### Presentation
Empty
