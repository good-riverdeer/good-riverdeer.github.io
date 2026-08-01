---
layout: post
title: "[paper-review] Mask TextSpotter v3: Segmentation Proposal Network for Robust Scene Text Spotting"
date: 2021-11-22 18:20 +0900
description: "Mask TextSpotter v3 논문 리뷰: Segmentation Proposal Network 기반 장면 텍스트 검출/인식"
thumbnail: https://images.velog.io/images/riverdeer/post/305c9289-3803-46cc-904a-6b7ea8d7f8d0/image.png
tags: deep_learning ocr scene_text segmentation
categories: [paper-review, Computer Vision]
related_posts: false
---

[Liao, M., Pang, G., Huang, J., Hassner, T., & Bai, X. (2020). Mask textspotter v3: Segmentation proposal network for robust scene text spotting. In _Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16 (pp. 706-722)._ Springer International Publishing.](https://link.springer.com/chapter/10.1007%2F978-3-030-58621-8_41)

_개인적인 논문해석을 포함하고 있으며, 의역 및 오역이 남발할 수 있습니다. 올바르지 못한 내용에 대한 피드백을 환영합니다 :)_

---

## 1. Introduction

**[@ Scene text spotter]**
- 최근 scene text spotting task에는 end-to-end 학습 방식의 딥러닝이 많이 적용되고 있음
- 좋은 scene text spotting 아래 세 가지 능력을 갖추어야 함
  - ***Rotation robustness***: 텍스트가 이미지 축에 잘 정렬되어 있지 않았을 때에 강건함
  - ***Aspect ratio robustness***: non-Latin scripts에는 주로 word 단위보다는 긴 텍스트 라인으로 텍스트 인스턴스가 구성되어 있는데, 이처럼 다양한 텍스트 인스턴스 종횡비에 강건함
  - ***Shape robustness***: Logo같은 텍스트에 주로 나타나는 일반적이지 않은 모양의 텍스트에 강건함

**[@ Mask TextSpotter series]**

- Region Proposal Network(RPN)의 한계
  1. manually pre-designed anchors를 사용하므로 극단적인 종횡비를 가지는 텍스트 인스턴스를 포착하기 쉽지 않음
  2. RPN이 생성하는 axis-aligned rectangular proposals는 그 box안에 인접한 다른 텍스트 인스턴스들이 함께 포함되는 경우가 많음

<img src="https://images.velog.io/images/riverdeer/post/b150f82b-a6ef-450c-85f2-16172a94593c/image.png" style="margin:50px 0 10px 0">
_anchor-box, [출처](https://herbwood.tistory.com/10)_

<img src="https://images.velog.io/images/riverdeer/post/548a99bb-cb6e-4cfe-966e-3757c9cd9e23/image.png" style="margin:50px 0 10px 0">
_2번 한계의 예시_
<br>

- 선행연구, **Mask TextSpotter v1**, **Mask TextSpotter v2**에서는 Region Proposal Network (RPN)을 통해 RoI feature를 추출하고 이 RPN이 제안한 proposal box들에 detection과 recognition을 수행
  - ***rotation robustness***, ***shape robustness***를 갖출 수 있는 RPN을 제안
  - But, ***aspect ratio robustness***까지 모두 갖추지는 못했음

**[@ Segmentation Proposal Network (SPN)]**

- Segmentation Proposal Network을 통해 정확한 polygonal 형태의 proposal 표현을 할 수 있음
- 더 나아가 정확한 형태의 proposal 표현을 통해 **hard RoI masking** 방법을 적용, 인접한 텍스트 인스턴스나 배경 노이즈의 간섭을 억제할 수 있음

**[@ Contributions]**
- **Segmentation Proposal Network(SPN)**
  - 극단적인 종횡비나 특이한 형태를 가진 텍스트 인스턴스를 정확하게 포착할 수 있는 SPN를 제안
- **hard RoI masking**
  - SPN이 생성해낸 proposal에 적용하여 배경 픽셀이나 인접한 다른 텍스트 인스턴스가 유발할 수 있는 노이즈를 제거
- **Mask TextSpotter v3**
  - rotation, aspect ratio, shape에 모두 robust한 text spotter 모델
  - 다양한 벤치마크에 높은 성능을 보임

---

## 2. Related Work
**[@ Two-stage scene text spotting]**
- [Wang et al.](https://ieeexplore.ieee.org/abstract/document/6460871?casa_token=mxjYSGOuaL8AAAAA:9-FiAX0KMCXYeu8DNSUVzYCINCdh3cz3OjLE3zaL_b7kx4F4W50KaWmCLMcbNwHwyLCzVqSBFVI) tried to detect and classify characters with CNNs.
- [Jaderberg et al.](https://link.springer.com/article/10.1007%2Fs11263-015-0823-z) proposed a scene text spotting method
  - proposal generation module
  - **a random forest classifier** to filter proposals
  - a CNN-based regression module for refining the proposals
  - a CNN-based word classifier for recognition
- [TextBoxes](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14202) and [TextBoxes++](https://ieeexplore.ieee.org/abstract/document/8334248?casa_token=JkAfo7MVaU8AAAAA:SkR9jLyHXZIocPQ9ghfYenqQMpCdaO5xf9whHHm7992FIlYXE6ZX3dT-LuKEehbeYcSbLQOuxZs) **combined** thier proposed scene text detector **with CRNN**
- [Zhan et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhan_GA-DAN_Geometry-Aware_Domain_Adaptation_Network_for_Scene_Text_Detection_and_ICCV_2019_paper.html) proposed to apply **multi-modal spatial learning** into the scene text detection and recognition system.

**[@ End-to-end trainable scene text spotting]**
- [Mask TextSpotter v1](https://openaccess.thecvf.com/content_ECCV_2018/html/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.html) is **the first end-to-end** trainable arbitrary-shape scene text spotter
  - consisting of a detection module based on **Mask R-CNN** and character segmentation module for recognition.
- [Mask TextSpotter v2](https://openaccess.thecvf.com/content_ECCV_2018/html/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.html) extends [Mask TextSpotter v1](https://openaccess.thecvf.com/content_ECCV_2018/html/Pengyuan_Lyu_Mask_TextSpotter_An_ECCV_2018_paper.html) by applying a **spatial attention module** for recognition
  - spatial attention module: character 수준의 공간적 왜곡을 바로 잡아줄 수 있는 모듈
- [Qin et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Qin_Towards_Unconstrained_End-to-End_Text_Spotting_ICCV_2019_paper.html) also combine a **Mask R-CNN detector** and **an attention-based recognizer** to deal with arbitrary-shape text instances
  - [Qin et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Qin_Towards_Unconstrained_End-to-End_Text_Spotting_ICCV_2019_paper.html)의 연구에선 mask map을 recognition에 성능향상을 위해 RoI feature에 대해 RoI masking을 수행
  - 하지만, mask map을 생성하는 데 RPN을 사용하기 때문에 proposals을 생성하는 데 부정확한 결과를 만들어낼 수 있음 (Introduction에서 밝힌 RPN의 단점)
- [Xing et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Xing_Convolutional_Character_Networks_ICCV_2019_paper.html) propose to **simultaneously detect/recognize** the characters and the text instances, using the text instance detection results to group the characters.
- [TextDragon](https://openaccess.thecvf.com/content_ICCV_2019/html/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.html) detects and recognizes text instances **by grouping and decoding** a series of local regions along with **their centerline**

**[@ Segmentation-based scene text detectors]**
- [Zhang et al.](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhang_Multi-Oriented_Text_Detection_CVPR_2016_paper.html) **first use FCN** to obtain the salient map of the text region
  - then estimate the text line hypotheses by combining the salient map and character components.
  - Finally, another FCN predicts the centroid of each character to remove the false hypotheses.
- [He et al.](https://arxiv.org/abs/1603.09423) propose Cascaded Convolutional Text Networks (CCTN) for text center lines and text regions.
- [PSENet](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html) adopts a progressive scale expansion algorithm to get the bounding boxes from multi-scale segmentation maps.
- [DB](https://ojs.aaai.org/index.php/AAAI/article/view/6812) proposes a differentiable binarization module for a segmentation network.
- 본 논문에서는 기존 Segmentation-based scene text detector에 비해 다양한 단서와 추가적인 모듈을 결합하여 detection task를 수행함
  - proposal generation에 segmentation network을 사용한다는 점을 강조할 수 있음

---

## 3. Methodology
![](https://images.velog.io/images/riverdeer/post/305c9289-3803-46cc-904a-6b7ea8d7f8d0/image.png)

Mask TextSpotter v3 consists of ...
- a ResNet-50 backbone, a Segmentation Proposal Network (SPN) for proposal generation
- a Fast R-CNN module for refining proposals
- a text instance segmentation module for accurate detection
- a character segmentation module and a spatial attentional module for recognition

추가적으로 Mask TextSpotter v3는 RoI feature의 형태를 다각형, polygonal 형태로 생성하기 때문에 정확한 detection을 할 수 있고 recognition의 성능에도 좋은 영향을 줄 수 있음

### 3.1. Segmentation proposal network
- U-Net의 형태를 사용, 다양한 크기의 다양한 feature를 사용
- SPN의 output $$F$$는 위 feature들을 결합하여 $${H\over 4} \times{W\over 4}$$ 크기로 생성됨
  - $$H, W$$는 각각 입력 이미지의 높이, 너비
- $$F$$를 통해 Segmentation을 수행하여 최종적으로 $$1\times H \times W$$ 크기의 predict segmentation map $$S$$를 생성함

<p align="center">
  <img src="https://images.velog.io/images/riverdeer/post/ddac7eff-8b8b-4cdb-884e-810d896de4ac/image.png" style="margin:50px 0 10px 0">
  <i>최종 Segmentation 수행 모듈의 구조</i>
</p>

**[@ Segmentation label generation]**
Segmentation 성능 향상을 위해 text instance들의 크기를 축소시킴으로써 인접한 text instance들을 분리하려는 테크닉이 일반적임

- _Vatti clipping algorithm_
  - $$d$$ pixel 만큼 텍스트 영역을 축소시키는 테크닉
  - the offset pixel $$d=A(1-r^2)/L$$
    - $$A$$는 텍스트 인스턴스 polygon의 면적
    - $$L$$은 텍스트 인스턴스 polygon의 둘레
    - $$r$$ is the shrink ratio

![](https://images.velog.io/images/riverdeer/post/2ab00cc8-ecc7-4725-a030-24efe777bf07/image.png)

**[@ Proposal generation]**
- 먼저 Segmentation map $$S$$를 이진화하여 binary map $$B$$를 계산

$$
B_{i,j} = \begin{cases}
1 & \mathrm{if} \space S_{i,j} \ge t,\\
0 & \mathrm{otherwise.}
\end{cases}
\\ \mathrm{Here,} \space t=0.5
$$
- 이후 _Vatti clipping algorithm_을 원복

$$\begin{matrix}
\hat d = \hat A \times \hat r / \hat L\\
\mathrm {Here,} \space \hat r = 3.0
\end{matrix}$$

### 3.2. Hard RoI masking
- 직사각형의 binary map $$B$$에서 각 text instance RoI feature와 크기가 동일한 polygon mask $$M$$을 생성
$$
M=\begin{cases}
1, & \mathrm{if \space in \space the \space polygon \space region}\\
0, & \mathrm{else}
\end{cases}
$$
- RoI feature $$R=R_0 * M$$, $$*$$는 element-wise multiplication
- hard RoI masking을 통해 배경 영역이나 인접한 다른 텍스트 인스턴스들의 방해를 억제할 수 있음
- 결과적으로 detection과 recognition 모두에 성능 향상을 도모할 수 있음

### 3.3. Detection and recognition
- text detection and recognition의 설계는 Mask TextSpotter v2의 것과 동일
  - Mask TextSpotter v2가 당시 최고의 detection, recognition 모델임
  - RPN-based scene text spotter와 (본 논문에서 제안하는) SPN-based scene text spotter의 공정한 비교를 위함
- hard RoI masking을 거친 masked RoI features는 Fast R-CNN의 입력으로 주어지고 localization을 가다듬고 character segmentation module과 spatial attentional module로 recognition을 수행

### 3.4. Optimization
$$
L = L_s + \alpha_1L_{rcnn}+\alpha_2L_{mask}
$$

- $$L_{rcnn}$$ is defined in Fast R-CNN
- $$L_{mask}$$ is defined in Mask TestSpotter v2, consisting of a **text instance segmentation loss**, **a chracter segmentation loss**, and **a spatial attentional decoder loss**.
- $$L_s$$ indicates the SPN loss
  - SPN loss엔 Dice loss를 사용
  $$
  I=\sum(S*G); \space U=\sum S + \sum G; \space L_s=1-{2.0\times I\over U}
  $$
  - $$S$$ is the segmentation map, $$G$$ is the target map, $$*$$ represents element-wise multiplication.
- $$\alpha_1=\alpha_2=1.0$$

---

## 4. Experiments

### 4.1. Datasets

- ***SynthText***
  - 800K 텍스트 이미지를 포함한 합성 데이터셋
  - annotations for word/character bounding boxes and text sequences.
- ***Rotated ICDAR 2013 dataset (RoIC13)***
  - ***ICDAR 2013*** 데이터셋에서 $$15^\circ, 30^\circ, 45^\circ, 60^\circ, 75^\circ, 90^\circ$$를 회전시켜 직접 제작
  - ***ICDAR 2013***의 텍스트 인스턴스들이 모두 수평적으로 정렬되어 있기 때문에 이 특성을 이용해 텍스트의 회전 방향에 대한 강건함(_rotation robustness_)을 테스트 할 수 있음
- ***MSRA-TD500***
  - 영어와 중국어로 구성된 multi-language scene text detection benchmark
  - 많은 수의 텍스트 인스턴스가 극단적인 종횡비로 구성됨
  - recognition annotations가 포함되지 않음
- ***Total-Text***
  - 다양한 형태의 텍스트 인스턴스, 가로세로 방향의 인스턴스, 곡선 형태의 텍스트들이 포함
  - polygonal bounding box와 transcription annotations 포함
- ***ICDAR 2015 (IC15)***
  - quadrilateral bounding boxes로 레이블 구성
  - 대부분의 이미지가 저해상도이고 작은 텍스트 인스턴스를 포함

### 4.2. Implementation details
**[@ Mask TextSpotter v2]**
- 공정한 비교를 위해 같은 학습 데이터와 Data augmentation 과정을 거침
- 한 가지 차이점
  - SPN이 더 극단적인 형태의 text instance에도 강건하기 때문에 rotation 각도 범위를 $$[-30^\circ, 30^\circ]$$에서 $$[-90^\circ, 90^\circ]$$로 확장

**[@ hyper-parameters & training details]**
- optimizer: SGD with a weight decay of 0.001 and momentum of 0.9
- ***SynthText***로 사전학습 수행 후 데이터셋을 조합하여 미세조정 수행
  - ***SynthText: ICDAR 2013: ICDAR 2015: SCUT: Total-Text $$= 2:2:2:1:1$$***
  - 학습에 사용되는 batch를 위 비율로 구성, 즉 batch를 8로 구성
  - pre-training
    - learning rate는 0.01로 시작
    - 100K, 200K iteration에서 각각 $$1/10$$씩 감소
  - fine-tuning
    - 학습 시와 동일한 환경을 사용
    - initial learning rate만 0.001로 시작
  - pre-training, fine-tuning 모두 250K번째의 가중치를 사용했음

### 4.3. Rotation robustness
자체적으로 구축한 ***RoIC13***에 테스트 수행

**[@ Detection task]**

![](https://images.velog.io/images/riverdeer/post/794cc67a-9e9a-48a2-9910-7979c4478a15/image.png)

**[@ End-to-end recognition task]**

![](https://images.velog.io/images/riverdeer/post/bc6d8c66-fba0-4ab4-8397-8dd7829b4911/image.png)

>- Evaluation protocol of **IC15** ([출처](https://rrc.cvc.uab.es/?ch=4&com=tasks))
  - ground truth bounding box와 50% 이상 겹치고 text 내용이 일치할 경우 true positive
  - 일부 작은 텍스트에 대해 **"do not care"**의 레이블이 되어있는 경우가 있음
    - ground truth bounding box와 50% 이상 겹치는 경우, 혹은 찾아내지 못하는 경우에도 evaluation에 포함되지 않는다.

### 4.4. Aspect ratio robustness
- 극단적인 종횡비의 text instance가 많이 출현하는 ***MSRA-TD500*** 데이터셋에 대해 평가 진행
- recognition annotation이 없기 때문에 detection task에 대한 평가만 수행

![](https://images.velog.io/images/riverdeer/post/12f3885a-994b-485f-9025-896f716082e7/image.png)

![](https://images.velog.io/images/riverdeer/post/2f896d1c-1180-44bd-bf99-187ee3579376/image.png)

### 4.5. Shape robustness

- horizontal, oriented, and curved 형태의 다양한 형태가 많이 포함된 ***Total-Text***에 대해 평가 진행

![](https://images.velog.io/images/riverdeer/post/c4972dec-aebf-4906-b389-7b3aadbb9af5/image.png)

![](https://images.velog.io/images/riverdeer/post/d0b4dc33-bdd9-4553-b3fb-c5c1ebc74952/image.png)

### 4.6. Small text instance robustness
- TextDragon이 `Strong`, `Weak` task에서는 가장 좋은 성능을 보였지만 일반적인 경우라고 볼 수 있는 `Generic` task에서 가장 좋은 성능을 큰 차이로 보임
  - `Strong`: `Weak`: `Generic` = text word의 종류 수 $$100: 1000+: 90k$$

![](https://images.velog.io/images/riverdeer/post/8f0dfea7-48bd-49a8-a440-ac15a3713040/image.png)

### 4.7. Ablation study
- options 1
  - direct: segmentation/binary map을 곧바로 사용
  - indirect: segmentation/binary map을 추가적인 레이어를 더해 후처리를 가하여 사용
- options 2
  - soft: soft probability map(값의 범위가 $$[0,1]$$)을 사용
  - hard: mask map의 값이 0과 1로만 구성된 것을 사용
- 논문에서 제안하는 hard RoI masking 방법이 의미가 있음

![](https://images.velog.io/images/riverdeer/post/84ae6000-bbd3-44bd-976f-55fe64eb48a9/image.png)

---

## 5. Conclusion

- end-to-end 학습 모델 Mask TextSpotter v3를 제안
  - SPN을 도입하여 정확한 텍스트 영역 폴리곤을 생성할 수 있음
- 다양한 데이터셋에 대한 검증 수행
  - ***Rotated ICDAR 2013*** 데이터셋에 rotation robustness 검증
  - ***MSRA-TD500*** 데이터셋에 aspect ratio robustness 검증
  - ***Total-Text*** 데이터셋에 shape robustness 검증
  - ***IC15*** 데이터셋에 작은 텍스트 인스턴스에 대한 강건함도 검증
