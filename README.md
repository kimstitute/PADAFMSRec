# PADAFMSRec

RecBole 1.2.1을 기반으로 한 멀티모달 순차 추천 연구용 포크입니다.

이 저장소의 목적은 Amazon Reviews 기반 데이터셋에서 동일한 RecBole 데이터 분할과 동일한 전처리 산출물을 사용해 다음 계열의 모델을 비교하는 것입니다.

- ID-only sequential recommendation baseline
- structured feature 기반 sequential recommendation baseline
- text/image dense cache를 사용하는 multimodal sequential recommendation baseline
- PADAFRec 및 관련 비교군

원본 프레임워크는 [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole)입니다. 본 저장소는 RecBole의 전체 프레임워크를 유지하면서 연구 모델과 실험용 데이터 로딩 계약을 추가한 포크입니다.

---

## 1. 원본 RecBole 대비 수정 사항

### 1.1 추가한 모델

| 모델명 | 파일 | 입력 모달리티 | 핵심 목적 |
| --- | --- | --- | --- |
| `PADAFRec` | `recbole/model/sequential_recommender/padafrec.py` | `category + brand + text + image` | ID representation을 value path로 유지하고, side feature를 attention score 조절에 사용하는 PADAF 계열 모델 |
| `IRIS` | `recbole/model/sequential_recommender/iris.py` | `category + brand + text + image` | IRIS 구조를 RecBole 1.2.1 및 PADAF dense cache 구조에 맞게 포팅한 비교군 |
| `SASRecD` | `recbole/model/sequential_recommender/sasrecd.py` | `category`, `category + brand`, 또는 `category + brand + text + image` | DIF-SR 스타일 score-level feature fusion 비교군 |
| `SASRecFDense` | `recbole/model/sequential_recommender/sasrecfdense.py` | `category + brand + text + image` | 모든 모달리티를 Transformer 입력 전에 concat 후 linear projection하는 early-fusion baseline |
| `GMUSASRecFDense` | `recbole/model/sequential_recommender/gmusasrecfdense.py` | `category + brand + text + image` | 모든 모달리티를 GMU 방식으로 early fusion하는 baseline |
| `SASRec` | RecBole 원본 | ID only | side feature 없이 item id sequence만 사용하는 기본 baseline |

### 1.2 추가한 layer

| 파일 | 내용 |
| --- | --- |
| `recbole/model/padaf_layers.py` | PADAFRec용 decoupled attention layer |
| `recbole/model/iris_layers.py` | IRIS 포팅용 attention/layer 구현 |
| `recbole/model/dif_layers.py` | DIF-SR/SASRecD용 feature score fusion layer |

### 1.3 추가한 모델 설정 파일

`recbole/properties/model/` 아래에 다음 설정 파일을 추가했습니다.

- `PADAFRec.yaml`
- `IRIS.yaml`
- `SASRecD.yaml`
- `SASRecFDense.yaml`
- `GMUSASRecFDense.yaml`

원본 RecBole의 `SASRec.yaml`도 그대로 사용할 수 있습니다.

### 1.4 dense cache 로딩 계약 추가

`PADAFRec`, `IRIS`, `SASRecD(full)`, `SASRecFDense`, `GMUSASRecFDense`는 전처리된 dense feature cache를 사용할 수 있습니다.

현재 dense cache 계약은 다음과 같습니다.

- `image_features.npy`: shape `(n_items + 1, 2048)`
- `text_features.npy`: shape `(n_items + 1, 768)`
- row `0`은 padding item용 zero vector
- row index는 RecBole remapping 이후의 `item_id`와 일치해야 함
- 즉, `.item` 파일의 item 수가 `n_items`라면 dense matrix row 수는 반드시 `n_items + 1`이어야 함

모델 내부에서는 dense matrix row 수가 `dataset.item_num`과 맞는지 검사합니다. 단, row 순서의 의미적 동일성은 전처리 단계에서 보장해야 합니다.

### 1.5 추가한 테스트

`tests/model/` 아래에 모델 등록, forward/loss, dense cache row count, fusion layer 동작을 검증하는 테스트를 추가했습니다.

대표 테스트 파일:

- `test_padaf_layers.py`
- `test_padafrec_registration.py`
- `test_iris_layers.py`
- `test_iris_registration.py`
- `test_difsr_layers.py`
- `test_sasrecd_registration.py`
- `test_sasrecfdense_registration.py`
- `test_gmusasrecfdense_registration.py`

---

## 2. 데이터셋 형식

RecBole sequential recommendation 형식을 사용합니다.

예를 들어 데이터셋 이름이 `Beauty_and_Personal_Care_PADAF`라면 다음 구조를 기대합니다.

```text
dataset/
└── Beauty_and_Personal_Care_PADAF/
    ├── Beauty_and_Personal_Care_PADAF.inter
    ├── Beauty_and_Personal_Care_PADAF.item
    ├── u_id_mapping.csv
    ├── i_id_mapping.csv
    └── padaf/
        ├── text_features.npy
        └── image_features.npy
```

### 2.1 `.inter` 필수 컬럼

```text
user_id:token
item_id:token
rating:float
timestamp:float
```

### 2.2 `.item` 권장 컬럼

멀티모달/side-feature 모델은 다음 컬럼을 사용합니다.

```text
item_id:token
category:token
brand:token
text_cache_key:token
image_cache_key:token
text_available:float
image_available:float
```

`SASRec` 같은 ID-only baseline은 `.item`을 로드하지 않도록 설정할 수 있습니다.

---

## 3. 설치

Python 3.11.9 환경에서 주로 검증했습니다.

```bash
git clone https://github.com/kimstitute/PADAFMSRec.git
cd PADAFMSRec
pip install -e . --verbose
```

GPU 환경에서는 사용 중인 CUDA/PyTorch 조합에 맞게 PyTorch를 먼저 설치하는 것을 권장합니다.

---

## 4. 실행 방법

### 4.1 공통 실행 명령

```bash
python run_recbole.py \
  --model PADAFRec \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/PADAFRec_Beauty_and_Personal_Care_PADAF.yaml
```

`--dataset` 값은 데이터셋 폴더명 및 `.inter`, `.item` prefix와 일치해야 합니다.

예:

```text
dataset/Beauty_and_Personal_Care_PADAF/Beauty_and_Personal_Care_PADAF.inter
dataset/Beauty_and_Personal_Care_PADAF/Beauty_and_Personal_Care_PADAF.item
```

이면 실행 시:

```bash
--dataset Beauty_and_Personal_Care_PADAF
```

을 사용합니다.

### 4.2 ID-only SASRec 실행 예시

side feature와 dense cache를 사용하지 않는 일반 SASRec baseline입니다.

```yaml
data_path: /content/padaf_workspace/dataset
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
load_col:
  inter: [user_id, item_id, rating, timestamp]
MAX_ITEM_LIST_LENGTH: 50
loss_type: CE
n_layers: 2
n_heads: 2
hidden_size: 64
inner_size: 256
hidden_dropout_prob: 0.5
attn_dropout_prob: 0.5
train_batch_size: 1024
eval_batch_size: 256
epochs: 10
stopping_step: 5
eval_args:
  split:
    LS: valid_and_test
  order: TO
  group_by: user
  mode: full
metrics: [Recall, MRR, NDCG, Hit, Precision]
topk: [5, 10, 20]
valid_metric: NDCG@10
```

실행:

```bash
python run_recbole.py \
  --model SASRec \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/SASRec_Beauty_and_Personal_Care_PADAF.yaml
```

### 4.3 PADAFRec 실행 예시

```yaml
data_path: /content/padaf_workspace/dataset
load_col:
  inter: [user_id, item_id, rating, timestamp]
  item: [item_id, category, brand, text_cache_key, image_cache_key, text_available, image_available]
selected_features: [category, brand, text, image]
structured_features: [category, brand]
dense_features: [text, image]
dense_feature_paths:
  text: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/text_features.npy
  image: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/image_features.npy
use_category_aux: true
use_brand_aux: true
category_aux_field: category
brand_aux_field: brand
lambda_cat: 0.1
lambda_brand: 0.05
```

실행:

```bash
python run_recbole.py \
  --model PADAFRec \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/PADAFRec_Beauty_and_Personal_Care_PADAF.yaml
```

### 4.4 IRIS 실행 예시

```yaml
selected_features: [category, brand, text, image]
structured_features: [category, brand]
dense_features: [text, image]
dense_feature_paths:
  text: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/text_features.npy
  image: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/image_features.npy
attribute_hidden_size: [64, 64, 64, 64]
fusion_type: sum
combine_type: mean
```

실행:

```bash
python run_recbole.py \
  --model IRIS \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/IRIS_Beauty_and_Personal_Care_PADAF.yaml
```

### 4.5 DIF-SR / SASRecD 실행 예시

이 저장소에서는 RecBole 내부 모델명을 `SASRecD`로 사용합니다. 실험표나 논문에서는 DIF-SR 비교군으로 표기할 수 있습니다.

#### canonical: category only

```yaml
selected_features: [category]
structured_features: [category]
dense_features: []
attribute_hidden_size: [64]
fusion_type: gate
attribute_predictor: linear
auxiliary_features: [category]
lamdas: [10]
```

#### structured: category + brand

```yaml
selected_features: [category, brand]
structured_features: [category, brand]
dense_features: []
attribute_hidden_size: [64, 64]
fusion_type: gate
attribute_predictor: linear
auxiliary_features: [category, brand]
lamdas: [10, 10]
```

#### full: category + brand + text + image

```yaml
selected_features: [category, brand, text, image]
structured_features: [category, brand]
dense_features: [text, image]
dense_feature_paths:
  text: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/text_features.npy
  image: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/image_features.npy
attribute_hidden_size: [64, 64, 64, 64]
fusion_type: gate
attribute_predictor: linear
auxiliary_features: [category, brand]
lamdas: [10, 10]
```

실행:

```bash
python run_recbole.py \
  --model SASRecD \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/SASRecD_Beauty_and_Personal_Care_PADAF_full.yaml
```

### 4.6 SASRecFDense 실행 예시

`SASRecFDense`는 `item embedding`, `category`, `brand`, `text`, `image`를 Transformer 입력 전에 concat한 뒤 linear projection합니다.

```yaml
selected_features: [category, brand, text, image]
structured_features: [category, brand]
dense_features: [text, image]
dense_feature_paths:
  text: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/text_features.npy
  image: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/image_features.npy
```

실행:

```bash
python run_recbole.py \
  --model SASRecFDense \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/SASRecFDense_Beauty_and_Personal_Care_PADAF.yaml
```

### 4.7 GMUSASRecFDense 실행 예시

`GMUSASRecFDense`는 `item embedding`, `category`, `brand`, `text`, `image` branch를 GMU 방식으로 가중합한 뒤 SASRec Transformer에 입력합니다.

```yaml
selected_features: [category, brand, text, image]
structured_features: [category, brand]
dense_features: [text, image]
dense_feature_paths:
  text: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/text_features.npy
  image: /content/padaf_workspace/dataset/Beauty_and_Personal_Care_PADAF/padaf/image_features.npy
```

실행:

```bash
python run_recbole.py \
  --model GMUSASRecFDense \
  --dataset Beauty_and_Personal_Care_PADAF \
  --config_files /path/to/GMUSASRecFDense_Beauty_and_Personal_Care_PADAF.yaml
```

---

## 5. Colab 사용 흐름

BDP 작업 루트의 `colab_trainer.ipynb`는 다음 작업을 자동화하기 위해 작성했습니다.

1. Google Drive에 저장된 전처리 산출물 복원
2. `kimstitute/PADAFMSRec` 클론 또는 업데이트
3. 선택한 모델에 맞는 RecBole config 생성
4. 학습 실행
5. stdout log, RecBole log, config, manifest, checkpoint를 실험 폴더로 저장

주요 설정값:

```python
DATASET = "Beauty_and_Personal_Care"
MODEL = "PADAFRec"  # PADAFRec / SASRec / IRIS / SASRecD / DIF-SR / SASRecFDense / GMUSASRecFDense
DIFSR_VARIANT = "full"  # SASRecD 전용: canonical / structured / full
DRIVE_ROOT = "/content/drive/MyDrive/BDP/Datasets3"
EXPERIMENT_ROOT = "/content/drive/MyDrive/BDP/Experiments"
SMOKE_RUN = False
```

일반 SASRec을 돌릴 때는 다음처럼 설정하면 됩니다.

```python
MODEL = "SASRec"
```

이 경우 dense cache 파일은 필요하지 않습니다.

---

## 6. 실험 비교 시 주의사항

### 6.1 모달리티 동일성과 objective 동일성은 별개

`PADAFRec`, `SASRecD`는 auxiliary objective를 사용할 수 있습니다. 반면 `IRIS`, `SASRecFDense`, `GMUSASRecFDense`, `SASRec`는 기본적으로 auxiliary objective 없이 next-item prediction만 수행합니다.

따라서 결과를 해석할 때 다음을 분리해서 보고해야 합니다.

- 사용 모달리티: `ID`, `category`, `brand`, `text`, `image`
- fusion 위치: attention 이전 / attention score level / attention 이후
- auxiliary loss 사용 여부
- dense cache encoder: text는 BERT, image는 ResNet 등

### 6.2 dense cache row order 검증

모델은 dense matrix row 수가 RecBole item 수와 일치하는지 검사합니다. 하지만 row `i`가 정확히 RecBole remapped item id `i`에 해당하는지는 전처리 단계에서 보장해야 합니다.

권장 확인:

- `.item`의 `item_id:token` 순서
- `i_id_mapping.csv`
- `text_features.npy`, `image_features.npy` row 수
- row 0 padding 여부

### 6.3 동일 데이터 분할 유지

공정 비교를 위해 모든 모델에서 다음 설정을 동일하게 유지하는 것을 권장합니다.

```yaml
eval_args:
  split:
    LS: valid_and_test
  order: TO
  group_by: user
  mode: full
MAX_ITEM_LIST_LENGTH: 50
loss_type: CE
metrics: [Recall, MRR, NDCG, Hit, Precision]
topk: [5, 10, 20]
valid_metric: NDCG@10
seed: 42
reproducibility: true
```

---

## 7. 테스트

로컬에서 주요 추가 모델 테스트를 실행하려면:

```bash
PYTHONPATH=. pytest \
  tests/model/test_padaf_layers.py \
  tests/model/test_padafrec_registration.py \
  tests/model/test_iris_layers.py \
  tests/model/test_iris_registration.py \
  tests/model/test_difsr_layers.py \
  tests/model/test_sasrecd_registration.py \
  tests/model/test_sasrecfdense_registration.py \
  tests/model/test_gmusasrecfdense_registration.py \
  -q
```

최근 검증 결과:

```text
49 passed
```

---

## 8. 원본 RecBole 인용

본 저장소는 RecBole 1.2.1을 기반으로 합니다. RecBole을 사용하는 연구에서는 원본 프로젝트의 라이선스와 인용 정보를 함께 확인해야 합니다.

- RecBole GitHub: https://github.com/RUCAIBox/RecBole
- RecBole paper: https://arxiv.org/abs/2011.01731
