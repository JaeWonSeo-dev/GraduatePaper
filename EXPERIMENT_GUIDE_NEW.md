# NEW EXPERIMENTAL DESIGN - EXECUTION GUIDE

## 실험 개요

**총 10개 실험 (Combo1: 4, Combo2: 2, Combo3: 4)**

### Combo1: AFS → LSTM (Gate L1 제거)
- **목표**: Gate L1 정규화 제거 후 topk=20/30 양쪽 재평가
- **설정**:
  - `gate_l1_lambda = 0.0` (강제 비활성)
  - `ap_lambda = 0.02` (완만한 규제)
  - `topk = [20, 30]`
  - Neg-guard, hard negative mining, soft precision penalty 유지
- **실험 ID**:
  - `C1_k20_gateL1off`: topk=20
  - `C1_k30_gateL1off`: topk=30
- **실행 횟수**: 4회 (각 topk당 baseline N-1 + proposed N-2)

### Combo2: XGB-FS → LSTM (정밀도 유지 강제)
- **목표**: 정밀도 하락 금지, 리콜 상승
- **단계적 수행**:
  1. Baseline(N-1) 먼저 실행 → 검증 정밀도 `P_base` 기록
  2. Proposed(N-2): `prec_floor = 0.97 * P_base` 적용
- **설정**:
  - `prec_floor_train = 0.97 * P_base` (동적 계산)
  - `hard_neg_topq = 0.30`, `hard_neg_weight = 1.8` (강화)
  - `ap_lambda = 0.09` (XGB 조합 권장 범위)
  - `topk = 20`
- **정밀도 체크**:
  - Proposed 검증 정밀도 < P_base 시:
    - 콘솔에 경고 출력
    - `notes`에 `precision_not_maintained` 태그 추가
- **실험 ID**: `C2_k20_precMaint`
- **실행 횟수**: 2회 (baseline N-1 + proposed N-2)

### Combo3: MI+RF → MLP (편차 확대)
- **목표**: 정밀도 유지 비활성, topk=40/50 변화폭 확인
- **설정**:
  - `prec_floor_train = 0.0` (비활성)
  - `ap_lambda = 0.07`
  - `rf_trees = 700`
  - `topk = [40, 50]`
- **실험 ID**:
  - `C3_k40_varBoost`: topk=40
  - `C3_k50_varBoost`: topk=50
- **실행 횟수**: 4회 (각 topk당 baseline N-1 + proposed N-2)

---

## 실행 방법

### 방법 1: 직접 실행 (권장)

**단일 실행** (모든 10개 실험 자동 수행):

```powershell
python runner_combine.py --results_path /mnt/data/results_run.csv
```

**CLI 인자**:
- `--results_path`: 결과 CSV 저장 경로 (기본값: `/mnt/data/results_run.csv`)
- `--topk_c1`, `--topk_c2`, `--topk_c3`: 각 콤보의 topk 값
- `--gate_l1_lambda`: Gate L1 정규화 강도
- `--ap_lambda`: AP-Head 가중치
- `--prec_floor_train`: 정밀도 유지 하한
- `--hard_neg_topq`, `--hard_neg_weight`: Hard negative mining 설정
- `--neg_guard_lambda`, `--neg_gamma`: Negative guard 설정
- `--rf_trees`: Random Forest 트리 수
- `--exp_id`: 실험 식별자

### 방법 2: 자동화 스크립트 (개별 실행)

```powershell
python run_new_experiments.py
```

이 스크립트는 각 조합을 개별적으로 실행하며, 실패한 실험을 추적합니다.

---

## 출력물

### 1. 결과 CSV
- **경로**: `/mnt/data/results_run.csv`
- **모드**: Append (기존 파일에 추가)
- **컬럼**:
  - `exp_id`: 실험 식별자
  - `combo`: 콤보 이름
  - `variant`: N-1 (baseline) / N-2 (proposed)
  - `topk_used`: 사용된 topk 값
  - `notes`: 실험 설정 요약
  - `threshold`: 평가 임계값 (고정 0.5)
  - `precision`, `recall`, `f1`: 주요 메트릭
  - `roc_auc`, `pr_auc`: AUC 메트릭
  - `tn`, `fp`, `fn`, `tp`: Confusion matrix 값
  - 기타 보조 메트릭

### 2. Confusion Matrix 파일
- **경로**: `runs/cm_{exp_id}_{combo}_{variant}.txt`
- **예시**:
  - `runs/cm_C1_k20_gateL1off_1_AFS_LSTM_N-1.txt`
  - `runs/cm_C2_k20_precMaint_2_XGBFS_LSTM+AP_FN_N-2.txt`

### 3. 콘솔 로그
각 실험 종료 시 한 줄 요약 출력:
```
[C1_k20_gateL1off] N-1: P=0.9234 R=0.8567 F1=0.8887 AUC=0.9456
[C2_k20_precMaint] N-2: P=0.9201 R=0.8712 F1=0.8950 AUC=0.9523
[WARN] Precision not maintained!  # C2에서만 발생 가능
```

---

## 실험 순서

코드는 다음 순서로 실험을 수행합니다:

1. **Combo1 루프** (topk=20, 30):
   - C1_k20_gateL1off N-1
   - C1_k20_gateL1off N-2
   - C1_k30_gateL1off N-1
   - C1_k30_gateL1off N-2

2. **Combo2 단일**:
   - C2_k20_precMaint N-1 (P_base 계산)
   - C2_k20_precMaint N-2 (정밀도 유지 체크)

3. **Combo3 루프** (topk=40, 50):
   - C3_k40_varBoost N-1
   - C3_k40_varBoost N-2
   - C3_k50_varBoost N-1
   - C3_k50_varBoost N-2

---

## 예상 실행 시간

- **Combo1**: ~30-40분 (각 topk당 15-20분, 총 2개 topk)
- **Combo2**: ~20-25분 (XGB FS 포함)
- **Combo3**: ~40-50분 (각 topk당 20-25분, 총 2개 topk)

**총 예상 시간**: ~90-115분

---

## 주의사항

1. **데이터 경로**: CICIDS2017 및 UNSW-NB15 데이터셋이 올바른 경로에 있어야 합니다.
   - CICIDS: `CSV/MachineLearningCSV/MachineLearningCVE`
   - UNSW: `CSV_NB15/CSV Files/Training and Testing Sets`

2. **XGBoost 필수**: Combo2 실행을 위해 XGBoost가 설치되어 있어야 합니다.

3. **GPU 권장**: CUDA 지원 GPU 사용 시 실행 시간 대폭 단축.

4. **디스크 공간**: 결과 CSV 및 CM 파일을 위해 충분한 공간 확보.

5. **기존 결과**: `results_path`에 지정한 CSV 파일이 있으면 append 모드로 추가됩니다.

---

## 문제 해결

### XGBoost 미설치
```
[WARN] xgboost not installed; skipping Combo2
```
→ `pip install xgboost` 실행

### CUDA 오류
```
RuntimeError: CUDA out of memory
```
→ `--batch_size` 줄이기 (기본 256 → 128)

### 정밀도 유지 실패 (Combo2)
```
[WARN] C2 precision not maintained: P_proposed=0.9123 < P_base=0.9234
```
→ 정상 작동, `notes`에 `precision_not_maintained` 기록됨

---

## 결과 분석

실험 완료 후 다음 분석 수행:

1. **Combo1**: Gate L1 제거가 topk=20 vs 30에 미치는 영향
2. **Combo2**: 정밀도 유지 여부 및 리콜 향상폭
3. **Combo3**: topk=40 vs 50의 변화폭 (정밀도 유지 없이)

CSV 파일을 pandas로 로드하여 비교:

```python
import pandas as pd

df = pd.read_csv("/mnt/data/results_run.csv")

# Combo별 그룹화
for combo in ["1_AFS_LSTM", "2_XGBFS_LSTM", "3_MI_RF_MLP"]:
    subset = df[df['combo'].str.contains(combo)]
    print(f"\n{combo}:")
    print(subset[['exp_id', 'variant', 'topk_used', 'precision', 'recall', 'f1']])
```

---

## 버전 정보

- **코드 버전**: v3.5 (New Experimental Design)
- **마지막 수정**: 2025-11-23
- **주요 변경**:
  - Combo1: Gate L1 제거, topk 루프
  - Combo2: 동적 정밀도 유지, 강화된 hard negative mining
  - Combo3: 정밀도 유지 비활성, topk 루프
  - 통합 결과 경로 및 CM 저장
