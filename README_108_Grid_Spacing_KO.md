# 108개 유닛 Grid Spacing 분석 (20×20 Mean Gridness)

## 준비

`analyze_108_grid_spacing.py`를 기존 `grid-cells-torch` 프로젝트의 최상위 폴더에 복사하세요. 이 스크립트는 해당 프로젝트의 `config.py`, `grid_cells` 패키지를 사용합니다. 별도 학습은 진행하지 않습니다.

1. 동작 검사:

```powershell
python analyze_108_grid_spacing.py --self-test
```

2. 분석 실행 (기존 checkpoint 30만 회 학습 결과를 그대로 이용):

```powershell
python analyze_108_grid_spacing.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu1m/shards --n-trajectories 4000
```

## 방법

* 한 번의 추론으로 512개 유닛의 **20×20** Ratemap을 계산합니다.
* DeepMind GitHub 방식: 내부 반지름 4 bins, 외부 8~20 bins 균등 간격 10개, **Mean** Gridness를 계산합니다.
* **G > 0.37인 유닛만 선택**합니다. 이전 실험의 108개와 일치해야 하며, 다르다면 checkpoint / trajectory / 파라미터 차이를 경고합니다. 이 0.37은 현재 Mean 방식에 새로 보정된 통계적 임계값은 아닙니다.
* 각 유닛의 39×39 SAC에서 중심 `(19, 19)`에 연결된 **상관계수 > 0.5** 영역을 찾고, 상·하·좌·우 축의 연속 길이 중 최대값을 중심 제외 원의 반지름으로 삼습니다.
* 그 밖에서 **8-이웃 엄격 국소 최댓값**을 찾고 SAC 중심에 가장 가까운 6개를 선택합니다. 선택된 봉우리는 억지로 바꾸거나 평활화하지 않습니다.
* 각 봉우리의 중심과의 거리(bin 및 cm), +x축 기준 극각, 각도순 인접한 여섯 각도 차이를 계산합니다.
* 유닛의 Grid Spacing은 여섯 거리의 **중앙값 × 11 cm/bin**입니다.
* 선택 봉우리 주변의 SAC overlap 픽셀 수가 100 미만인 경우 **경고만 표시**하고, 임의로 제외하지 않습니다. 작은 overlap으로 생기는 가짜 피크가 있을 수 있기 때문입니다.
* 개별 spacing의 분포를 만들고, 1~8개 군집의 가우시안 혼합모형(GMM)을 비교하여 BIC 최적 k를 찾습니다. 별도로 k=3을 강제한 기술적 비교값도 출력하지만 **주장의 근거로 사용하지 않습니다**. 원논문 보고값 약 47, 70, 106 cm 및 인접비 1.49, 1.51은 **비교 대상일 뿐 목표값이 아닙니다**.

## 결과 파일

`results/sonnet_deepmind20_mean/grid_spacing_108/` 폴더에 다음 파일들이 생성됩니다.

- `all_selected_108_one_page.pdf`: 108개의 Ratemap과 SAC(선택한 6개 봉우리 표시)를 한 페이지에 표시
- `all_selected_peak_details.pdf`: 9개 유닛씩 12페이지로 확대, 선택한 봉우리 번호와 중심 제외 원 표시
- `grid_spacing_108_units.csv`: 유닛별 gridness, 제외 반지름, grid spacing, 6개 각도차, 품질 경고
- `six_peaks_per_unit.csv`: 각 유닛의 6개 봉우리 좌표/거리/방향/겹치는 픽셀 수
- `spacing_histogram_gmm.png`: Grid Spacing 히스토그램, BIC가 선택한 군집 중심, 논문 기준선
- `gmm_bic.csv`: 군집 개수 1~8의 BIC와 AIC
- `summary.json`: 최종 분포 및 BIC 최적 군집 수, 3-군집 강제 비교값과 각 비율

파일을 Windows에서 열기 편하게 완료 후 overview PDF가 자동 열립니다. `--no-open`으로 끌 수 있습니다.

## 주의

(1) 원본 20×20 Ratemap의 SAC는 가장자리에서 겹침 픽셀 수가 적어 큰 가짜 봉우리가 생길 수 있습니다. 특히 212번 유닛의 209cm 추정처럼 길이가 커지는 값은 반드시 SAC와 함께 검토하세요.
(2) 이 분석은 요청한 탐색적 절차이며 Banino 논문의 원본 Grid Spacing 검출법을 완전히 재현하는 것으로 주장하지 않습니다.
(3) 이미 저장한 `unit_statistics.csv`가 있으면 이번 Gridness 결과와 대조하지만, Shuffling은 실행하지 않습니다.
