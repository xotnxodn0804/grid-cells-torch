# 20×20 Gridness 평가 — DeepMind 공개 코드의 10개 Annulus

**기존 학습 모델/체크포인트와 전문가 코드를 수정하지 않습니다.** `checkpoint_epoch299.pt`에서 훈련된 512개 Bottleneck 유닛을 평가합니다.

## 정확한 매개변수

- Ratemap: **20 × 20**
- Spatial autocorrelogram: **39 × 39**
- Annulus: 내부 **0.2×20 = 4 bins**, 외부 **np.linspace(0.4, 1.0, 10) × 20**
- 외부 반지름: **8, 9.3333, 10.6667, 12, 13.3333, 14.6667, 16, 17.3333, 18.6667, 20** bins
- 10개의 영역 각각에 대해 회전 상관값으로 Gridness를 구하고 최댓값을 선택
- `--formula mean` (기본): DeepMind **공개 소스**의 `GridScorer(..., min_max=False)` 방식. `(r60+r120)/2 - (r30+r90+r150)/3`.
- `--formula minmax`: **논문 Methods**에 서술된 방식. `min(r60,r120)-max(r30,r90,r150)`.

**주의:** 원논문 Methods의 확장 반지름(8→20, 2씩 증가, 7개)과 공개 코드의 10개 링은 서로 다릅니다. 또한 0.37이라는 역치가 여기의 `mean` 방식으로 새로 검증된 것은 아닙니다. 본 분석은 100회 Field Shuffling 결과에서 유닛별 95번째 백분위수를 별도로 도출합니다.

## 설치

ZIP에서 다음 3개 파일을 압축 해제한 후, 기존 `grid-cells-torch`의 `train.py`, `config.py`, `grid_cells/`와 같은 위치로 복사합니다.

- `run_field_shuffle_deepmind20.py` — 새 분석 실행 파일
- `field_shuffle_nosmoothing.py` — 활성 영역 재배치 기능 (전에 설치했다면 동일 파일)
- `compare_field_smoothing.py` — Watershed 활성 영역 분할 (전에 설치했다면 동일 파일)

원래 프로젝트의 `grid_cells` 파일은 수정하지 않습니다.

## 검증

```powershell
python run_field_shuffle_deepmind20.py --self-test
```

## 4개 유닛 파일럿 (통계적 판정 아님)

```powershell
python run_field_shuffle_deepmind20.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu1m/shards --unit-ids 212,353,137,191 --n-shuffles 10 --workers 1 --formula mean --out-dir results/sonnet_deepmind20_pilot
```

## 전체 512개 유닛 / 유닛당 100회 Field Shuffling

```powershell
python run_field_shuffle_deepmind20.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu1m/shards --n-trajectories 4000 --n-shuffles 100 --workers 4 --formula mean --out-dir results/sonnet_deepmind20_mean
```

논문 Min–Max 수식과 비교하고 싶으면 `--formula minmax` 및 별도 `--out-dir`를 사용하세요. 후자의 통계적 임계값은 **그 수식의 100회 Shuffling으로 따로 산출**해야 합니다.

```powershell
python run_field_shuffle_deepmind20.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --shard-dir data/datasets/gpu1m/shards --n-trajectories 4000 --n-shuffles 100 --workers 4 --formula minmax --out-dir results/sonnet_deepmind20_minmax
```

## 생성 결과

- `unit_statistics.csv`: 유닛별 원점수, Shuffle Null 95번째 백분위수, 경험적 p값, Field 수
- `summary.json`: 평균 Gridness 및 유닛별 Shuffle 판정 수
- `null_distributions.npz`: 100회 shuffled score 전부
- `gridness_null95.png`: 분포 그림
- `cache/`: 중단 후 동일 조건 재시작을 위한 결과 캐시

## 연구상 주의사항

Field Shuffling의 Watershed 세부 영역 식별/배치 규칙은 Banino 원논문에 완전히 기술되지 않았습니다. 본 코드는 이를 근사한 것으로, 원저자의 정확한 구현과 동등하다고 주장할 수 없습니다. Watershed 식별용 지도만 σ=0.9 bin으로 평활화하고 **원래 활성값과 Gridness 계산에는 평활화를 적용하지 않습니다**. 이것도 검증 가능한 실험 설정이지, 논문에서 검증된 σ 값은 아닙니다.

20×20에서 0.9 bins의 물리적 공간 범위는 32×32에서 0.9 bins와 달라집니다. Field Shuffling 결과를 평가할 때 민감도 분석이 필요합니다.

0.37(논문 기준)은 현 실험에서 **참조용으로만** 보고하고, 20×20의 평균/Min–Max 각 방식에 대해서는 Shuffle 95번째 백분위수 판정을 우선 활용하세요.
