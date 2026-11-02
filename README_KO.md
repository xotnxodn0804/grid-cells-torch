# Sonnet 1.27 LSTM 초기화 비교 실험

전문가가 만든 `soluslan/grid-cells-torch` **기존 파일을 수정하지 않는** 별도 실행용 파일입니다.

## 설치

이 ZIP을 압축 해제한 뒤 **네 개의 `.py` 파일을 전부** `grid-cells-torch` 저장소의 최상위 폴더(`train.py`, `config.py`, `analyze_supervised.py`가 있는 곳)에 놓으세요. 기존 `grid_cells/model.py`, `grid_cells/training.py` 등은 그대로 유지합니다.

- `sonnet_init_override.py`: 기존 모델 생성 시 **LSTM 내부만** Sonnet 1.27 방식으로 초기화하도록 변경
- `train_sonnet_init.py`: 기존 `train.py`의 학습 설정을 그대로 사용하되 새 checkpoint 폴더에 저장
- `analyze_sonnet_init.py`: 기존 `analyze_supervised.py`를 그대로 사용해 새 checkpoint를 분석
- `test_sonnet_init.py`: 초기화 방식 간단 검증

## 실행

```bash
python test_sonnet_init.py
python train_sonnet_init.py --shard-dir data/datasets/gpu1m/shards
python analyze_sonnet_init.py --checkpoint data/checkpoints/sonnet_v127_lstm_512/checkpoint_epoch299.pt --analysis gridness
```

데이터가 `data/latest/shards`에 있으면 `--shard-dir`를 생략해도 됩니다. 다른 데이터 경로라면 실제 경로로 변경하세요.

처음엔 소규모 점검 가능:

```bash
python train_sonnet_init.py --epochs 1 --steps-per-epoch 2 --results-dir data/checkpoints/sonnet_smoke
```

**주의**: 동일한 결과 폴더에 재실행하면 덮어쓸 위험이 있어, 코드가 기존 checkpoint가 있는 폴더를 거부합니다. 재실행할 때는 다른 `--results-dir`를 주세요.

## 원본과의 정합성

Sonnet 1.27 `LSTM._create_gate_variables`: 입력 3 + hidden 128 = 131 기준 truncated normal을 **가중치 및 gate bias**에 적용하고, forget bias=1을 추가합니다. PyTorch LSTM은 gate 순서와 bias 구조가 달라 이를 반영했습니다.

다른 계층(이미 절단정규분포), Gaussian sigma=0.01, von Mises kappa=20, `neurons_seed=0`, 데이터 생성, optimizer, dropout은 **변경하지 않았습니다.** 이 비교 실험은 LSTM 초기화의 영향만 분리해서 확인합니다.

이렇게 만들어도 DeepMind의 원본 데이터셋/연산의 모든 부분이 동일해지는 것은 아닙니다. 기존 코드를 기반으로 한 *LSTM 초기화 분포 재현 실험*입니다.

원본 Sonnet 코드: https://github.com/google-deepmind/sonnet/blob/v1.27/sonnet/python/modules/gated_rnn.py
