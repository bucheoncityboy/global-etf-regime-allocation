# 재현과 산출물

Windows PowerShell에서 검증했다. Node.js22 이상, uv, Python3.12가 필요하며 Python은 uv가 설치할 수 있다. 모델·수치 라이브러리는 uv.lock, TypeScript 도구는 package-lock.json으로 고정한다.

```powershell
npm ci
uv sync --frozen
npm run typecheck
npx tsx src/harness.ts
npm run reproduce
```

전체 실행은 준비→학습→후보선택·배분→표·그래프→무결성·거래 재생 순서다. 네트워크는 의존성 설치에만 필요하고, 가격은 보존된 ZIP에서 복원한다. 라이브러리·OS·하드웨어에 따른 최적화 수치 차이가 있을 수 있으며, 별도 플랫폼의 전체 재학습 결과가 바이트 단위로 같다고 보장하지 않는다.

## 실행 단계

| 명령 | 작업 |
| --- | --- |
| `npm run study -- prepare` | 입력 ZIP·응답 해시 검증, 가격·피처 생성, 작업 목록·설계 해시 기록 |
| `npm run study -- fit` | 자산·반기별 JM과 LR·XGB 적합, 재개 가능한 캐시 작성 |
| `npm run study -- allocate` | 순차 검증으로 후보 선택 후 일별 계정 실행 |
| `npm run study -- report` | 표·부트스트랩·그림·README 생성 |
| `npm run verify` | 기록된 소스·산출물 해시 검사, 전체 거래 독립 재생 |

`prepare`는 새로운 실행의 설계 지문을 기록한다. 의도적으로 구현을 수정할 때는 소스를 고친 뒤 prepare부터 다시 실행한다. 기존 발표 결과의 소스 불일치를 숨기기 위해 prepare만 실행해서는 안 된다. 기존 결과 검증을 위한 입력 복원은 아래 명령으로 수행하며, 이 명령은 설계 지문을 덮어쓰지 않는다.

설계 해시는 산출물을 재계산할 코드의 지문이며 통계적 사전등록의 증명은 아니다.

```powershell
npm run study -- restore
npm run verify
```

학습 캐시는 kernel·protocol·uv.lock·입력ZIP 해시로 분리해 다른 모델 설정을 재사용하지 않는다. 캐시를 삭제해도 원본과 코드로 전체 재계산할 수 있다. `STUDY_WORKERS=4`가 기본이며 병렬 프로세스마다 수치 라이브러리 스레드는1개다. XGB의 용량을 OOS 결과를 보고 추가 튜닝하지 않는다.

TypeScript는 프로세스·입출력·작업 분배·해시·검증 실행을 관리한다. 수치 라이브러리 어댑터는 `.ts` 모듈의 문자열로 보존되고 길이가 지정된 UTF-8 표준입력으로 Python에 전달된다. 셸 코드 보간이나 생성된 Python 스크립트 파일, 이전 저장소 import를 사용하지 않는다. 이 구조는 상위 작업공간의 새 스크립트 TypeScript 규칙과 Python 과학 라이브러리 의존성을 함께 충족하기 위한 선택이다.

## 산출물

| 파일 | 내용 |
| --- | --- |
| `data/manifest.json` | 개별 원본 URL·조회시점·해시·가격 검증 |
| `results/design-freeze.json` | 실행 소스·설계·환경 잠금 해시 |
| `results/jobs.json` | 744개 자산·반기 작업 |
| `results/fits.json` | 3,720개 JM·분류기 학습 경계·라벨 해시·평균·상태수 |
| `results/tuning.json` | 매 반기 벌점·평활·검증기간·후보별 검증 점수 |
| `results/predictions.parquet` | 선택된 확률·신호/목표/학습일·국면 평균 |
| `results/targets.parquet` | 각 계정의 일별13자산 목표 비중 |
| `results/trades.parquet` | 시가에 실제 거래한13자산 금액 |
| `results/nav.parquet` | 일별 NAV·전일/시가/거래후 NAV·비용·밤사이/장중 손익 |
| `results/summary.csv` | 전체·공통구간 성과·비용·노출 |
| `results/individual.csv` | 개별ETF 0/1 전략 성과 |
| `results/paired_intervals.csv` | XGB−LR의 paired 블록 부트스트랩 구간 |
| `results/forecast_quality.csv` | 온라인 JM 라벨 대비 보조 예측 지표 |
| `results/return_forecast_ic.csv` | 자산별 예측 평균과 다음 수익률의 상관; 결과 확인 후 추가한 설명용 진단 |
| `results/environment.json` | 실제 설치 버전 |
| `results/verification.json` | 독립 재생·경계·공유 조건 검증 |
| `results/artifact-manifest.json` | 발표 산출물 SHA-256 |

CI는 타입 검사·수치 경계 테스트·입력 복원·발표 결과 재생을 실행한다. 비용이 큰 전체 학습은 CI에 포함하지 않는다. 새 결과를 발표할 때 전체 학습과 배분, 보고서 생성을 수행해야 한다.
