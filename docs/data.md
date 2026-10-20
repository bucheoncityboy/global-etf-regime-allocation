# 데이터 명세

원본은 [input_snapshot.zip](../data/input_snapshot.zip), 출처·개별 응답 해시는 [manifest.json](../data/manifest.json)에 있다. ZIP 해시는 `816dfd73aad49c59c75dc2baf4c4328794a0b74345b9ba34c0082b55251b9df1`이다. 이전 연구의 성과 산출물은 사용하지 않았고, 새로 수집한 원본 응답만 재사용한다.

| 자산군 | 이번 ETF | 원문 부록 ETF | 차이 |
| --- | --- | --- | --- |
| 미국 대형주 | SPY | IVV | 다른 ETF |
| 미국 중형주 | IJH | IJH | 동일 티커 대용 |
| 미국 소형주 | IWM | IWM | 동일 티커 대용 |
| 미국 외 선진국 주식 | EFA | EFA | 동일 티커 대용 |
| 신흥국 주식 | EEM | EEM | 동일 티커 대용 |
| 미국 종합채권 | AGG | AGG | 동일 티커 대용 |
| 장기 미국 국채 | TLT | SPTL | 만기·지수 구성 차이 |
| 하이일드 회사채 | HYG | HYG | 동일 티커 대용 |
| 투자등급 회사채 | LQD | SPBO | 지수·구성 차이 |
| 미국 리츠 | IYR | IYR | 동일 티커 대용 |
| 원자재 | DBC | DBC | ETF 총수익과 원문 선물 excess-return 지수 차이 |
| 금 | GLD | GLD | ETF 비용과 원문 금 가격 지수 차이 |
| 현금 | BIL | 3개월 Treasury 금리 | 투자 가능한 현금 대용과 금리 시계열의 차이 |

동일 티커라도 원문이 사용한 것은 ETF가 추종하는 지수다. 추적오차·보수·담보수익·배당 처리 차이가 남는다. [원문 자산 표·부록](https://arxiv.org/html/2406.09578v2#A1)

가격은 Yahoo Finance chart API의 `interval=1d`, `events=div,splits`, `includeAdjustedClose=true` 요청이다. 4년 이하 청크로 받았다. 중복 날짜·누락 거래일·빈 값·0 이하 가격·일별 빈도가 아닌 응답은 실패시킨다. 거래 달력은 pandas_market_calendars NYSE 5.5.0의 보존된 세션 목록이며 2025-09-30 이후는 사용하지 않는다.

일별 `adj_close/close` 비율을 시가에도 적용한다. 공급자 가격에 이미 반영된 split을 다시 적용하지 않는다. 배당조정 종가 수익률과 `(종가+분배금)/전일종가−1`을 대조해 최대 오차30bp 미만을 요구한다. 이는 공급자의 반올림과 수정자료를 감안한 허용치로, 분배금 사건 하나하나의 법적·회계적 정확성을 보증하지 않는다. 실제 최대 오차는 manifest의 자산별 필드에 기록한다.

시장 입력은 다음 [FRED](https://fred.stlouisfed.org/) 시리즈다.

- [DGS2](https://fred.stlouisfed.org/series/DGS2): 2년 국채 금리.
- [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y): 10년−2년 금리차.
- [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS): VIX 종가.
- [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO): 수익 피처·조건부 평균용 일별 무위험 금리 대용.

날짜만 있는 관측치는 신호일보다 엄격히 이전 날짜의 값만 사용한다. 결측 표시는 버리고 최신 유효값의 경과일이10일을 넘으면 실패시킨다. 배포 지연을 보수적으로 처리했지만 당시 빈티지를 복원한 데이터베이스는 아니다. ETF 조정가격도 2026-10-20 조회 시점의 수정 자료다.

12개 자산 모두의 피처가 갖춰지는 가장 이른 시점은 2007-07-11이다. 5년 학습의 첫 반기 모델은2012년말 적합, 순차 검증은2013년부터, 평가는2016년부터다. 짧은 ETF 이력을 지수나 다른 펀드의 수익률로 소급 연결하지 않는다. 상장폐지 ETF를 포함한 사전 정의 모집단이 아니라 원문 자산군을 현재 ETF로 대응한 구성이라 생존 편향 가능성도 남는다.

ZIP에는 대용 자산 선택 과정에서 받은 미사용 티커도 포함된다. 실제 사용 티커는 protocol의12개+BIL뿐이며 다른 티커와 Yahoo 응답 메타데이터의 현재 시점 필드는 모델 입력으로 사용하지 않는다. 원시 자료의 권리는 각 공급자에 귀속된다.
