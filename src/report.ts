export const reporting = String.raw`
def paired_intervals(a,b,cash,block,seed):
    rng=np.random.default_rng(seed);n=len(a);draws=[];gamma=C['bootstrap']['risk_aversion']
    for _ in range(C['bootstrap']['replications']):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        aa=a[idx];bb=b[idx];rf=cash[idx]
        cer=252*((aa.mean()-.5*gamma*aa.var(ddof=1))-(bb.mean()-.5*gamma*bb.var(ddof=1)))
        draws.append([cer,sharpe(aa,rf)-sharpe(bb,rf)])
    low,high=np.quantile(draws,[.025,.975],axis=0)
    point=[252*((a.mean()-.5*gamma*a.var(ddof=1))-(b.mean()-.5*gamma*b.var(ddof=1))),sharpe(a,cash)-sharpe(b,cash)]
    return {'cer_delta':point[0],'cer_low':low[0],'cer_high':high[0],
        'sharpe_delta':point[1],'sharpe_low':low[1],'sharpe_high':high[1]}
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def pct(x):return f'{x*100:.2f}%'
def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=load_data();nav=pd.read_parquet(ROOT/'results/nav.parquet');s=pd.read_csv(ROOT/'results/summary.csv')
    p=pd.read_parquet(ROOT/'results/predictions.parquet');ind=pd.read_csv(ROOT/'results/individual.csv')
    intervals=[]
    for profile in C['profiles']:
        pid=profile['id']
        for period,start in [('full',profile['start']),('common','2024-01-02')]:
            frame=nav.loc[(nav.profile==pid)&(nav.study=='controlled')&(nav.cost_bp==5)&(nav.date>=start)]
            a=frame.loc[frame.strategy=='mv_xgb'].set_index('date').net_return
            b=frame.loc[frame.strategy=='mv_lr'].set_index('date').net_return.reindex(a.index)
            cash=d['returns'][C['cash']].reindex(a.index).to_numpy()
            for block in [21,63]:
                intervals.append({'profile':pid,'period':period,'block_days':block,
                    **paired_intervals(a.to_numpy(),b.to_numpy(),cash,block,C['seed'])})
    ci=pd.DataFrame(intervals);ci.to_csv(ROOT/'results/paired_intervals.csv',index=False)
    quality=[]
    for key,frame in p.groupby(['profile','study','classifier']):
        y=(frame.truth.to_numpy()==0).astype(float);prob=frame.p0.to_numpy().clip(1e-8,1-1e-8)
        hit=(prob>=.5)==y;rates=[hit[y==v].mean() for v in [0,1] if (y==v).any()]
        quality.append({'profile':key[0],'study':key[1],'classifier':key[2],
            'accuracy':float(hit.mean()),'balanced_accuracy':float(np.mean(rates)),
            'brier':float(np.mean((prob-y)**2)),'log_loss':float(-np.mean(y*np.log(prob)+(1-y)*np.log(1-prob))),
            'bullish_fraction':float((prob>=.5).mean()),'rows':len(frame)})
    pd.DataFrame(quality).to_csv(ROOT/'results/forecast_quality.csv',index=False)
    ic=[];unconditional=d['excess'].ewm(halflife=C['baseline_mean_halflife'],adjust=False,min_periods=252).mean()
    for profile in C['profiles']:
        pid=profile['id'];dates=d['cal'][(d['cal']>=profile['start'])&(d['cal']<=C['end'])]
        signals=d['cal'][d['cal'].get_indexer(dates)-1]
        for a in ASSETS:
            actual=d['excess'][a].loc[dates].to_numpy()
            intraday=(d['close'][a]/d['open'][a]-d['close'][C['cash']]/d['open'][C['cash']]).loc[dates].to_numpy()
            for name in ['ewma','lr','xgb']:
                if name=='ewma':forecast=unconditional[a].loc[signals].to_numpy()
                else:
                    f=p.loc[(p.profile==pid)&(p.study=='controlled')&(p.classifier==name)&(p.asset==a)].set_index('date').loc[dates]
                    forecast=np.where(f.p0>=.5,f.mu0,np.minimum(f.mu1,C['bearish_mean_cap']))
                ic.append({'profile':pid,'asset':a,'forecast':name,
                    'ic_close_close':float(np.corrcoef(forecast,actual)[0,1]),
                    'ic_open_close_over_cash':float(np.corrcoef(forecast,intraday)[0,1])})
    pd.DataFrame(ic).to_csv(ROOT/'results/return_forecast_ic.csv',index=False)
    figdir=ROOT/'results/figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'figure.dpi':140,'axes.grid':True,'grid.alpha':.2,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ['paper','controlled']:
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        if kind=='paper':lines=[('baseline','mv','MV'),('paper','mv_xgb','MV + JM-XGB'),('baseline','minvar','MinVar'),
            ('paper','minvar_xgb','MinVar + JM-XGB'),('baseline','ew','Equal weight'),('paper','ew_xgb','EW + JM-XGB')]
        else:lines=[('controlled','mv_prior','Prior'),('controlled','mv_jm','JM persistence'),('controlled','mv_lr','JM-LR'),
            ('controlled','mv_xgb','JM-XGB'),('baseline','60_40','SPY/AGG 60/40')]
        for study,name,label in lines:
            f=nav.loc[(nav.profile=='main')&(nav.study==study)&(nav.strategy==name)&(nav.cost_bp==5)].set_index('date')
            wealth=f.nav;peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
            axes[0].plot(f.index,wealth,label=label,lw=1.5);axes[1].plot(f.index,wealth/peak-1,lw=1.1)
        axes[0].set_ylabel('Wealth (initial 1)');axes[1].set_ylabel('Drawdown');axes[0].legend(ncol=3,fontsize=8)
        fig.suptitle(f'12 ETF proxies | {kind} comparison | next-open execution, one-way 5 bp')
        fig.tight_layout();fig.savefig(figdir/f'{kind}.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for axis,a in zip(axes,['SPY','AGG','GLD']):
        cl=d['close'][a].loc['2016-01-04':];axis.plot(cl.index,cl/cl.iloc[0],color='#555555',label=a)
        axis2=axis.twinx()
        for name,color in [('lr','#cf8d12'),('xgb','#0077b6')]:
            f=p.loc[(p.profile=='main')&(p.study=='controlled')&(p.classifier==name)&(p.asset==a)].set_index('date')
            axis2.plot(f.index,f.p0,color=color,alpha=.55,lw=.7,label=name.upper())
        axis2.set_ylim(0,1);axis2.set_ylabel('Bull probability');axis.legend(loc='upper left');axis2.legend(loc='upper right')
    fig.suptitle('Asset-specific regime forecasts | shared labels and selection');fig.tight_layout();fig.savefig(figdir/'regimes.png');plt.close(fig)
    full=s.loc[(s.profile=='main')&(s.period=='full')&(s.cost_bp==5)]
    def row(study,strategy):return full.loc[(full.study==study)&(full.strategy==strategy)].iloc[0]
    paper_rows=[]
    for name,label,original in [('mv','평균분산','0.37 → 1.02'),('minvar','최소분산','0.70 → 1.12'),('ew','동일가중','0.51 → 0.91')]:
        before=row('baseline',name);after=row('paper',name+'_xgb')
        paper_rows.append([label,original,f'{before.sharpe:.3f} → {after.sharpe:.3f}',f'{pct(before.mdd)} → {pct(after.mdd)}'])
    controlled_rows=[]
    for name,label in [('prior','빈도 기준'),('jm','JM 유지'),('lr','JM–로지스틱'),('xgb','JM–XGBoost')]:
        r=row('controlled','mv_'+name)
        controlled_rows.append([label,pct(r.cagr),f'{r.sharpe:.3f}',pct(r.mdd),pct(r.cer),pct(r.annual_cost),pct(r.risk_exposure)])
    lr=row('controlled','mv_lr');xgb=row('controlled','mv_xgb');primary=ci.loc[(ci.profile=='main')&(ci.period=='full')&(ci.block_days==21)].iloc[0]
    higher=int(sum(row('paper',n+'_xgb').sharpe>row('baseline',n).sharpe for n in ['mv','minvar','ew']))
    ci_text=f"연 CER 차이(XGB−LR) {primary.cer_delta*100:+.2f}%p, 95% 구간 [{primary.cer_low*100:+.2f}, {primary.cer_high*100:+.2f}]%p"
    interpretation=('이 조건에서 XGBoost의 추가 가치가 관측됐다.' if primary.cer_low>0 else
        '이 조건에서 로지스틱의 우위가 관측됐다.' if primary.cer_high<0 else '구간이 0을 포함하므로 XGBoost의 우위를 확인하지 못했다.')
    sensitivity=[]
    for profile in C['profiles']:
        for period,label in [('full','전체'),('common','공통 2024.01–2025.09')]:
            f=s.loc[(s.profile==profile['id'])&(s.study=='controlled')&(s.period==period)&(s.cost_bp==5)]
            l=f.loc[f.strategy=='mv_lr'].iloc[0];x=f.loc[f.strategy=='mv_xgb'].iloc[0]
            sensitivity.append([f"{profile['train_years']}년/{profile['validation_years']}년",label,str(l.days),f'{l.sharpe:.3f}',f'{x.sharpe:.3f}',f'{(x.cer-l.cer)*100:+.2f}%p'])
    indiv=[]
    for a in ASSETS:
        f=ind.loc[(ind.profile=='main')&(ind.asset==a)].set_index('strategy')
        indiv.append([a,*[f"{f.loc[n,'sharpe']:.3f}" for n in ['buy_hold','jm','xgb']],pct(f.loc['xgb','mdd'])])
    costs=[]
    for cost in [0,5,10,25]:
        f=s.loc[(s.profile=='main')&(s.study=='controlled')&(s.period=='full')&(s.cost_bp==cost)]
        l=f.loc[f.strategy=='mv_lr'].iloc[0];x=f.loc[f.strategy=='mv_xgb'].iloc[0]
        costs.append([str(cost),f'{l.sharpe:.3f}',f'{x.sharpe:.3f}',f'{(x.cer-l.cer)*100:+.2f}%p'])
    q=pd.DataFrame(quality).query("profile == 'main' and study == 'controlled'").set_index('classifier')
    ic_main=pd.DataFrame(ic).query("profile == 'main' and forecast == 'xgb'")
    positive_ic=int((ic_main.ic_close_close>0).sum())
    mean_ic=float(ic_main.ic_close_close.mean())
    bt=chr(96)
    readme=f'''# 국면 예측을 활용한 글로벌 ETF 자산배분

**12개 자산군으로 다시 구성한 Jump Model–XGBoost ETF 응용 연구.**

[![검증](https://github.com/bucheoncityboy/global-etf-regime-allocation/actions/workflows/ci.yml/badge.svg)](https://github.com/bucheoncityboy/global-etf-regime-allocation/actions/workflows/ci.yml)

{bt}실제 ETF 12개 + BIL{bt} · {bt}일별 판단·다음 시가 체결{bt} · {bt}편도 5bp{bt} · {bt}117개월 주 실험{bt} · {bt}TypeScript + Python 수치 라이브러리{bt}

Shu·Yu·Mulvey의 자산별 국면 예측을 공개 ETF에 적용했다. **논문형 배분의 개선**과 **동일 조건에서 XGBoost의 추가 가치**를 분리해 평가한다. 기존 저장소의 결과를 이식하지 않고 새 입력·프로토콜·계정으로 계산했다.

> **연구 범위** — 원문의 1991–2023년 12개 지수가 아닌 ETF 대용 자료다. 주 실험은 5년 학습·3년 검증이며 원문의 11년·5년과 다르다. 과거 결과를 확인한 뒤 설계한 후속 연구로, 전향적으로 확보한 미사용 표본은 아니다.

[핵심 결과](#핵심-결과) · [논문과의 비교](#논문과의-비교) · [국면과 ML 비교](#국면과-ml-비교) · [실험 설계](#실험-설계) · [데이터 명세](#데이터-명세) · [검증 방법론](#검증-방법론) · [결론](#결론) · [재현](#재현)

## 핵심 결과

- **전체 배분:** 논문형 JM–XGBoost의 샤프가 기본 배분보다 높은 경우는 세 배분 중 **{higher}개**다.
- **분류기 비교:** 같은 국면·입력·평균 추정·배분·비용에서 로지스틱 샤프 **{lr.sharpe:.3f}**, XGBoost **{xgb.sharpe:.3f}**다.
- **불확실성:** {ci_text}. {interpretation}

## 논문과의 비교

원문은 2007–2023년, 아래 ETF 적용은 **2016.01.04–2025.09.30**이다. 원문과의 수치 차이는 자료·기간·학습창이 함께 달라 원인별 인과효과로 해석할 수 없다.

{table(['배분','원문 샤프 기본 → 국면','ETF 샤프 기본 → 국면','ETF 최대낙폭 기본 → 국면'],paper_rows)}

이 비교는 예측·조건부 평균·위험회피·거래벌점·현금 규칙을 함께 바꾼다. XGBoost 단독 효과는 다음 절에서 비교한다. [원문 §4–5·표6](https://arxiv.org/html/2406.09578v2#S5.SS2) · [구현 대응표](docs/paper-comparison.md)

![논문형 배분의 NAV와 낙폭](results/figures/paper.png)

<details>
<summary>개별 ETF 0/1 전략 · 비용 차감 성과</summary>

{table(['ETF','B&H 샤프','JM 유지 샤프','JM–XGB 샤프','JM–XGB 최대낙폭'],indiv)}

JM 유지는 XGBoost가 선택한 벌점을 공유하고, 당일 온라인 국면을 다음 날로 유지한다. XGB만 추가 시장 피처와 확률 평활을 사용하므로 공정한 분류기 비교는 아니다. ETF 수를 독립 통계 표본 수로 취급하지 않는다. [전체 CSV](results/individual.csv)

</details>

## 국면과 ML 비교

두 국면은 자산마다 따로 추정한다. 아래 네 모델은 **동일한 JM 라벨, 벌점 선택 경로, 평활 반감기 4일, 조건부 평균, 공분산, 제약·비용**을 공유한다. 빈도 기준과 JM 유지는 분류기를 학습하지 않으며, LR·XGB는 같은 13개 입력을 받는다. 벌점은 직전 검증창의 LR·XGB 순초과 샤프 평균으로 선택했다.

{table(['모델','CAGR','샤프','최대낙폭','연 CER(γ=5)','연 비용','평균 위험비중'],controlled_rows)}

{ci_text}. {interpretation} 같은 날짜를 함께 뽑는 21일 원형 블록 부트스트랩 2,000회 결과다. 63일 블록·다른 창의 결과도 공개한다. 전략 선택과 모델 학습을 다시 수행한 불확실성은 포함하지 않는다.

![동일 조건의 국면 예측 비교](results/figures/controlled.png)

![SPY·AGG·GLD 국면 확률](results/figures/regimes.png)

### 학습창과 평가기간

{table(['학습/검증','평가 구간','거래일','LR 샤프','XGB 샤프','XGB−LR 연 CER'],sensitivity)}

전체 구간 비교에는 시장 시기 차이가 섞인다. 공통 21개월 비교는 동일한 시장 구간에서 학습·검증창을 함께 바꾼 민감도다. 원문 창의 표본이 짧아 장기 성과를 대변하지 못한다.

### 거래비용 민감도

{table(['편도 bp','LR 샤프','XGB 샤프','XGB−LR 연 CER'],costs)}

벌점과 평활은 5bp에서 선택한 경로로 고정하고, 비용별 거래벌점·실제 체결비용을 함께 바꿔 재최적화했다.

[전체 성과](results/summary.csv) · [신뢰구간](results/paired_intervals.csv) · [온라인 JM 라벨 대비 예측 지표](results/forecast_quality.csv) · [기대수익 예측 IC](results/return_forecast_ic.csv)

### 결과가 달라지는 지점

주 실험에서 XGBoost의 온라인 JM 라벨 대비 균형정확도는 **{pct(q.loc['xgb','balanced_accuracy'])}**, 로지스틱은 **{pct(q.loc['lr','balanced_accuracy'])}**다. 이 라벨은 경제 상태의 정답이 아니며 높은 분류 정확도가 수익 예측력으로 이어진다고 볼 수 없다.

통제 경로의 XGB 국면 평균 예측과 다음 날 종가 기준 초과수익의 상관이 양수인 자산은 **{positive_ic}/12개**, 자산별 상관의 단순 평균은 **{pct(mean_ic)}**다. XGB의 양방향 연 회전율은 **{xgb.annual_turnover:.2f}배**, 연 비용은 **{pct(xgb.annual_cost)}**다. 이 표본에서는 기대수익 예측의 약함과 잦은 거래가 함께 관측된다. 자료·학습창·기간이 함께 달라 원문과의 차이를 하나의 원인으로 확정하지 않는다. 이 진단은 결과를 본 뒤 추가한 설명용 분석이다.

## 실험 설계

| 항목 | 고정 규칙 |
| --- | --- |
| 주 실험 | 5년 학습 / 3년 순차 검증 / 2016.01–2025.09, 117개월 |
| 추가 창 | 7년/3년, 93개월 · 11년/5년, 21개월 |
| 재학습 | 매년 1·7월 첫 거래일 직전 종가, 6개월 유지 |
| 라벨 | 자산별 2상태 JM, 누적 초과수익 높은 상태가 강세 |
| 예측 | 당일 피처 → 다음 거래일 라벨; 마지막 학습 목표일 ≤ 재학습일 |
| 벌점 후보 | 0, 0.1, 1, 10, 100; JM 초기화 10회 |
| 논문형 선택 | XGB 0/1 순초과 샤프 최대; 첫 검증창에서 평활 0/2/4/8일 선택 후 고정 |
| 통제 비교 선택 | LR·XGB 검증 샤프 평균 최대; 평활 4일 고정 |
| 배분 | 롱온리·차입 없음·개별 위험자산 ≤40%; 강세 자산 ≤3개면 전액 BIL |
| 비교 모델 | 빈도 기준·JM 유지·로지스틱·XGBoost; 추가 BIL·SPY/AGG 60/40 |

[상세 수식·선택 및 실행 시점](docs/methodology.md) · [설계 파일](configs/protocol.json)

## 데이터 명세

SPY·IJH·IWM·EFA·EEM·AGG·TLT·HYG·LQD·IYR·DBC·GLD와 현금 BIL을 사용한다. 장기국채 TLT와 회사채 LQD는 원문 부록 ETF와 다른 대용 자산이다. 지수와 ETF의 만기 구성·비용·담보수익 차이가 남는다.

성과는 USD 기준이다. 가격은 Yahoo 일별 배당조정 종가와 동일 조정계수의 시가, 시장 피처는 FRED DGS2·T10Y2Y·VIXCLS·DGS3MO다. 날짜만 있는 FRED 값은 관측일보다 늦은 거래일에 사용한다. 시장 상관은 **SPY–AGG 252거래일**이다. 미래 빈티지가 완전히 보존된 실시간 데이터는 아니다.

2026-10-20에 받은 입력 중 **2025-09-30까지**만 사용했다. 원본 ZIP·개별 응답 URL·SHA-256을 보관해 재다운로드 없이 재계산할 수 있다. [자산 대응·출처·가격 검증](docs/data.md) · [입력 명세](data/manifest.json)

## 검증 방법론

모든 학습과 스케일링은 재학습일 이전에 끝난다. 검증도 이전 모델의 순차 예측만 사용한다. 예측일 다음 시가에 체결하며 기존 보유분의 밤사이 수익은 기존 자산에 귀속한다. 매수·매도와 BIL 거래를 모두 포함해 자기금융 조건으로 비용을 정산한다.

단위 검증은 비용·체결 시점·평활 인과성·공분산·12자산 제약을 확인한다. 산출물 검증은 모든 계정의 목표 비중을 별도 수량 기반 엔진으로 재생해 NAV·거래량·비용을 대조한다. 학습 목표일, 온라인 JM의 미래 경로 독립성, 모델 간 공유 조건, 원본·소스·출력 해시도 검사한다. [검증 결과](results/verification.json)

## 결론

12개 자산군과 실효성 있는 학습창으로 기존의 짧은 표본·자산 구성 문제를 줄였다. **{interpretation}** 논문의 전체 시스템 개선은 XGBoost가 다른 분류기보다 항상 우수하다는 증거와 다르다. 다음 검증은 이번 결과를 보지 않은 미래 구간 또는 원문 지수 데이터에서 수행해야 한다.

## 재현

Node.js 22 이상과 uv가 필요하다. TypeScript가 실행·프로토콜·작업 분배·해시를 관리하고 잠긴 Python 3.12 수치 라이브러리가 JM·LR·XGB·최적화·그래프를 계산한다.

{bt*3}powershell
npm ci
uv sync --frozen
npm run typecheck
npx tsx src/harness.ts
npm run reproduce
{bt*3}

저장된 결과의 무결성과 거래 재생만 확인하려면 {bt}npm run study -- restore{bt} 뒤 {bt}npm run verify{bt}를 실행한다. 전체 재학습은 744개 자산·반기 작업, 3,720개 JM과 최대 7,440개 분류기를 계산한다. {bt}STUDY_WORKERS{bt} 환경변수로 병렬 프로세스 수를 조절한다(기본 4).

[실행 단계·산출물·환경](docs/reproduction.md) · [원문](https://arxiv.org/html/2406.09578v2) · [이전 연구](https://github.com/bucheoncityboy/korean-macro-factor-allocation)
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8',newline='\n')
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','xgboost','jumpmodels','cvxpy','osqp','matplotlib','pyarrow']}
    write_json(ROOT/'results/environment.json',{'python':sys.version,'platform':sys.platform,'libraries':versions,
        'xgboost_parameters':C['xgb'],'jumpmodels_commit':'d0fa00ce10126791695a259a45c5ddd41fbced80'})
    print(json.dumps({'paper_improvements':higher,'controlled_lr_sharpe':lr.sharpe,'controlled_xgb_sharpe':xgb.sharpe,'cer_interval':[primary.cer_low,primary.cer_high]},default=encode),flush=True)
`;

export const verification = String.raw`
def verify(save=False):
    from scipy.optimize import brentq
    d=load_data();p=pd.read_parquet(ROOT/'results/predictions.parquet')
    nav=pd.read_parquet(ROOT/'results/nav.parquet');targets=pd.read_parquet(ROOT/'results/targets.parquet');trades=pd.read_parquet(ROOT/'results/trades.parquet')
    require((p['asof']<=p.signal_date).all() and (p.signal_date<p.date).all(),'Forecast time order')
    require(np.isfinite(p[['p0','mu0','mu1']]).all().all(),'Nonfinite forecast')
    fit=json.loads((ROOT/'results/fits.json').read_text())
    profile_ids={profile['id'] for profile in C['profiles']}
    expected_jobs=[job for job in json.loads((ROOT/'results/jobs.json').read_text()) if job['profile'] in profile_ids]
    require(len(fit)==len(expected_jobs)*len(C['jump_penalties']),'Fit count')
    require(all(x['max_training_target']<=x['asof'] and x['train_end']<=x['asof'] for x in fit),'Training leakage')
    for key,frame in p.groupby(['profile','study','asset','date']):
        for col in ['asof','penalty','mu0','mu1','truth']:
            require(frame[col].nunique()==1,f'Models do not share {col}')
    for key,frame in p.groupby(['profile','study','classifier','asset']):
        profile=next(x for x in C['profiles'] if x['id']==key[0])
        expected=d['cal'][(d['cal']>=profile['start'])&(d['cal']<=C['end'])]
        require(pd.DatetimeIndex(frame.date).equals(expected),'Missing prediction dates')
    keys=['profile','study','strategy','cost_bp'];max_nav=0.;max_fee=0.;max_trade=0.;accounts=0;days=0
    ng={k:g for k,g in nav.groupby(keys)};tg={k:g for k,g in trades.groupby(keys)}
    for key,frame in targets.groupby(keys):
        frame=frame.sort_values('date');n=ng[key].set_index('date').loc[frame.date];ledger=tg[key].set_index('date').loc[frame.date]
        require(np.allclose(frame[ALL].sum(axis=1),1,atol=1e-8) and frame[ALL].min().min()>=-1e-9,'Invalid target')
        if key[2]!='60_40':require(frame[ASSETS].max().max()<=C['weight_cap']+1e-7,'Risk cap')
        expected_signals=d['cal'][d['cal'].get_indexer(frame.date)-1]
        require(np.array_equal(frame.signal_date.to_numpy(),expected_signals.to_numpy()),'Wrong signal date')
        units=np.zeros(13);first_signal=frame.signal_date.iloc[0]
        units[-1]=1/d['close'].loc[first_signal,C['cash']]
        cost=key[3]/10000
        for i,(date,w) in enumerate(zip(frame.date,frame[ALL].to_numpy())):
            op=d['open'].loc[date].to_numpy();cl=d['close'].loc[date].to_numpy();notional=units*op;total=notional.sum()
            value=total if cost==0 else brentq(lambda v:v+cost*np.abs(w*v-notional).sum()-total,total*(1-2*cost),total,xtol=1e-13)
            amounts=w*value-notional;fee=cost*np.abs(amounts).sum();units=w*value/op;wealth=units@cl
            max_nav=max(max_nav,abs(wealth-n.nav.iloc[i]));max_fee=max(max_fee,abs(fee-n.fee.iloc[i]))
            max_trade=max(max_trade,float(np.max(np.abs(amounts-ledger[ALL].iloc[i].to_numpy()))))
            require(abs(n.nav_before_trade.iloc[i]-n.nav_after_trade.iloc[i]-n.fee.iloc[i])<1e-9,'NAV-fee identity')
        accounts+=1;days+=len(frame)
        print(f'verified account {accounts}: {key}',flush=True)
    require(max_nav<1e-8 and max_fee<1e-9 and max_trade<1e-8,'Independent replay mismatch')
    for profile in C['profiles']:
        dates=d['cal'][(d['cal']>=profile['start'])&(d['cal']<=C['end'])]
        cash=nav.loc[(nav.profile==profile['id'])&(nav.strategy=='bil')].set_index('date')
        require(np.allclose(cash.net_return,d['returns'][C['cash']].loc[dates],atol=1e-12),'BIL benchmark reconciliation')
    summary=pd.read_csv(ROOT/'results/summary.csv')
    for _,row in summary.iterrows():
        f=nav.loc[(nav.profile==row.profile)&(nav.study==row.study)&(nav.strategy==row.strategy)&(nav.cost_bp==row.cost_bp)]
        if row.period=='common':f=f.loc[f.date>='2024-01-02']
        m=metrics(f,d['returns'][C['cash']].reindex(f.date).to_numpy())
        for name,value in m.items():require(abs(value-row[name])<1e-9,f'Summary drift {name}')
    result={'status':'passed','accounts':accounts,'account_days':days,'fits':len(fit),'prediction_rows':len(p),
        'independent_replay_max_nav_error':max_nav,'independent_replay_max_fee_error':max_fee,
        'independent_replay_max_trade_error':max_trade,'training_target_cutoff':'passed',
        'shared_model_conditions':'passed','complete_prediction_calendar':'passed','summary_recalculation':'passed',
        'future_online_prefix_checks':'performed for every JM fit'}
    if save:write_json(ROOT/'results/verification.json',result)
    print(json.dumps(result),flush=True)
`;
