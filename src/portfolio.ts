export const assembly = String.raw`
def cache_signature():
    return hashlib.sha256((digest(ROOT/'src/kernel.ts')+digest(ROOT/'configs/protocol.json')+digest(ROOT/'uv.lock')+C['input_snapshot_sha256']).encode()).hexdigest()[:20]
def assemble():
    d=load_data(); selected=[]; tuning=[]; fits=[]
    for profile in C['profiles']:
        pid=profile['id']; folds=d['folds'][pid]
        for a in ASSETS:
            chunks=[]
            for fold in folds:
                path=ROOT/'results/cache'/cache_signature()/pid/f"{a}_{fold['asof']}.pkl"
                item=pickle.loads(path.read_bytes()); chunks.append(item['predictions']); fits.extend(item['fits'])
            candidate=pd.concat(chunks).sort_values(['penalty','date'])
            streams={}
            for penalty,frame in candidate.groupby('penalty',sort=True):
                frame=frame.set_index('date').sort_index()
                require(not frame.index.has_duplicates,'Duplicate candidate dates')
                for name in C['classifiers']:
                    for h in set(C['paper_smoothing']+[C['controlled_smoothing']]):
                        streams[(penalty,name,h)]=pd.Series(smooth(frame[name].to_numpy(),h),index=frame.index)
            fixed_h=None
            for fold in folds:
                if fold['first']<profile['start']: continue
                asof=pd.Timestamp(fold['asof'])
                val_start=pd.Timestamp(fold['boundary'])-pd.DateOffset(years=profile['validation_years'])
                validation=d['cal'][(d['cal']>=val_start)&(d['cal']<=asof)]
                targets=d['cal'][(d['cal']>=fold['first'])&(d['cal']<=fold['last'])]
                close=d['close'][[a,C['cash']]]; op=d['open'][[a,C['cash']]]
                overnight=(op/close.shift()).reindex(validation).to_numpy()
                intraday=(close/op).reindex(validation).to_numpy()
                cash=d['returns'][C['cash']].reindex(validation).to_numpy()
                def score(penalty,name,h):
                    p=streams[(penalty,name,h)].reindex(validation)
                    require(not p.isna().any(),'Validation predictions unavailable')
                    return sharpe(binary_returns(p.to_numpy(),overnight,intraday,C['cost_bp']/10000),cash)
                choices=[]
                for h in C['paper_smoothing'] if fixed_h is None else [fixed_h]:
                    for penalty in C['jump_penalties']: choices.append((score(penalty,'xgb',h),-h,-penalty,h,penalty))
                best=max(choices); fixed_h=best[3]; paper_penalty=best[4]
                h=C['controlled_smoothing']
                shared=max((.5*(score(p,'lr',h)+score(p,'xgb',h)),-p,p) for p in C['jump_penalties'])
                for study,penalty,names,half,objective in [
                    ('paper',paper_penalty,['jm','xgb'],fixed_h,best[0]),
                    ('controlled',shared[2],C['classifiers'],h,shared[0])]:
                    tuning.append({'profile':pid,'asset':a,'study':study,'asof':asof,
                        'validation_first':validation[0],'validation_last':validation[-1],
                        'penalty':penalty,'halflife':half,'validation_score':objective,
                        'candidates':[{'penalty':p,'lr_sharpe':score(p,'lr',half),'xgb_sharpe':score(p,'xgb',half)} for p in C['jump_penalties']]})
                    f=candidate.loc[candidate.penalty==penalty].set_index('date').loc[targets]
                    for name in names:
                        half_used=0 if study=='paper' and name=='jm' else half
                        out=f[['signal_date','asof','penalty','mu0','mu1','truth']].copy()
                        out['p0']=streams[(penalty,name,half_used)].loc[targets]
                        out['profile']=pid;out['study']=study;out['classifier']=name;out['asset']=a
                        out['halflife']=half_used; selected.append(out.rename_axis('date').reset_index())
            print(f'assembled {pid} {a}',flush=True)
    pred=pd.concat(selected,ignore_index=True)
    require((pd.to_datetime(pred.signal_date)<pd.to_datetime(pred.date)).all(),'Signal/target order')
    pred.to_parquet(ROOT/'results/predictions.parquet',index=False)
    write_json(ROOT/'results/tuning.json',tuning)
    write_json(ROOT/'results/fits.json',fits)
`;

export const allocation = String.raw`
class Optimizer:
    def __init__(self,risk,trade,cost):
        import cvxpy as cp
        self.cp=cp;self.w=cp.Variable(12);self.mu=cp.Parameter(12);self.factor=cp.Parameter((12,12));self.prev=cp.Parameter(12)
        self.cap=C['weight_cap'];self.entry=trade*cost/10000
        self.problem=cp.Problem(cp.Maximize(self.mu@self.w-risk*cp.sum_squares(self.factor@self.w)
            -252*trade*cost/10000*cp.norm1(self.w-self.prev)),[self.w>=0,self.w<=self.cap,cp.sum(self.w)<=1])
    def solve(self,mu,cov,prev):
        chol=np.linalg.cholesky((cov+cov.T)*.5*252)
        if np.count_nonzero(prev)==0 and np.all(mu<=self.entry): return np.r_[np.zeros(12),1.]
        self.mu.value=mu*252;self.factor.value=chol.T;self.prev.value=prev
        self.problem.solve(solver='OSQP',eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True,warm_start=True,enforce_dpp=True)
        require(self.problem.status=='optimal',f'Optimizer status {self.problem.status}')
        w=np.asarray(self.w.value).ravel()
        require(w.min()>=-1e-7 and w.max()<=self.cap+1e-7 and w.sum()<=1+1e-7,'Weight constraints violated')
        w=w.clip(0,self.cap)
        if w.sum()>1:w/=w.sum()
        return np.r_[w,1-w.sum()]
def covariance_series(returns):
    delta=2**(-1/C['covariance_halflife']);s0=0.;q=0.;s1=np.zeros(12);s2=np.zeros((12,12));result={}
    for i,(date,row) in enumerate(returns.dropna().iterrows()):
        x=row.to_numpy();s0=delta*s0+1;q=delta**2*q+1;s1=delta*s1+x;s2=delta*s2+np.outer(x,x)
        if i>=251:result[date]=(s2-np.outer(s1,s1)/s0)/(s0-q/s0)
    return result
def account(d,dates,decide,cost):
    cal=d['cal'];signals=cal[cal.get_indexer(dates)-1]
    prevclose=d['close'].loc[signals].to_numpy();opens=d['open'].loc[dates].to_numpy();closes=d['close'].loc[dates].to_numpy()
    require(np.isfinite(np.r_[prevclose,opens,closes]).all(),'Missing execution quote')
    holding=np.r_[np.zeros(12),1.];navs=[];targets=[];ledgers=[]
    for i,date in enumerate(dates):
        prevnav=holding.sum();prev=holding/prevnav
        w=np.asarray(decide(i,signals[i],prev.copy()))
        targets.append([date,signals[i],*w])
        atopen=holding*(opens[i]/prevclose[i]);after,fee,traded=settle(atopen,w,cost/10000)
        holding=after*(closes[i]/opens[i]);nav=holding.sum()
        navs.append([date,prevnav,atopen.sum(),after.sum(),nav,nav/prevnav-1,fee,fee/prevnav,
            np.abs(traded).sum()/atopen.sum(),1-w[-1],atopen.sum()-prevnav,nav-after.sum()])
        ledgers.append([date,*traded])
    nav=pd.DataFrame(navs,columns=['date','nav_previous','nav_before_trade','nav_after_trade','nav','net_return',
        'fee','cost_fraction','turnover','risk_exposure','overnight_pnl','intraday_pnl'])
    target=pd.DataFrame(targets,columns=['date','signal_date',*ALL])
    ledger=pd.DataFrame(ledgers,columns=['date',*ALL])
    return nav,target,ledger
def metrics(nav,cash):
    r=nav.net_return.to_numpy();x=r-np.asarray(cash);wealth=np.r_[1.,np.cumprod(1+r)]
    return {'days':len(r),'cagr':float(wealth[-1]**(252/len(r))-1),
        'sharpe':sharpe(r,cash),'mdd':float((wealth/np.maximum.accumulate(wealth)-1).min()),
        'volatility':float(r.std(ddof=1)*np.sqrt(252)),
        'cer':float((r.mean()-.5*C['bootstrap']['risk_aversion']*r.var(ddof=1))*252),
        'annual_cost':float(nav.cost_fraction.mean()*252),'annual_turnover':float(nav.turnover.mean()*252),
        'risk_exposure':float(nav.risk_exposure.mean())}
def backtest():
    d=load_data();pred=pd.read_parquet(ROOT/'results/predictions.parquet')
    covs=covariance_series(d['excess'])
    unconditional=d['excess'].ewm(halflife=C['baseline_mean_halflife'],adjust=False,min_periods=252).mean()
    navs=[];targets=[];ledgers=[];summaries=[];individual=[]
    for profile in C['profiles']:
        pid=profile['id'];dates=d['cal'][(d['cal']>=profile['start'])&(d['cal']<=C['end'])]
        cash=d['returns'][C['cash']].loc[dates].to_numpy()
        streams={}
        pp=pred.loc[pred.profile==pid]
        for (study,name),frame in pp.groupby(['study','classifier']):
            streams[(study,name)]={col:frame.pivot(index='date',columns='asset',values=col).loc[dates,ASSETS].to_numpy()
                                  for col in ['p0','mu0','mu1']}
        specs=[('baseline',name,5) for name in ['mv','minvar','ew','bil','60_40']]
        specs += [('paper',name,5) for name in ['mv_xgb','minvar_xgb','ew_xgb']]
        specs += [('controlled','mv_'+name,5) for name in C['classifiers']]
        if pid=='main':specs += [('controlled','mv_'+name,c) for name in ['lr','xgb'] for c in C['cost_sensitivity_bp']]
        for study,name,cost in specs:
            base=name.split('_')[0]
            risk=C['baseline_mv_risk_aversion'] if study=='baseline' and base=='mv' else C['regime_risk_aversion']
            trade=0 if study=='baseline' else 1
            opt=Optimizer(risk,trade,cost)
            stream=streams.get((study,name.split('_')[-1]))
            def decide(i,signal,previous):
                if name=='bil':return np.r_[np.zeros(12),1.]
                if name=='60_40':
                    w=np.zeros(13);w[0]=.6;w[5]=.4;return w
                if study=='baseline':
                    if base=='ew':return np.r_[np.full(12,1/12),0.]
                    mu=np.full(12,.001) if base=='minvar' else unconditional.loc[signal].to_numpy()
                else:
                    bull=stream['p0'][i]>=.5
                    if bull.sum()<C['minimum_bullish_assets']:return np.r_[np.zeros(12),1.]
                    if base=='ew':return np.r_[bull/bull.sum(),0.]
                    mu=(np.where(bull,.001,0.) if base=='minvar' else
                        np.where(bull,stream['mu0'][i],np.minimum(stream['mu1'][i],C['bearish_mean_cap'])))
                return opt.solve(mu,covs[signal],previous[:12])
            nav,target,ledger=account(d,dates,decide,cost)
            tags={'profile':pid,'study':study,'strategy':name,'cost_bp':cost}
            for frame in [nav,target,ledger]:
                for k,v in tags.items():frame[k]=v
            navs.append(nav);targets.append(target);ledgers.append(ledger)
            for period,start in [('full',profile['start']),('common','2024-01-02')]:
                sub=nav.loc[nav.date>=start];rf=d['returns'][C['cash']].reindex(sub.date).to_numpy()
                summaries.append({**tags,'period':period,'first':str(sub.date.min().date()),'last':str(sub.date.max().date()),**metrics(sub,rf)})
            print(f'account {pid} {study} {name} {cost}bp Sharpe={summaries[-2]["sharpe"]:.3f}',flush=True)
        for a in ASSETS:
            columns=[a,C['cash']];cl=d['close'][columns];op=d['open'][columns]
            over=(op/cl.shift()).loc[dates].to_numpy();day=(cl/op).loc[dates].to_numpy()
            for name in ['buy_hold','jm','xgb']:
                prob=np.ones(len(dates)) if name=='buy_hold' else pp.loc[(pp.study=='paper')&(pp.classifier==name)&(pp.asset==a)].set_index('date').p0.loc[dates].to_numpy()
                rr=binary_returns(prob,over,day,C['cost_bp']/10000)
                nav=pd.DataFrame({'net_return':rr,'cost_fraction':np.zeros(len(rr)),'turnover':np.zeros(len(rr)),
                    'risk_exposure':(prob>=.5).astype(float)})
                individual.append({'profile':pid,'asset':a,'strategy':name,**metrics(nav,cash)})
    pd.concat(navs).to_parquet(ROOT/'results/nav.parquet',index=False)
    pd.concat(targets).to_parquet(ROOT/'results/targets.parquet',index=False)
    pd.concat(ledgers).to_parquet(ROOT/'results/trades.parquet',index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'results/summary.csv',index=False)
    pd.DataFrame(individual).drop(columns=['annual_cost','annual_turnover']).to_csv(ROOT/'results/individual.csv',index=False)
`;
