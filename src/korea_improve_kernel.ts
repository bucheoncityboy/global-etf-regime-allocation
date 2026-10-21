/** Allocation-only follow-up. Numerical libraries and original forecasts stay locked. */
export const koreaImprovement=String.raw`

def set_improvement(config):
    global Q,IR
    Q=config;IR=ROOT/'results/korea_improvement'

def improve_specs():
    specs=[]
    for delay in Q['delays']:
        for v in Q['variants']:
            for model in Q['classifiers']:
                specs.append({'account':v['id']+'_'+model+'_d'+str(delay),'variant':v['id'],'classifier':model,'delay':delay,'cost_bp':Q['cost_bp']})
        for kind in Q['baselines']:
            specs.append({'account':'baseline_'+kind+'_d'+str(delay),'variant':'baseline','classifier':kind,'delay':delay,'cost_bp':Q['cost_bp']})
    return specs

def improve_mu(probability,mu0,mu1,variant):
    p=np.asarray(probability);lo=np.minimum(np.asarray(mu1),C['bearish_mean_cap'])
    mu=p*np.asarray(mu0)+(1-p)*lo if variant['probability_weighted'] else np.where(p>=.5,mu0,lo)
    return mu*variant['mean_multiplier']

def improve_buffer(proposal,previous,threshold,forced_cash):
    turnover=float(np.abs((proposal-previous)[:-1]).sum())
    valid_previous=previous[:-1].max()<=C['weight_cap']+1e-8
    buffered=bool(not forced_cash and threshold>0 and turnover<threshold and valid_previous)
    return (previous.copy() if buffered else proposal),buffered,turnover

def improve_context(d,p):
    dates=d['cal'][(d['cal']>=Q['evaluation_start'])&(d['cal']<=Q['end'])]
    streams={}
    for model in Q['classifiers']:
        f=p.loc[(p.study=='controlled')&(p.classifier==model)]
        streams[model]={col:f.pivot(index='date',columns='asset',values=col).loc[dates,ASSETS].to_numpy() for col in ['p0','mu0','mu1']}
        require(np.isfinite(streams[model]['p0']).all(),'Missing frozen probabilities')
    return {'dates':dates,'streams':streams,'covs':korea_covariances(d['excess']),
        'base':d['excess'].ewm(halflife=C['baseline_mean_halflife'],adjust=False,min_periods=252).mean()}

def improve_account(d,context,spec):
    delay=spec['delay'];model=spec['classifier'];variant=spec['variant'];baseline=variant=='baseline'
    rule=next((x for x in Q['variants'] if x['id']==variant),None)
    risk=C['baseline_mv_risk_aversion'] if baseline and model=='mv' else C['regime_risk_aversion']
    opt=KoreaOptimizer(risk,0 if baseline else 1,spec['cost_bp'])
    holding=np.r_[np.zeros(N),1.];nav=[];weights=[];trades=[]
    for i,date in enumerate(context['dates']):
        signal=d['cal'][d['cal'].get_loc(date)-1-delay];total=holding.sum();previous=holding/total
        buffered=False;forced=False;proposed_turnover=0.
        if (delay and i==0) or (baseline and model=='cash'):
            w=np.r_[np.zeros(N),1.];forced=True;proposed_turnover=float(np.abs(previous[:-1]).sum())
        elif baseline:
            if model=='ew':w=np.r_[np.full(N,1/N),0.]
            else:
                mu=np.full(N,.001) if model=='minvar' else context['base'].loc[signal].to_numpy()
                w=opt.solve(mu,context['covs'][signal],previous[:-1])
            proposed_turnover=float(np.abs((w-previous)[:-1]).sum())
        else:
            j=i-delay;s=context['streams'][model];p=s['p0'][j]
            forced=bool((p>=.5).sum()<C['minimum_bullish_assets'])
            proposal=np.r_[np.zeros(N),1.] if forced else opt.solve(improve_mu(p,s['mu0'][j],s['mu1'][j],rule),context['covs'][signal],previous[:-1])
            w,buffered,proposed_turnover=improve_buffer(proposal,previous,rule['turnover_buffer'],forced)
        after,fee,trade=index_settle(holding,w,spec['cost_bp']/10000)
        holding=after*(1+np.r_[d['returns'].loc[date].to_numpy(),d['rf'].loc[date]])
        nav.append([date,total,after.sum(),holding.sum(),holding.sum()/total-1,fee,fee/total,np.abs(trade[:-1]).sum()/total,1-w[-1]])
        weights.append([date,signal,*w,buffered,forced,proposed_turnover]);trades.append([date,*trade])
    n=pd.DataFrame(nav,columns=['date','nav_previous','nav_after_trade','nav','net_return','fee','cost_fraction','turnover','risk_exposure'])
    t=pd.DataFrame(weights,columns=['date','signal_date',*ALL,'buffered','forced_cash','proposal_turnover'])
    l=pd.DataFrame(trades,columns=['date',*ALL])
    for table in [n,t,l]:
        for key,value in spec.items():table[key]=value
    return n,t,l

def improve_run():
    d=korea_load();require(not d.get('synthetic_test_fixture',False),'Synthetic fixture cannot produce financial results')
    p=pd.read_parquet(KR/'predictions.parquet');context=improve_context(d,p)
    require(len(context['dates'])==1783,'Changed evaluation length')
    fingerprint=digest(IR/'design-freeze.json');cache=ROOT/'data/private/korea/improvement'/fingerprint;cache.mkdir(parents=True,exist_ok=True)
    tables=[[],[],[]]
    for number,spec in enumerate(improve_specs(),1):
        path=cache/(spec['account']+'.pkl')
        if path.exists():
            entry=pickle.loads(path.read_bytes());require(entry['spec']==spec,'Wrong account cache');frames=entry['frames'];cached=True
        else:
            frames=improve_account(d,context,spec);temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                handle.write(pickle.dumps({'spec':spec,'frames':frames},protocol=5));handle.flush();os.fsync(handle.fileno())
            temporary.replace(path);cached=False
        for collection,frame in zip(tables,frames):collection.append(frame)
        print(f"[{number}/24] {spec['account']} {'cached' if cached else 'computed'}",flush=True)
    for name,collection in zip(['nav','targets','trades'],tables):pd.concat(collection,ignore_index=True).to_parquet(IR/(name+'.parquet'),index=False)

def improve_original_matches(nav):
    original=pd.read_parquet(KR/'nav.parquet');maximum=0.;accounts=0
    for spec in improve_specs():
        if spec['variant']=='hard':
            old=original.loc[(original.study=='controlled')&(original.strategy=='mv_'+spec['classifier'])&(original.delay==spec['delay'])&(original.cost_bp==5)&original.cash_rule]
        elif spec['variant']=='baseline' and spec['delay']==0:
            old=original.loc[(original.study=='baseline')&(original.strategy==spec['classifier'])]
        else:continue
        one=nav.loc[nav.account==spec['account']].set_index('date').sort_index();two=old.set_index('date').sort_index()
        require(one.index.equals(two.index),'Original reference calendar mismatch')
        error=float(np.max(np.abs(one.nav-two.nav)));maximum=max(maximum,error);accounts+=1
        require(error<1e-8,'Original account was not reproduced: '+spec['account'])
    require(accounts==8,'Missing original-account reference checks')
    return {'accounts':accounts,'max_nav_error':maximum}

def improve_replay(d,nav,target,ledger):
    from scipy.optimize import brentq
    expected=d['cal'][(d['cal']>=Q['evaluation_start'])&(d['cal']<=Q['end'])]
    maximum=0.;count=0
    require(set(nav.account)==set(x['account'] for x in improve_specs()),'Account set changed')
    for spec in improve_specs():
        key=spec['account'];g=target.loc[target.account==key].set_index('date').sort_index()
        n=nav.loc[nav.account==key].set_index('date').sort_index();l=ledger.loc[ledger.account==key].set_index('date').sort_index()
        require(g.index.equals(expected) and n.index.equals(expected) and l.index.equals(expected),'Missing/duplicate evaluation sessions')
        require(np.isfinite(g[ALL]).all().all() and g[ALL].min().min()>=-1e-8,'Invalid weights')
        require(g[ASSETS].max().max()<=C['weight_cap']+1e-7 and np.allclose(g[ALL].sum(axis=1),1,atol=1e-10),'Cap/leverage violation')
        require(np.array_equal(g.signal_date.to_numpy(),d['cal'][d['cal'].get_indexer(expected)-1-spec['delay']].to_numpy()),'Signal timing violation')
        holding=np.r_[np.zeros(N),1.];cost=spec['cost_bp']/10000
        for date,w in zip(expected,g[ALL].to_numpy()):
            before=holding.sum();previous=holding/before
            if g.loc[date,'buffered']:
                require(spec['variant']=='buffer' and not g.loc[date,'forced_cash'],'Invalid buffer override')
                require(np.allclose(w,previous,atol=1e-10) and g.loc[date,'proposal_turnover']<.02,'Buffer traded or exceeded threshold')
            if g.loc[date,'forced_cash']:require(np.max(np.abs(w[:-1]))<1e-10,'Forced cash retained risky assets')
            value=brentq(lambda v:v+cost*np.abs((v*w-holding)[:-1]).sum()-before,before*(1-2*cost),before,xtol=1e-13)
            trade=value*w-holding;fee=cost*np.abs(trade[:-1]).sum()
            holding=value*w*(1+np.r_[d['returns'].loc[date].to_numpy(),d['rf'].loc[date]])
            error=abs(holding.sum()-n.loc[date,'nav']);maximum=max(maximum,float(error))
            require(error<1e-8 and abs(holding.sum()/before-1-n.loc[date,'net_return'])<1e-10,'NAV/return replay mismatch')
            require(np.max(np.abs(trade-l.loc[date,ALL].to_numpy(dtype=float)))<1e-8 and abs(fee-n.loc[date,'fee'])<1e-9,'Fee/trade replay mismatch')
            count+=1
    return {'accounts':len(improve_specs()),'account_days':count,'max_nav_error':maximum}

def improve_verify():
    d=korea_load();require(not d.get('synthetic_test_fixture',False),'Synthetic fixtures cannot validate financial results')
    korea_check_inputs();nav=pd.read_parquet(IR/'nav.parquet');t=pd.read_parquet(IR/'targets.parquet');l=pd.read_parquet(IR/'trades.parquet')
    proof=improve_replay(d,nav,t,l);proof['original_reference']=improve_original_matches(nav)
    require(proof['account_days']==24*1783,'Incomplete financial replay')
    write_json(IR/'verification.json',{'status':'passed',**proof,'prepared_sha256':digest(ROOT/'data/private/korea/prepared.pkl'),
        'forecasts_retrained':False,'new_untouched_holdout':False,'scope':'independent settlement replay with frozen real returns; allocation quality is not certified'})
    print('Independent improvement replay passed: '+str(proof),flush=True)

def improve_comparison_pairs():
    pairs=[]
    for delay in Q['delays']:
        suffix='_d'+str(delay)
        for model in Q['classifiers']:
            for a,b in [('probability','hard'),('shrink','probability'),('buffer','shrink'),('buffer','hard')]:
                pairs.append((a+'_'+model+suffix,b+'_'+model+suffix))
            pairs.append(('buffer_'+model+suffix,'baseline_minvar'+suffix))
        for v in Q['variants']:pairs.append((v['id']+'_xgb'+suffix,v['id']+'_lr'+suffix))
    return pairs

def improve_statistics():
    d=korea_load();nav=pd.read_parquet(IR/'nav.parquet');target=pd.read_parquet(IR/'targets.parquet')
    summaries=[];contributions=[];years=[];series={};gamma=Q['bootstrap']['risk_aversion']
    for spec in improve_specs():
        g=nav.loc[nav.account==spec['account']].set_index('date').sort_index();w=target.loc[target.account==spec['account']].set_index('date').sort_index()
        rf=d['rf'].loc[g.index];m=metrics(g,rf);series[spec['account']]=g.net_return.to_numpy()
        m['average_cash_weight']=float(w[ALL[-1]].mean());m['buffered_days']=int(w.buffered.sum());m['all_cash_days']=int((w[ALL[-1]]>1-1e-8).sum())
        m['annual_arithmetic_excess']=float((g.net_return-rf).mean()*252);summaries.append({**spec,'first':str(g.index[0].date()),'last':str(g.index[-1].date()),**m})
        risky=w[ASSETS]*d['excess'].loc[g.index,ASSETS];gross=rf+risky.sum(axis=1);fee_drag=g.cost_fraction*(1+gross)
        require(np.allclose(risky.sum(axis=1)-fee_drag,g.net_return-rf,atol=1e-12),'Contribution decomposition mismatch')
        for asset in ASSETS:contributions.append({'account':spec['account'],'component':asset,'annual_excess_contribution':float(risky[asset].mean()*252)})
        contributions.append({'account':spec['account'],'component':'fees','annual_excess_contribution':float(-fee_drag.mean()*252)})
        for year,group in g.groupby(g.index.year):years.append({'account':spec['account'],'year':int(year),**metrics(group,d['rf'].loc[group.index])})
    pd.DataFrame(summaries).to_csv(IR/'summary.csv',index=False);pd.DataFrame(contributions).to_csv(IR/'contributions.csv',index=False);pd.DataFrame(years).to_csv(IR/'years.csv',index=False)
    rng=np.random.default_rng(Q['bootstrap']['seed']);tests=[]
    def cer(r):return 252*(r.mean()-.5*gamma*r.var(ddof=1))
    for one,two in improve_comparison_pairs():
        a=series[one];b=series[two];length=len(a)
        for block in Q['bootstrap']['block_lengths']:
            samples=[]
            for _ in range(Q['bootstrap']['replications']):
                starts=rng.integers(0,length,size=int(np.ceil(length/block)));indices=((starts[:,None]+np.arange(block))%length).ravel()[:length]
                samples.append(cer(a[indices])-cer(b[indices]))
            low,high=np.quantile(samples,[.025,.975]);tests.append({'candidate':one,'reference':two,'estimate':float(cer(a)-cer(b)),
                'block_sessions':block,'replications':Q['bootstrap']['replications'],'ci_low':float(low),'ci_high':float(high),
                'scope':'post-result exploratory, conditional on frozen forecasts, no refitting or multiple-comparison adjustment'})
    write_json(IR/'bootstrap.json',tests)
    predictions=pd.read_parquet(KR/'predictions.parquet');rows=[]
    for model in Q['classifiers']:
        for v in Q['variants']:
            for asset in ASSETS:
                g=predictions.loc[(predictions.study=='controlled')&(predictions.classifier==model)&(predictions.asset==asset)].set_index('date').sort_index()
                mu=improve_mu(g.p0.to_numpy(),g.mu0.to_numpy(),g.mu1.to_numpy(),v);r=d['excess'][asset].loc[g.index].to_numpy()
                corr=float(np.corrcoef(mu,r)[0,1]) if np.std(mu)>1e-12 else None
                rows.append({'variant':v['id'],'classifier':model,'asset':asset,'ic':corr,'scope':'next-period excess return, before additional execution delay'})
    pd.DataFrame(rows).to_csv(IR/'return-ic.csv',index=False)

def improve_manifest():
    paths=['design-freeze.json','summary.csv','nav.parquet','targets.parquet','trades.parquet','bootstrap.json',
        'contributions.csv','years.csv','return-ic.csv','verification.json','status.json','README.md','nav.png']
    write_json(IR/'artifact-manifest.json',{'files':{p:digest(IR/p) for p in paths},'scope':'public output integrity; real-return replay additionally requires private source cache'})

def improve_verify_public():
    manifest=json.loads((IR/'artifact-manifest.json').read_text())
    for path,value in manifest['files'].items():require(digest(IR/path)==value,'Published artifact changed: '+path)
    n=pd.read_parquet(IR/'nav.parquet');s=pd.read_csv(IR/'summary.csv');cash=n.loc[n.account=='baseline_cash_d0'].set_index('date').sort_index().net_return
    require(len(s)==24 and len(n)==24*1783,'Incomplete public account set')
    for spec in improve_specs():
        g=n.loc[n.account==spec['account']].set_index('date').sort_index();stored=s.loc[s.account==spec['account']].iloc[0]
        require(g.index.equals(cash.index),'Public calendar mismatch')
        recalculated=metrics(g,cash)
        for key,value in recalculated.items():require(np.isclose(float(stored[key]),float(value),atol=1e-11,rtol=1e-10),'Published metric mismatch: '+key)
    proof=improve_original_matches(n)
    require(json.loads((IR/'verification.json').read_text())['status']=='passed','No financial verification')
    report=(IR/'README.md').read_text(encoding='utf-8');require(report.count('\n## ')==8,'Report format changed')
    print('Public artifact hashes, 24 account metrics and 8 original references passed: '+str(proof),flush=True)
`;
