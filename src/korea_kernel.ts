/** Locked-library numerical routines for the Korean transfer experiment. */
export const koreaNumerics=String.raw`

def set_korea(config):
    global C,ASSETS,ALL,N,KR,MC
    C=config;ASSETS=C['assets'];ALL=ASSETS+[C['cash']];N=len(ASSETS);KR=ROOT/'results/korea'
    MC=['yield3_change','slope10_3','slope10_3_change','vix_change','stock_bond_corr']

def korea_load():return pickle.loads((ROOT/'data/private/korea/prepared.pkl').read_bytes())

def korea_input_receipt():
    paths=['results/korea/source-manifest.json','results/korea/bond-source-manifest.json','data/input_snapshot.zip']
    return {p:digest(ROOT/p) for p in paths}

def korea_check_inputs():
    receipt=json.loads((KR/'input-receipt.json').read_text())
    require(receipt['input_files']==korea_input_receipt(),'Korean source manifest/snapshot changed')
    require(receipt['prepared_sha256']==digest(ROOT/'data/private/korea/prepared.pkl'),'Prepared Korean input changed')
    for manifest in ['source-manifest.json','bond-source-manifest.json']:
        for item in json.loads((KR/manifest).read_text())['sources']:
            require(digest(ROOT/'data/private/korea/raw'/item['path'])==item['sha256'],'Korean raw response changed: '+item['path'])

def korea_prior_observations(s,cal,name):
    # A long Korean exchange closure is not a missing published observation.
    # Keep the calendar-day limit unless the source reaches the prior XKRX session.
    pos=s.index.searchsorted(cal,side='left')-1;ok=pos>=0
    out=pd.Series(np.nan,index=cal);out.loc[ok]=s.iloc[pos[ok]].to_numpy()
    observed=pd.Series(pd.NaT,index=cal);observed.loc[ok]=s.index[pos[ok]].to_numpy()
    previous=pd.Series(cal,index=cal).shift(1)
    age=pd.Series(np.nan,index=cal);age.loc[ok]=(cal[ok]-s.index[pos[ok]]).days
    long_gap=age>10;verified_closure=observed>=previous
    require(not (long_gap & ~verified_closure).any(),'Stale ECOS observation '+name)
    closures=[{'signal_date':date,'observation_date':observed.loc[date],
        'previous_krx_session':previous.loc[date],'calendar_days':int(age.loc[date])}
        for date in cal[long_gap & verified_closure]]
    return out,{'max_age_calendar_days':int(age.max()),'verified_exchange_closures':closures}

def korea_prepare():
    import re,io,pandas_market_calendars as pmc
    raw=ROOT/'data/private/korea/raw';stock={}
    for name in ['KOSPI','KOSDAQ']:
        rows=[json.loads(line.rstrip().rstrip(',')) for line in (raw/f'naver-{name}.txt').read_text().splitlines() if re.match(r'^\["\d{8}",',line)]
        f=pd.DataFrame(rows,columns=['date','open','high','low','close','volume','foreign']);f['date']=pd.to_datetime(f.date)
        require(not f.date.duplicated().any(),'Duplicate Naver stock session')
        stock[name]=f.set_index('date').sort_index().loc[C['raw_start']:C['end'],'close'].astype(float)
    cal=pmc.get_calendar('XKRX').schedule(C['raw_start'],C['end']).index
    for name,s in stock.items():require(s.index.equals(cal) and (s>0).all(),f'KRX calendar mismatch {name}')
    levels=pd.DataFrame(stock,index=cal);agg=pd.Series(index=cal,dtype=float);cross=[];cross_match=[];cross_issues=[]
    previous=None;rounding_count=0;rounding_max_error=0.
    hk=json.loads((raw/'hankyung-kis.json').read_text())['KISKBI']['chart']['total']
    hk=pd.Series([x[1] for x in hk],index=pd.to_datetime([x[0] for x in hk],unit='ms'))
    for date in cal:
        data=json.loads((raw/'kis'/f'{date:%Y%m%d}.json').read_text())
        items={x['indexId']:x for x in data};require(len(items)==len(data),'Duplicate KIS index identity')
        require('BMM01' in items,'Missing KIS aggregate index')
        agg.loc[date]=float(items['BMM01']['value01'])
        if date in hk.index:
            secondary=hk.loc[date];error=abs(agg.loc[date]-secondary) if np.isfinite(secondary) else None
            matched=error is not None and error<.011;cross_match.append(matched)
            if error is not None:cross.append(float(error))
            if not matched:cross_issues.append({'date':date,'primary_kis':agg.loc[date],'secondary_hankyung':float(secondary) if np.isfinite(secondary) else None})
        for a,spec in C['series'].items():
            if spec['provider']!='KIS':continue
            require(spec['ticker'] in items,f'Missing observed KIS level {a} {date}')
            require(re.sub(r'\s+','',items[spec['ticker']]['indexName'])==re.sub(r'\s+','',spec['name']),'KIS ticker/name mismatch '+a)
            levels.loc[date,a]=float(items[spec['ticker']]['value01'])
            if previous is not None and (date-previous[0]).days==1 and items[spec['ticker']]['value02'] is not None:
                now=levels.loc[date,a];old=float(previous[1][spec['ticker']]['value01'])
                error=abs(now/old-1-float(items[spec['ticker']]['value02'])/100)
                bound=.005/old+.005*now/old**2+1e-6
                require(error<=bound,f'KIS level/daily-return mismatch outside published rounding: {a} {date}')
                rounding_count+=1;rounding_max_error=max(rounding_max_error,error)
        previous=(date,items)
    require(len(cross_match)>5000 and np.mean(cross_match)>=.99,'KIS/Hankyung aggregate identity alignment below 99%')
    known={};observations={};ages={};observation_checks={}
    for name,spec in C['ecos'].items():
        entries=[]
        for path in raw.glob(name+'_*.json'):
            if path.name.endswith('.meta.json'):continue
            payload=json.loads(path.read_text());entries.extend(payload['StatisticSearch']['row'])
        require(entries and all(x['STAT_CODE']==spec['table'] and x['ITEM_CODE1']==spec['item'] and x['ITEM_NAME1']==spec['name'] for x in entries),'Wrong ECOS identity '+name)
        s=pd.Series([float(x['DATA_VALUE']) for x in entries],index=pd.to_datetime([x['TIME'] for x in entries])).sort_index()
        require(not s.index.has_duplicates,'Duplicate ECOS observation '+name);observations[name]=s
        # The first raw sessions can precede the first eligible lagged observation; they are warmup only.
        known[name],observation_checks[name]=korea_prior_observations(s,cal,name)
        ages[name]=observation_checks[name]['max_age_calendar_days']
    levels['USD']=observations['USD'].reindex(cal)
    levels=levels.reindex(columns=ASSETS);require(levels.notna().all().all() and (levels>0).all().all(),'Missing asset level; no interpolation allowed')
    returns=levels.pct_change(fill_method=None);rf=known['cash']/100/252
    excess=returns.subtract(rf,axis=0);m=pd.DataFrame(index=cal)
    m[MC[0]]=known['yield3'].diff().ewm(halflife=21,adjust=False).mean()
    slope=known['yield10']-known['yield3'];m[MC[1]]=slope.ewm(halflife=10,adjust=False).mean()
    m[MC[2]]=slope.diff().ewm(halflife=21,adjust=False).mean()
    with zipfile.ZipFile(ROOT/'data/input_snapshot.zip') as z:
        raw_vix=z.read('paper_replication_next/fred/VIXCLS.csv');desc=json.loads(z.read('paper_replication_next/fred/VIXCLS.csv.meta.json'))
        require(hashlib.sha256(raw_vix).hexdigest()==desc['sha256'],'VIX snapshot changed')
        v=pd.read_csv(io.BytesIO(raw_vix),index_col=0,parse_dates=True).VIXCLS
        v=pd.to_numeric(v,errors='coerce').dropna();pos=v.index.searchsorted(cal,side='left')-1
        require((pos>=0).all() and (cal-v.index[pos]).days.max()<=10,'Missing/stale preceding-date VIX')
        v=pd.Series(v.iloc[pos].to_numpy(),index=cal)
    m[MC[3]]=np.log(v).diff().ewm(halflife=63,adjust=False).mean()
    m[MC[4]]=returns.KOSPI.rolling(252).corr(agg.pct_change(fill_method=None))
    features={}
    for a in ASSETS:
        x=excess[a];f=pd.DataFrame(index=cal)
        means={h:x.ewm(halflife=h,adjust=False,min_periods=63).mean() for h in [5,10,21]}
        dd={h:np.sqrt(x.clip(upper=0).pow(2).ewm(halflife=h,adjust=False,min_periods=63).mean()) for h in [5,10,21]}
        for h in [5,21]:f[f'log_dd_{h}']=np.log(dd[h].clip(lower=1e-12))
        for h in [5,10,21]:f[f'mean_{h}']=means[h]
        for h in [5,10,21]:f[f'sortino_{h}']=means[h]/dd[h].clip(lower=1e-12)
        features[a]=pd.concat([f,m],axis=1).dropna();require(np.isfinite(features[a]).all().all(),'Invalid feature')
        require(features[a].index.min()<pd.Timestamp(C['train_start']),'Insufficient feature warmup')
    jobs=[]
    for boundary in pd.date_range(C['validation_start'],C['end'],freq='6MS'):
        dates=cal[(cal>=boundary)&(cal<boundary+pd.DateOffset(months=6))];asof=cal[cal.get_loc(dates[0])-1]
        for a in ASSETS:jobs.append({'asset':a,'boundary':str(boundary.date()),'asof':str(asof.date()),
            'train_start':str((boundary-pd.DateOffset(years=11)).date()),'first':str(dates[0].date()),'last':str(dates[-1].date())})
    require(len(jobs)==25*N,'Incorrect half-year schedule')
    data={'cal':cal,'levels':levels,'returns':returns,'rf':rf,'excess':excess,'features':features,'jobs':jobs}
    (ROOT/'data/private/korea/prepared.pkl').write_bytes(pickle.dumps(data,protocol=5))
    write_json(KR/'jobs.json',jobs)
    write_json(KR/'input-receipt.json',{'input_files':korea_input_receipt(),'prepared_sha256':digest(ROOT/'data/private/korea/prepared.pkl'),'sessions':len(cal),'first':cal[0],'last':cal[-1],
        'assets':C['series'],'market_max_age_days':ages,'observation_availability_checks':observation_checks,
        'price_interpolations':0,'calendar_intersection_drops':0,
        'calendar_library':importlib.metadata.version('pandas-market-calendars'),'calendar':list(cal.strftime('%Y-%m-%d')),
        'cross_checks':{'kis_hankyung_days':len(cross_match),'match_fraction_within_0_011':float(np.mean(cross_match)),
            'max_secondary_difference':max(cross),'secondary_discrepancies':cross_issues,
            'kis_level_vs_reported_return_checks':rounding_count,'max_return_rounding_error':rounding_max_error},
        'feature_start':{a:f.index[0] for a,f in features.items()},
        'runtime':{'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','xgboost','jumpmodels','cvxpy','osqp']}}})
    korea_check_inputs();print(f'Korean panel prepared: {len(cal)} sessions; {len(jobs)} fold/asset jobs',flush=True)

def korea_cache():
    key=hashlib.sha256((digest(ROOT/'src/kernel.ts')+digest(ROOT/'src/korea_kernel.ts')+digest(ROOT/'configs/korea.json')+digest(KR/'input-receipt.json')+digest(ROOT/'uv.lock')).encode()).hexdigest()
    return ROOT/'data/private/korea/cache'/key

def korea_fit(job,d,cache_dir=None):
    from jumpmodels.jump import JumpModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    path=(cache_dir or korea_cache())/f"{job['asset']}_{job['asof']}.pkl"
    if path.exists():return 'cached'
    a=job['asset'];f=d['features'][a];asof=pd.Timestamp(job['asof']);cal=d['cal']
    train=f.loc[job['train_start']:asof];require(len(train)>2500,'Incomplete 11-year training window')
    targets=cal[(cal>=job['first'])&(cal<=job['last'])];signals=cal[cal.get_indexer(targets)-1]
    cols=[x for x in RC if not(x.startswith('log_dd') and a in C['jump_exclude_dd'])]
    scale=StandardScaler().fit(train[cols]);z=pd.DataFrame(scale.transform(train[cols]),index=train.index,columns=cols)
    rows=[];fits=[]
    with threadpool_limits(limits=1):
        for penalty in C['jump_penalties']:
            jm=JumpModel(n_components=2,jump_penalty=penalty,n_init=C['jump_n_init'],random_state=C['seed'])
            jm.fit(z,ret_ser=d['excess'][a].reindex(train.index),sort_by='cumret')
            labels=pd.Series(np.asarray(jm.labels_,dtype=int),index=train.index)
            nd=cal[cal.get_indexer(train.index)+1];y=labels.reindex(nd).to_numpy();valid=(nd<=asof)&np.isfinite(y)
            require(not valid[-1] and nd[valid].max()<=asof,'Training target leak')
            x=train.loc[valid,RC+MC];y=y[valid].astype(int);scaler=StandardScaler().fit(x)
            xx=scaler.transform(x);future=scaler.transform(f.loc[signals,RC+MC]);prior=(np.bincount(y,minlength=2)+1)/(len(y)+2)
            pred={}
            for name in ['lr','xgb']:
                if len(np.unique(y))==1:p=np.full(len(targets),prior[0])
                else:
                    model=LogisticRegression(**C['logistic'],random_state=C['seed']) if name=='lr' else XGBClassifier(**C['xgb'],random_state=C['seed'],objective='binary:logistic',eval_metric='logloss')
                    model.fit(xx,y);p=model.predict_proba(future)[:,0]
                pred[name]=p
            online_x=f.loc[job['train_start']:targets[-1],cols]
            zz=pd.DataFrame(scale.transform(online_x),index=online_x.index,columns=cols)
            online=pd.Series(jm.predict_online(zz),index=zz.index)
            require(np.array_equal(jm.predict_online(zz.loc[:asof]),online.loc[:asof]),'Future-dependent JM online prediction')
            pred['jm']=1-online.reindex(signals).to_numpy(dtype=int)
            means=d['excess'][a].reindex(train.index).groupby(labels).mean().reindex([0,1]).fillna(d['excess'][a].reindex(train.index).mean())
            fits.append({'asset':a,'asof':asof,'penalty':penalty,'train_start':train.index[0],'train_end':train.index[-1],
                'last_train_target':nd[valid].max(),'n_train':len(y),'label_hash':frame_hash(labels),'features_hash':frame_hash(x),'means':means.to_numpy()})
            for i,date in enumerate(targets):rows.append({'date':date,'signal_date':signals[i],'asof':asof,'penalty':penalty,
                'mu0':means.iloc[0],'mu1':means.iloc[1],'truth':int(online.loc[date]),**{name:float(v[i]) for name,v in pred.items()}})
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_bytes(pickle.dumps({'predictions':pd.DataFrame(rows),'fits':fits},protocol=5));temp.replace(path);return 'fit'

def korea_worker():
    d=korea_load()
    for line in sys.stdin:
        job=json.loads(line);status=korea_fit(job,d);print(json.dumps({'done':job,'status':status}),flush=True)

def index_binary(p,r,rf,c):
    on=np.asarray(p)>=.5;previous=np.r_[False,on[:-1]]
    charge=np.where(on&~previous,1/(1+c),np.where(~on&previous,1-c,1.))
    return charge*(1+np.where(on,r,rf))-1

def index_settle(holding,w,c):
    require(np.isfinite(w).all() and w.min()>=-1e-9 and abs(w.sum()-1)<1e-8,'Invalid target')
    total=holding.sum();lo=total*(1-2*c);hi=total
    for _ in range(48):
        mid=(lo+hi)/2
        if mid+c*np.abs((mid*w-holding)[:-1]).sum()>total:hi=mid
        else:lo=mid
    value=(lo+hi)/2;trades=value*w-holding;fee=c*np.abs(trades[:-1]).sum()
    require(abs(value+fee-total)<1e-10*max(1,total),'Self-financing failure')
    return value*w,fee,trades

def korea_select(d,cache_dir=None):
    selected=[];tuning=[];allfits=[]
    for a in ASSETS:
        chunks=[]
        jobs=[x for x in d['jobs'] if x['asset']==a]
        for job in jobs:
            item=pickle.loads(((cache_dir or korea_cache())/f"{a}_{job['asof']}.pkl").read_bytes());chunks.append(item['predictions']);allfits.extend(item['fits'])
        candidate=pd.concat(chunks);streams={}
        for penalty,g in candidate.groupby('penalty'):
            f=g.set_index('date').sort_index();require(not f.index.has_duplicates,'Duplicate forecast session')
            for model in ['jm','lr','xgb']:
                for h in C['halflives']:streams[(penalty,model,h)]=pd.Series(smooth(f[model].to_numpy(),h),index=f.index)
        fixed_h=None
        for job in jobs:
            if job['boundary']<C['evaluation_start']:continue
            asof=pd.Timestamp(job['asof']);start=pd.Timestamp(job['boundary'])-pd.DateOffset(years=5)
            validation=d['cal'][(d['cal']>=start)&(d['cal']<=asof)];targets=d['cal'][(d['cal']>=job['first'])&(d['cal']<=job['last'])]
            def score(penalty,model,h):
                pp=streams[(penalty,model,h)].reindex(validation);require(pp.notna().all(),'Incomplete five-year validation')
                rr=index_binary(pp,d['returns'][a].loc[validation],d['rf'].loc[validation],C['cost_bp']/10000)
                return sharpe(rr,d['rf'].loc[validation])
            choices=[(score(p,'xgb',h),-h,-p,h,p) for h in (C['halflives'] if fixed_h is None else [fixed_h]) for p in C['jump_penalties']]
            best=max(choices);fixed_h=best[3]
            half=C['controlled_halflife'];shared=max((.5*(score(p,'lr',half)+score(p,'xgb',half)),-p,p) for p in C['jump_penalties'])
            for study,penalty,h,names in [('paper',best[4],fixed_h,['jm','xgb']),('controlled',shared[2],half,['jm','lr','xgb'])]:
                tuning.append({'asset':a,'study':study,'asof':asof,'validation_first':validation[0],'validation_last':validation[-1],
                    'penalty':penalty,'halflife':h,'xgb_score':score(penalty,'xgb',h),'lr_score':score(penalty,'lr',h)})
                f=candidate.loc[candidate.penalty==penalty].set_index('date').loc[targets]
                for model in names:
                    h_used=0 if study=='paper' and model=='jm' else h
                    out=f[['signal_date','asof','penalty','mu0','mu1','truth']].copy();out['p0']=streams[(penalty,model,h_used)].loc[targets]
                    out['asset']=a;out['study']=study;out['classifier']=model;out['halflife']=h_used
                    selected.append(out.rename_axis('date').reset_index())
        print(f'Selected Korea {a}',flush=True)
    p=pd.concat(selected,ignore_index=True);p.to_parquet(KR/'predictions.parquet',index=False)
    write_json(KR/'fits.json',allfits);write_json(KR/'tuning.json',tuning);return p

class KoreaOptimizer:
    def __init__(self,risk,trade,cost):
        import cvxpy as cp
        self.w=cp.Variable(N);self.mu=cp.Parameter(N);self.factor=cp.Parameter((N,N));self.prev=cp.Parameter(N)
        self.cap=C['weight_cap'];self.entry=trade*cost/10000
        self.problem=cp.Problem(cp.Maximize(self.mu@self.w-risk*cp.sum_squares(self.factor@self.w)
            -252*trade*cost/10000*cp.norm1(self.w-self.prev)),[self.w>=0,self.w<=self.cap,cp.sum(self.w)<=1])
    def solve(self,mu,cov,previous):
        if np.count_nonzero(previous)==0 and np.all(mu<=self.entry):return np.r_[np.zeros(N),1.]
        self.mu.value=np.asarray(mu)*252;self.factor.value=np.linalg.cholesky(cov*252).T;self.prev.value=previous
        self.problem.solve(solver='OSQP',eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True,warm_start=True,enforce_dpp=True)
        require(self.problem.status=='optimal','Optimizer failed: '+str(self.problem.status));w=np.asarray(self.w.value).ravel()
        require(w.min()>=-1e-7 and w.max()<=self.cap+1e-7 and w.sum()<=1+1e-7,'Allocation constraint')
        w=w.clip(0,self.cap)
        if w.sum()>1:w/=w.sum()
        return np.r_[w,1-w.sum()]

def korea_covariances(excess):
    delta=2**(-1/252);s0=0.;q=0.;s1=np.zeros(N);s2=np.zeros((N,N));out={}
    for i,(date,row) in enumerate(excess.dropna().iterrows()):
        x=row.to_numpy();s0=delta*s0+1;q=delta**2*q+1;s1=delta*s1+x;s2=delta*s2+np.outer(x,x)
        if i>=251:
            cov=(s2-np.outer(s1,s1)/s0)/(s0-q/s0);values,vectors=np.linalg.eigh((cov+cov.T)/2)
            out[date]=(vectors*np.maximum(values,1e-12))@vectors.T
    return out

def korea_accounts(d,p):
    dates=d['cal'][(d['cal']>=C['evaluation_start'])&(d['cal']<=C['end'])];cal=d['cal']
    covs=korea_covariances(d['excess']);base=d['excess'].ewm(halflife=1260,adjust=False,min_periods=252).mean()
    streams={}
    for (study,model),g in p.groupby(['study','classifier']):
        streams[(study,model)]={col:g.pivot(index='date',columns='asset',values=col).loc[dates,ASSETS].to_numpy() for col in ['p0','mu0','mu1']}
    specs=[('baseline',x,5,0,True) for x in ['mv','minvar','ew','cash']]
    specs += [('paper',x+'_xgb',5,0,True) for x in ['mv','minvar','ew']]
    specs += [('controlled',x+'_'+model,5,0,True) for x in ['mv','minvar','ew'] for model in ['jm','lr','xgb']]
    specs += [('controlled','mv_'+model,cost,0,True) for cost in C['cost_sensitivity_bp'] for model in ['jm','lr','xgb']]
    specs += [('controlled','mv_'+model,5,1,True) for model in ['jm','lr','xgb']]
    specs += [('controlled','mv_'+model,5,0,False) for model in ['jm','lr','xgb']]
    allnav=[];alltargets=[];alltrades=[];summaries=[]
    for study,name,cost,delay,cash_rule in specs:
        regime=study!='baseline';kind=name.split('_')[0];risk=5 if name=='mv' else 10
        opt=KoreaOptimizer(risk,1 if regime else 0,cost);holding=np.r_[np.zeros(N),1.];nav=[];weights=[];trades=[]
        tag={'study':study,'strategy':name,'cost_bp':cost,'delay':delay,'cash_rule':cash_rule}
        stream=streams.get((study,name.split('_')[-1]))
        for i,date in enumerate(dates):
            signal=cal[cal.get_loc(date)-1-delay];total=holding.sum();previous=holding/total
            if name=='cash' or (delay and i==0):w=np.r_[np.zeros(N),1.]
            elif name=='ew':w=np.r_[np.full(N,1/N),0.]
            else:
                if regime:
                    j=i-delay;bull=stream['p0'][j]>=.5
                    if cash_rule and bull.sum()<C['minimum_bullish_assets']:w=np.r_[np.zeros(N),1.]
                    elif kind=='ew':
                        if bull.sum()==0:w=np.r_[np.zeros(N),1.]
                        else:
                            w=np.minimum(bull/bull.sum(),C['weight_cap']);w=np.r_[w,1-w.sum()]
                    else:
                        mu=np.where(bull,.001,0.) if kind=='minvar' else np.where(bull,stream['mu0'][j],np.minimum(stream['mu1'][j],-.001))
                        w=opt.solve(mu,covs[signal],previous[:N])
                else:
                    mu=np.full(N,.001) if kind=='minvar' else base.loc[signal].to_numpy();w=opt.solve(mu,covs[signal],previous[:N])
            after,fee,trade=index_settle(holding,w,cost/10000);holding=after*(1+np.r_[d['returns'].loc[date].to_numpy(),d['rf'].loc[date]])
            nav.append([date,total,after.sum(),holding.sum(),holding.sum()/total-1,fee,fee/total,np.abs(trade[:-1]).sum()/total,1-w[-1]])
            weights.append([date,signal,*w]);trades.append([date,*trade])
        frame=pd.DataFrame(nav,columns=['date','nav_previous','nav_after_trade','nav','net_return','fee','cost_fraction','turnover','risk_exposure'])
        target=pd.DataFrame(weights,columns=['date','signal_date',*ALL]);ledger=pd.DataFrame(trades,columns=['date',*ALL])
        for table in [frame,target,ledger]:
            for k,v in tag.items():table[k]=v
        allnav.append(frame);alltargets.append(target);alltrades.append(ledger)
        summaries.append({**tag,'first':str(dates[0].date()),'last':str(dates[-1].date()),**metrics(frame,d['rf'].loc[dates])})
        print('Korea account '+json.dumps(tag)+f' Sharpe={summaries[-1]["sharpe"]:.3f}',flush=True)
    pd.concat(allnav).to_parquet(KR/'nav.parquet',index=False);pd.concat(alltargets).to_parquet(KR/'targets.parquet',index=False)
    pd.concat(alltrades).to_parquet(KR/'trades.parquet',index=False);pd.DataFrame(summaries).to_csv(KR/'summary.csv',index=False)
    individual=[];ic=[]
    for (study,model,a),g in p.groupby(['study','classifier','asset']):
        f=g.set_index('date').loc[dates];r=d['returns'][a].loc[dates];rf=d['rf'].loc[dates]
        rr=index_binary(f.p0,r,rf,.0005);on=f.p0>=.5;mu=np.where(on,f.mu0,np.minimum(f.mu1,-.001))
        wealth=np.r_[1,np.cumprod(1+rr)]
        individual.append({'study':study,'classifier':model,'asset':a,'sharpe':sharpe(rr,rf),'mdd':float((wealth/np.maximum.accumulate(wealth)-1).min())})
        corr=float(np.corrcoef(mu,d['excess'][a].loc[dates])[0,1]) if np.std(mu)>1e-12 else None
        ic.append({'study':study,'classifier':model,'asset':a,'ic':corr,'balanced_accuracy':float(np.mean([(on[f.truth==k]==(k==0)).mean() for k in [0,1] if (f.truth==k).any()])),
            'bull_excess_annual_arithmetic':float(d['excess'][a].loc[dates][on].mean()*252) if on.any() else None,
            'bear_excess_annual_arithmetic':float(d['excess'][a].loc[dates][~on].mean()*252) if (~on).any() else None})
    pd.DataFrame(individual).to_csv(KR/'individual.csv',index=False);pd.DataFrame(ic).to_csv(KR/'return-ic.csv',index=False)
    korea_bootstrap(pd.concat(allnav),d)

def korea_bootstrap(nav,d):
    main=nav.loc[(nav.study=='controlled')&(nav.cost_bp==5)&(nav.delay==0)&nav.cash_rule]
    tests=[];rng=np.random.default_rng(C['seed'])
    for kind in ['mv','minvar','ew']:
        g=main.loc[main.strategy.str.startswith(kind+'_')].pivot(index='date',columns='strategy',values='net_return')
        for other in ['lr','jm']:
            a=g[kind+'_xgb'].to_numpy();b=g[kind+'_'+other].to_numpy();n=len(a);gamma=C['bootstrap']['risk_aversion']
            def difference(x,y):return 252*((x.mean()-.5*gamma*x.var(ddof=1))-(y.mean()-.5*gamma*y.var(ddof=1)))
            actual=difference(a,b)
            for block in C['bootstrap']['block_lengths']:
                samples=[]
                for _ in range(C['bootstrap']['replications']):
                    starts=rng.integers(0,n,size=int(np.ceil(n/block)));idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
                    samples.append(difference(a[idx],b[idx]))
                tests.append({'allocation':kind,'comparison':'xgb-'+other,'metric':'annual_CER_difference','estimate':actual,
                    'block_sessions':block,'replications':len(samples),'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),
                    'scope':'paired conditional-on-fitted-forecasts uncertainty, not model-retraining or multiple-testing adjusted'})
    write_json(KR/'bootstrap.json',tests)

def korea_verify():
    from scipy.optimize import brentq
    d=korea_load();require(not d.get('synthetic_test_fixture',False),'Synthetic fixtures cannot validate a financial experiment')
    korea_check_inputs();nav=pd.read_parquet(KR/'nav.parquet');target=pd.read_parquet(KR/'targets.parquet');ledger=pd.read_parquet(KR/'trades.parquet')
    fits=json.loads((KR/'fits.json').read_text());tuning=json.loads((KR/'tuning.json').read_text())
    predictions=pd.read_parquet(KR/'predictions.parquet')
    require(len(fits)==25*N*len(C['jump_penalties']),'Incomplete model fits')
    require(all(x['last_train_target']<=x['asof'] for x in fits),'Training target leak')
    require(all(x['validation_last']<=x['asof'] for x in tuning),'Validation future-return leak')
    tags=['study','strategy','cost_bp','delay','cash_rule'];maximum=0.;count=0;days=0
    expected=d['cal'][(d['cal']>=C['evaluation_start'])&(d['cal']<=C['end'])]
    require(len(predictions)==len(expected)*N*5,'Missing model/asset predictions')
    require(np.isfinite(predictions.p0).all() and predictions.p0.between(0,1).all(),'Invalid probabilities')
    for _,g in predictions.groupby(['study','classifier','asset']):
        require(pd.DatetimeIndex(g.sort_values('date').date).equals(expected),'Model forecast calendar gap')
    controlled=predictions.loc[predictions.study=='controlled'].groupby(['date','asset'])
    require(controlled.classifier.nunique().eq(3).all(),'Missing shared-condition classifier')
    require(controlled[['penalty','mu0','mu1','halflife','asof','signal_date']].nunique().eq(1).all().all(),'Model comparison conditions differ')
    require(predictions.loc[predictions.study=='controlled','halflife'].eq(C['controlled_halflife']).all(),'Controlled smoothing changed')
    for key,g in target.groupby(tags):
        g=g.sort_values('date');dates=pd.DatetimeIndex(g.date);require(dates.equals(expected),'Evaluation calendar changed')
        require((g[ALL].min().min()>=-1e-8) and np.allclose(g[ALL].sum(axis=1),1),'Invalid target weights')
        require(g[ASSETS].max().max()<=C['weight_cap']+1e-7,'Weight cap exceeded')
        require(np.array_equal(g.signal_date.to_numpy(),d['cal'][d['cal'].get_indexer(dates)-1-key[3]].to_numpy()),'Signal delay mismatch')
        mask=np.ones(len(nav),dtype=bool);tm=np.ones(len(ledger),dtype=bool)
        for col,value in zip(tags,key):mask &=nav[col].to_numpy()==value;tm &=ledger[col].to_numpy()==value
        n=nav.loc[mask].set_index('date').loc[dates];t=ledger.loc[tm].set_index('date').loc[dates]
        holding=np.r_[np.zeros(N),1.];cost=key[2]/10000
        for i,(date,w) in enumerate(zip(dates,g[ALL].to_numpy())):
            before=holding.sum();value=before if cost==0 else brentq(lambda v:v+cost*np.abs((v*w-holding)[:-1]).sum()-before,before*(1-2*cost),before,xtol=1e-13)
            trade=value*w-holding;fee=cost*np.abs(trade[:-1]).sum();holding=value*w*(1+np.r_[d['returns'].loc[date].to_numpy(),d['rf'].loc[date]])
            maximum=max(maximum,abs(holding.sum()-n.nav.iloc[i]));require(np.max(np.abs(trade-t[ALL].iloc[i].to_numpy()))<1e-8,'Ledger mismatch')
            require(abs(fee-n.fee.iloc[i])<1e-9,'Fee mismatch');days+=1
        count+=1;print('Korea replay '+str(key),flush=True)
    require(count==31 and maximum<1e-8,'Independent replay failed/incomplete accounts')
    write_json(KR/'verification.json',{'status':'passed','accounts':count,'account_days':days,'fits':len(fits),'max_nav_error':maximum,
        'evaluation_months':len(pd.period_range(C['evaluation_start'],C['end'],freq='M')),'stock_total_return':False,'original_paper_replication':False})
`;
