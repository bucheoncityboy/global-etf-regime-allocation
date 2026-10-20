/** Scientific-library adapter. The runner sends source directly to a locked Python runtime.
 * No external shell interpolation, generated executable files, or legacy-repo imports. */
export const common = String.raw`
import sys, os, json, hashlib, pickle, zipfile, warnings, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore', message='The default value of raise_error')
ROOT = Path.cwd()
C = json.loads((ROOT/'configs/protocol.json').read_text())
ASSETS = C['assets']; ALL = ASSETS + [C['cash']]
RC = ['log_dd_5','log_dd_21','mean_5','mean_10','mean_21','sortino_5','sortino_10','sortino_21']
MC = ['yield2_change','slope','slope_change','vix_change','stock_bond_corr']
def require(ok, message):
    if not ok: raise RuntimeError(message)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(x):
    if isinstance(x, (pd.Timestamp, Path)): return str(x)
    if isinstance(x, (np.integer,np.floating)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)
def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,default=encode,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def frame_hash(x):
    return hashlib.sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()).hexdigest()
def load_data(): return pickle.loads((ROOT/'data/prepared.pkl').read_bytes())
def smooth(x,h):
    x=np.asarray(x,dtype=float)
    if h==0: return x.copy()
    return pd.Series(x).ewm(halflife=h,adjust=False).mean().to_numpy()
def settle(holding,w,c):
    require(np.isfinite(w).all() and w.min()>=-1e-9 and abs(w.sum()-1)<1e-8,'Invalid allocation')
    total=holding.sum()
    if c==0: return total*w,0.,total*w-holding
    lo,hi=total*(1-2*c),total
    for _ in range(48):
        mid=(lo+hi)/2
        if mid+c*np.abs(mid*w-holding).sum()>total: hi=mid
        else: lo=mid
    nav=(lo+hi)/2; trade=nav*w-holding; fee=c*np.abs(trade).sum()
    require(abs(nav+fee-total)<1e-11*max(1,total),'Self-financing failure')
    return nav*w,fee,trade
def binary_returns(p,overnight,intraday,c):
    risk=np.asarray(p)>=.5; held=np.r_[False,risk[:-1]]
    over=np.where(held,overnight[:,0],overnight[:,1])
    day=np.where(risk,intraday[:,0],intraday[:,1])
    return over*day*np.where(risk!=held,(1-c)/(1+c),1)-1
def sharpe(r,cash):
    x=np.asarray(r)-np.asarray(cash); sd=x.std(ddof=1)
    return float(x.mean()/sd*np.sqrt(252)) if sd>1e-12 else 0.
`;

export const preparation = String.raw`
def prepare():
    archive=ROOT/'data/input_snapshot.zip'
    require(digest(archive)==C['input_snapshot_sha256'],'Input archive changed')
    raw=ROOT/'data/raw'; raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            relative=Path(*Path(member.filename).parts[1:])
            if not relative.parts or member.is_dir(): continue
            target=(raw/relative).resolve()
            require(target.is_relative_to(raw.resolve()),'Unsafe archive path')
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(member))
    objects=[]
    for meta in sorted(raw.rglob('*.meta.json')):
        m=json.loads(meta.read_text()); path=Path(str(meta)[:-10])
        require(digest(path)==m['sha256'],f'Raw hash changed: {path}')
        objects.append({'path':path.relative_to(ROOT).as_posix(),'sha256':m['sha256'],
                        'url':m['url'],'fetched_at_utc':m.get('fetched_at_utc',m.get('downloaded_at_utc'))})
    cal=pd.DatetimeIndex(json.loads((raw/'calendar.json').read_text())['sessions'])
    cal=cal[(cal>='1991-01-01')&(cal<=C['end'])]
    prices={}; audit=[]
    for asset in ALL:
        chunks=[]; events={}
        for path in sorted((raw/'yahoo').glob(asset+'_*.json')):
            if path.name.endswith('.meta.json'): continue
            payload=json.loads(path.read_text()); require(not payload['chart']['error'],str(path))
            row=payload['chart']['result'][0]
            require(row['meta']['dataGranularity']=='1d','Non-daily quote')
            if not row.get('timestamp'): continue
            idx=pd.to_datetime(row['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
            q=row['indicators']['quote'][0]
            chunks.append(pd.DataFrame({'open':q['open'],'close':q['close'],
                'adj_close':row['indicators']['adjclose'][0]['adjclose']},index=idx))
            for kind,values in row.get('events',{}).items():
                for key,value in values.items(): events[(kind,key)]=value
        p=pd.concat(chunks).sort_index().loc[:C['end']]
        require(not p.index.has_duplicates and not p.isna().any().any() and (p>0).all().all(),f'Invalid quotes {asset}')
        expected=cal[cal>=p.index.min()]
        require(expected.equals(p.index),f'Calendar gap {asset}')
        p['adj_open']=p.open*p.adj_close/p.close
        div=pd.Series(0.,index=p.index)
        for (kind,key),v in events.items():
            if kind=='dividends':
                date=pd.Timestamp(v['date'],unit='s',tz='UTC').tz_convert('America/New_York').tz_localize(None).normalize()
                if date in div.index: div.loc[date]+=v['amount']
        err=((p.close+div)/p.close.shift()-1-p.adj_close.pct_change(fill_method=None)).abs().max()
        require(err<.003,f'Distribution mismatch {asset}')
        prices[asset]=p
        audit.append({'asset':asset,'first':str(p.index.min().date()),'last':str(p.index.max().date()),
                      'rows':len(p),'missing_sessions':0,'max_dividend_reconciliation_error':err})
    close=pd.DataFrame({a:p.adj_close for a,p in prices.items()}).reindex(cal)
    opens=pd.DataFrame({a:p.adj_open for a,p in prices.items()}).reindex(cal)
    returns=close.pct_change(fill_method=None)
    known=pd.DataFrame(index=cal); max_age={}
    for name in ['DGS2','T10Y2Y','VIXCLS','DGS3MO']:
        f=pd.read_csv(raw/'fred'/f'{name}.csv',index_col=0,parse_dates=True)
        values=pd.to_numeric(f[name],errors='coerce').dropna()
        positions=values.index.searchsorted(cal,side='left')-1
        require((positions>=0).all(),'Missing FRED history')
        age=(cal-values.index[positions]).days; require(age.max()<=10,'Stale FRED')
        known[name]=values.iloc[positions].to_numpy(); max_age[name]=int(age.max())
    m=pd.DataFrame(index=cal)
    m[MC[0]]=known.DGS2.diff().ewm(halflife=21,adjust=False).mean()
    m[MC[1]]=known.T10Y2Y.ewm(halflife=10,adjust=False).mean()
    m[MC[2]]=known.T10Y2Y.diff().ewm(halflife=21,adjust=False).mean()
    m[MC[3]]=np.log(known.VIXCLS).diff().ewm(halflife=63,adjust=False).mean()
    m[MC[4]]=returns.SPY.rolling(252).corr(returns.AGG)
    excess=returns[ASSETS].subtract(known.DGS3MO/100/252,axis=0)
    features={}
    for a in ASSETS:
        x=excess[a]; f=pd.DataFrame(index=cal)
        means={h:x.ewm(halflife=h,adjust=False,min_periods=63).mean() for h in [5,10,21]}
        dd={h:np.sqrt(x.clip(upper=0).pow(2).ewm(halflife=h,adjust=False,min_periods=63).mean()) for h in [5,10,21]}
        for h in [5,21]: f[f'log_dd_{h}']=np.log(dd[h].clip(lower=1e-12))
        for h in [5,10,21]: f[f'mean_{h}']=means[h]
        for h in [5,10,21]: f[f'sortino_{h}']=means[h]/dd[h].clip(lower=1e-12)
        features[a]=pd.concat([f,m],axis=1).dropna()
        require(np.isfinite(features[a]).all().all(),'Nonfinite feature')
    jobs=[]; boundaries={}
    for profile in C['profiles']:
        folds=[]
        for boundary in pd.date_range(profile['validation_start'][:4]+'-01-01',C['end'],freq='6MS'):
            targets=cal[(cal>=boundary)&(cal<boundary+pd.DateOffset(months=6))]
            asof=cal[cal.get_loc(targets[0])-1]; start=asof-pd.DateOffset(years=profile['train_years'])
            require(all(f.index.min()<=start for f in features.values()),'Incomplete training history')
            fold={'boundary':str(boundary.date()),'asof':str(asof.date()),'first':str(targets[0].date()),'last':str(targets[-1].date())}
            folds.append(fold)
            for a in ASSETS: jobs.append({'profile':profile['id'],'asset':a,**fold})
        require(folds[0]['first']==profile['validation_start'],'Validation mismatch')
        evalfolds=[f for f in folds if f['first']>=profile['start']]
        require(evalfolds[0]['first']==profile['start'],'Evaluation mismatch')
        require(len(pd.period_range(profile['start'],C['end'],freq='M'))==profile['months'],'Month mismatch')
        boundaries[profile['id']]=folds
    payload={'cal':cal,'close':close[ALL],'open':opens[ALL],'returns':returns[ALL],
             'excess':excess,'features':features,'folds':boundaries}
    (ROOT/'data/prepared.pkl').write_bytes(pickle.dumps(payload,protocol=5))
    write_json(ROOT/'data/manifest.json',{'snapshot_sha256':digest(archive),'objects':objects,
        'calendar_sha256':digest(raw/'calendar.json'),'prices':audit,'fred_max_age_days':max_age,
        'first_features':{a:str(f.index.min().date()) for a,f in features.items()}})
    write_json(ROOT/'results/jobs.json',jobs)
    print(json.dumps({'jobs':len(jobs),'prices':len(audit),'first_common_features':str(max(f.index.min() for f in features.values()).date())}),flush=True)
`;

export const fitting = String.raw`
def fit_job(job,d):
    from jumpmodels.jump import JumpModel
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    profile=next(p for p in C['profiles'] if p['id']==job['profile'])
    a=job['asset']; asof=pd.Timestamp(job['asof']); start=asof-pd.DateOffset(years=profile['train_years'])
    f=d['features'][a]; train=f.loc[start:asof]; cal=d['cal']
    targets=cal[(cal>=job['first'])&(cal<=job['last'])]; signals=cal[cal.get_indexer(targets)-1]
    require(len(train)>profile['train_years']*240,'Insufficient training rows')
    signature=hashlib.sha256((digest(ROOT/'src/kernel.ts')+digest(ROOT/'configs/protocol.json')+digest(ROOT/'uv.lock')+C['input_snapshot_sha256']).encode()).hexdigest()[:20]
    path=ROOT/'results/cache'/signature/job['profile']/f"{a}_{job['asof']}.pkl"
    if path.exists(): return 'cached'
    cols=[c for c in RC if not(c.startswith('log_dd') and a in C['jump_exclude_dd'])]
    scale=StandardScaler().fit(train[cols]); z=pd.DataFrame(scale.transform(train[cols]),index=train.index,columns=cols)
    rows=[]; metas=[]
    with threadpool_limits(limits=1):
        for penalty in C['jump_penalties']:
            jm=JumpModel(n_components=2,jump_penalty=penalty,n_init=C['jump_n_init'],random_state=C['seed'])
            jm.fit(z,ret_ser=d['excess'][a].reindex(train.index),sort_by='cumret')
            labels=pd.Series(np.asarray(jm.labels_,dtype=int),index=train.index)
            online_x=f.loc[start:targets[-1],cols]
            zz=pd.DataFrame(scale.transform(online_x),index=online_x.index,columns=cols)
            online=pd.Series(jm.predict_online(zz),index=zz.index)
            require(np.array_equal(jm.predict_online(zz.loc[:asof]),online.loc[:asof]),'Future-dependent online JM')
            means=d['excess'][a].reindex(train.index).groupby(labels).mean().reindex([0,1])
            counts=labels.value_counts().reindex([0,1],fill_value=0)
            means=means.fillna(d['excess'][a].reindex(train.index).mean())
            next_dates=cal[cal.get_indexer(train.index)+1]
            y=labels.reindex(next_dates).to_numpy(); valid=(next_dates<=asof)&np.isfinite(y)
            require(next_dates[valid].max()<=asof and not valid[-1],'Shifted training target leakage')
            y=y[valid].astype(int); x=train.loc[valid,RC+MC]
            prior=(np.bincount(y,minlength=2)+1)/(len(y)+2)
            pred={'prior':np.full(len(targets),prior[0]),'jm':1-online.reindex(signals).to_numpy(dtype=int)}
            sc=StandardScaler().fit(x)
            xx=sc.transform(x); future=sc.transform(f.loc[signals,RC+MC])
            for name in ['lr','xgb']:
                if len(np.unique(y))==1: p=np.full(len(targets),prior[0])
                else:
                    model=(LogisticRegression(**C['logistic'],solver='lbfgs',random_state=C['seed']) if name=='lr'
                        else XGBClassifier(**C['xgb'],random_state=C['seed'],objective='binary:logistic',eval_metric='logloss'))
                    model.fit(xx,y); p=model.predict_proba(future)[:,0]
                require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'Invalid probability')
                pred[name]=p
            meta={'profile':profile['id'],'asset':a,'asof':asof,'penalty':penalty,
                'train_start':train.index.min(),'train_end':train.index.max(),'max_training_target':next_dates[valid].max(),
                'n_train':len(y),'counts':counts.to_numpy(),'means':means.to_numpy(),
                'train_labels_sha256':frame_hash(labels),'training_features_sha256':frame_hash(x),
                'scaler_mean':sc.mean_,'single_class':len(np.unique(y))==1}
            metas.append(meta)
            for i,date in enumerate(targets):
                rows.append({'date':date,'signal_date':signals[i],'asof':asof,'penalty':penalty,
                    'mu0':means.iloc[0],'mu1':means.iloc[1],'truth':int(online.loc[date]),
                    **{name:float(p[i]) for name,p in pred.items()}})
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(pickle.dumps({'predictions':pd.DataFrame(rows),'fits':metas},protocol=5))
    return 'fit'
def worker():
    d=load_data()
    for line in sys.stdin:
        job=json.loads(line); status=fit_job(job,d)
        print(json.dumps({'done':job,'status':status}),flush=True)
`;
