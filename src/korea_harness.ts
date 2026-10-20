import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { root } from './study.ts';
import { koreaRun, validateKorea, type KoreaConfig } from './korea.ts';
const config=JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as KoreaConfig;
validateKorea(config);assert.throws(()=>validateKorea({...config,train_years:5}));
assert.equal(config.assets.length,8);assert.equal((config['minimum_bullish_assets'] as number)-1,config.assets.length/4);
await koreaRun(config,String.raw`
import tempfile,copy
from scipy.optimize import brentq
# The 2017 Chuseok closure spans 11 calendar days between observed sessions.
holiday_cal=pd.to_datetime(['2017-09-28','2017-09-29','2017-10-10','2017-10-11'])
rates=pd.Series([2.1,2.2,2.3,2.4],index=holiday_cal)
lagged,availability=korea_prior_observations(rates,holiday_cal,'holiday-fixture')
require(lagged.loc['2017-10-10']==2.2,'Closure handling used a same-day or invented observation')
require(availability['max_age_calendar_days']==11 and len(availability['verified_exchange_closures'])==1,'Closure exception was not audited')
mutated=rates.copy();mutated.loc['2017-10-10':]+=100
require(korea_prior_observations(mutated,holiday_cal,'future-fixture')[0].loc['2017-10-10']==2.2,'Rate alignment used current/future information')
try:korea_prior_observations(rates.iloc[:2],pd.bdate_range('2017-09-28','2017-10-11'),'stale-fixture')
except RuntimeError as e:require('Stale ECOS observation' in str(e),'Wrong stale-rate guard')
else:raise RuntimeError('Missing observations across open sessions accepted as a holiday')
with tempfile.TemporaryDirectory(prefix='korea-unit-fixture-') as td:
    ROOT=Path(td);KR=ROOT/'results/korea';KR.mkdir(parents=True)
    (ROOT/'data/private/korea').mkdir(parents=True)
    C=copy.deepcopy(C);C['jump_penalties']=[0.,10.];C['jump_n_init']=2;C['bootstrap']['replications']=3
    rng=np.random.default_rng(240609578);cal=pd.bdate_range('2001-01-02','2025-09-30')
    returns=pd.DataFrame(rng.normal(.0001,.01,(len(cal),N)),index=cal,columns=ASSETS)
    rf=pd.Series(.02/252,index=cal);excess=returns.subtract(rf,axis=0)
    features={a:pd.DataFrame(rng.normal(size=(len(cal),13)),index=cal,columns=RC+MC) for a in ASSETS}
    d={'cal':cal,'returns':returns,'rf':rf,'excess':excess,'features':features,'synthetic_test_fixture':True}
    job={'asset':ASSETS[0],'boundary':'2013-07-01','asof':'2013-06-28','train_start':'2002-07-01','first':'2013-07-01','last':'2013-07-03'}
    korea_fit(job,d,ROOT/'original')
    changed=pickle.loads(pickle.dumps(d));a=ASSETS[0]
    changed['features'][a].loc['2013-07-02':]+=100
    changed['excess'].loc['2013-07-02':]*=-10
    korea_fit(job,changed,ROOT/'future-change')
    filename=f"{a}_{job['asof']}.pkl";one=pickle.loads((ROOT/'original'/filename).read_bytes());two=pickle.loads((ROOT/'future-change'/filename).read_bytes())
    require([x['label_hash'] for x in one['fits']]==[x['label_hash'] for x in two['fits']],'Future data changed training labels')
    for col in ['xgb','lr','jm','mu0','mu1']:
        require(np.allclose(one['predictions'].loc[one['predictions'].date=='2013-07-01',col],two['predictions'].loc[two['predictions'].date=='2013-07-01',col]),'First forecast depends on future '+col)
    require(all(x['last_train_target']<=pd.Timestamp(job['asof']) for x in one['fits']),'Training targets exceed cutoff')
    # Cover every rolling validation boundary with independent synthetic candidate forecasts.
    jobs=[];directory=ROOT/'candidates';directory.mkdir()
    for boundary in pd.date_range(C['validation_start'],C['end'],freq='6MS'):
        dates=cal[(cal>=boundary)&(cal<boundary+pd.DateOffset(months=6))];asof=cal[cal.get_loc(dates[0])-1]
        for asset in ASSETS:
            j={'asset':asset,'boundary':str(boundary.date()),'asof':str(asof.date()),'first':str(dates[0].date()),'last':str(dates[-1].date())};jobs.append(j)
            records=[]
            for penalty in C['jump_penalties']:
                for k,date in enumerate(dates):
                    wave=np.sin(k/7+penalty);records.append({'date':date,'signal_date':cal[cal.get_loc(date)-1],'asof':asof,'penalty':penalty,
                        'xgb':.5+.25*wave,'lr':.5+.2*wave,'jm':float(wave>0),'truth':int(wave<0),'mu0':.0004,'mu1':-.0002})
            (directory/f'{asset}_{asof.date()}.pkl').write_bytes(pickle.dumps({'predictions':pd.DataFrame(records),'fits':[]},protocol=5))
    d['jobs']=jobs;require(len(jobs)==25*N,'Fixture schedule mismatch')
    selected=korea_select(d,directory);before=json.loads((KR/'tuning.json').read_text())
    changed=pickle.loads(pickle.dumps(d));changed['returns'].loc['2020-01-01':]*=-50
    korea_select(changed,directory);after=json.loads((KR/'tuning.json').read_text())
    require([x for x in before if x['asof']<'2020']==[x for x in after if x['asof']<'2020'],'Future OOS returns changed historical tuning')
    require(all(x['validation_last']<=x['asof'] for x in before),'Validation boundary leak')
    # Test all account paths on five sessions; every temporary output is removed with this directory.
    C['end']='2018-07-06';dates=cal[(cal>=C['evaluation_start'])&(cal<=C['end'])]
    pp=selected.loc[selected.date.isin(dates)]
    korea_accounts(d,pp)
    nav=pd.read_parquet(KR/'nav.parquet');target=pd.read_parquet(KR/'targets.parquet');trades=pd.read_parquet(KR/'trades.parquet')
    tags=['study','strategy','cost_bp','delay','cash_rule'];accounts=0
    for key,g in target.groupby(tags):
        money=np.r_[np.zeros(N),1.];g=g.sort_values('date');mask=np.ones(len(nav),dtype=bool);tm=np.ones(len(trades),dtype=bool)
        for column,value in zip(tags,key):mask &=nav[column].to_numpy()==value;tm &=trades[column].to_numpy()==value
        n=nav.loc[mask].set_index('date');t=trades.loc[tm].set_index('date');cost=key[2]/10000
        for row in g.itertuples(index=False):
            w=np.array([getattr(row,a) for a in ALL]);before_nav=money.sum()
            value=before_nav if cost==0 else brentq(lambda v:v+cost*np.abs((v*w-money)[:-1]).sum()-before_nav,before_nav*(1-2*cost),before_nav,xtol=1e-13)
            require(np.allclose(value*w-money,t.loc[row.date,ALL].to_numpy(dtype=float),atol=1e-10),'Independent fixture trade replay mismatch')
            money=value*w*(1+np.r_[d['returns'].loc[row.date],d['rf'].loc[row.date]])
            require(abs(money.sum()-n.loc[row.date,'nav'])<1e-10,'Independent fixture NAV replay mismatch')
        accounts+=1
    require(accounts==31,'Missing allocation/cost/delay/cash-rule paths')
    cash=nav.loc[(nav.study=='baseline')&(nav.strategy=='cash')]
    require(np.allclose(cash.net_return,d['rf'].loc[dates]),'Cash account earned risky-asset returns or fees')
    (ROOT/'data/private/korea/prepared.pkl').write_bytes(pickle.dumps(d))
    try:korea_verify()
    except RuntimeError as e:require('Synthetic fixtures' in str(e),'Wrong synthetic-input guard')
    else:raise RuntimeError('Synthetic fixture accepted as financial verification')
    write_json(KR/'verification.json',{'accounts':31,'account_days':31*len(dates),'max_nav_error':0.})
    write_json(KR/'input-receipt.json',{'sessions':len(cal)})
    korea_report();require((KR/'README.md').exists() and (KR/'nav.png').exists(),'Report generator failed')
print('PASS: causal fits, rolling validation, 31 allocation paths, independent fee/NAV replay, report generation, synthetic rejection')
`,true);
console.log('Korea harness passed: future-data invariance, 11/5 schedule, 31 account paths, independent NAV replay, report generation. All fixtures were temporary; no Korean financial performance was generated.');
