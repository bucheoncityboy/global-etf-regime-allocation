import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { root } from './study.ts';
import { improvementRun, validateImprovement, type ImprovementConfig } from './korea_improve.ts';
import { type KoreaConfig } from './korea.ts';

const c=JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as KoreaConfig;
const q=JSON.parse(await readFile(resolve(root,'configs/korea-improvement.json'),'utf8')) as ImprovementConfig;
validateImprovement(q);assert.throws(()=>validateImprovement({...q,cost_bp:0}));
await improvementRun(c,q,String.raw`
import tempfile,copy
v={x['id']:x for x in Q['variants']}
p=np.array([0.,1.,.5]);hi=np.full(3,.004);lo=np.full(3,-.0001)
require(np.allclose(improve_mu(p,hi,lo,v['probability']),[-.001,.004,.0015]),'Probability mixture or bearish cap is wrong')
require(np.allclose(improve_mu(p,hi,lo,v['shrink']),np.array([-.001,.004,.0015])*.5),'Mean shrinkage is wrong')
require(np.allclose(improve_mu(p[:2],hi[:2],lo[:2],v['hard']),improve_mu(p[:2],hi[:2],lo[:2],v['probability'])),'Certain probabilities must match hard decisions')
previous=np.r_[.2,.2,np.zeros(N-2),.6];proposal=previous.copy();proposal[0]+=.005;proposal[1]-=.005
w,buffered,turnover=improve_buffer(proposal,previous,.02,False)
require(buffered and np.array_equal(w,previous) and abs(turnover-.01)<1e-12,'Buffer did not retain drifted holdings')
w,buffered,_=improve_buffer(np.r_[np.zeros(N),1.],previous,.99,True)
require(not buffered and w[-1]==1,'Buffer blocked forced liquidation')
overweight=np.r_[.41,np.zeros(N-1),.59];proposal=np.r_[.4,np.zeros(N-1),.6]
w,buffered,_=improve_buffer(proposal,overweight,.02,False)
require(not buffered and w[0]==.4,'Buffer retained an existing cap breach')
with tempfile.TemporaryDirectory(prefix='korea-improve-fixture-') as td:
    ROOT=Path(td);KR=ROOT/'results/korea';IR=ROOT/'results/korea_improvement';KR.mkdir(parents=True);IR.mkdir(parents=True)
    (ROOT/'data/private/korea').mkdir(parents=True)
    C=copy.deepcopy(C);Q=copy.deepcopy(Q);Q['end']='2018-07-06';Q['bootstrap']['replications']=3
    rng=np.random.default_rng(240609579);cal=pd.bdate_range('2001-01-02',Q['end'])
    returns=pd.DataFrame(rng.normal(.0002,.005,(len(cal),N)),index=cal,columns=ASSETS);rf=pd.Series(.02/252,index=cal)
    d={'cal':cal,'returns':returns,'rf':rf,'excess':returns.subtract(rf,axis=0),'synthetic_test_fixture':True}
    dates=cal[cal>='2018-07-01'];records=[]
    for i,date in enumerate(dates):
        for model in Q['classifiers']:
            for j,a in enumerate(ASSETS):
                prob=.2 if i==2 and j>1 else .7+.02*np.sin(i+j)
                records.append({'date':date,'asset':a,'classifier':model,'study':'controlled','p0':prob,'mu0':.001,'mu1':-.0004})
    predictions=pd.DataFrame(records);context=improve_context(d,predictions)
    changed=copy.deepcopy(d);changed['returns'].loc[dates[1]:]*=-10;changed['excess']=changed['returns'].subtract(rf,axis=0)
    pp=predictions.copy();pp.loc[pp.date>=dates[1],'p0']=.1
    future_context=improve_context(changed,pp);tables=[[],[],[]]
    for spec in improve_specs():
        frames=improve_account(d,context,spec);mutated=improve_account(changed,future_context,spec)
        require(np.allclose(frames[1].iloc[0][ALL].to_numpy(dtype=float),mutated[1].iloc[0][ALL].to_numpy(dtype=float)),'First allocation depends on future returns or probabilities')
        for collection,frame in zip(tables,frames):collection.append(frame)
    nav,target,ledger=[pd.concat(x,ignore_index=True) for x in tables]
    proof=improve_replay(d,nav,target,ledger);require(proof['account_days']==24*len(dates),'Missing fixture account paths')
    broken=ledger.copy();broken.loc[0,ASSETS[0]]+=.001
    try:improve_replay(d,nav,target,broken)
    except RuntimeError as e:require('Fee/trade replay mismatch' in str(e),'Wrong corruption guard')
    else:raise RuntimeError('Corrupted transaction ledger was accepted')
    old=[]
    for spec in improve_specs():
        if spec['variant']=='hard' or (spec['variant']=='baseline' and spec['delay']==0):
            f=nav.loc[nav.account==spec['account']].copy();f['study']='controlled' if spec['variant']=='hard' else 'baseline'
            f['strategy']='mv_'+spec['classifier'] if spec['variant']=='hard' else spec['classifier'];f['cash_rule']=True;old.append(f)
    pd.concat(old).to_parquet(KR/'nav.parquet',index=False)
    proof['original_reference']=improve_original_matches(nav);proof['synthetic_fixture']=True
    for name,frame in [('nav',nav),('targets',target),('trades',ledger)]:frame.to_parquet(IR/(name+'.parquet'),index=False)
    predictions.to_parquet(KR/'predictions.parquet',index=False)
    (ROOT/'data/private/korea/prepared.pkl').write_bytes(pickle.dumps(d))
    try:improve_verify()
    except RuntimeError as e:require('Synthetic fixtures' in str(e),'Wrong synthetic-input guard')
    else:raise RuntimeError('Synthetic financial verification accepted')
    write_json(IR/'verification.json',{'status':'passed',**proof});improve_statistics();improve_report()
    report=(IR/'README.md').read_text(encoding='utf-8')
    require(report.count('\n## ')==8 and '사후' in report and '추가1거래일' in report,'Incomplete report or scope')
    require((IR/'nav.png').exists() and json.loads((IR/'status.json').read_text())['status']=='synthetic_fixture_only','Fixture output mislabeled financial')
print('PASS: probability formula, shrinkage, drifted buffer, risk overrides, 24 paths, future invariance, independent replay, corruption rejection and eight-section report; temporary fixtures only')
`,true).then(output=>process.stdout.write(output));
