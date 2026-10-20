import assert from 'node:assert/strict';
import { run, sha256 } from './study.ts';
assert.equal(sha256('abc'),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
await run(`
h=np.array([0.,1.]);w=np.array([1.,0.]);c=.0005
after,fee,trade=settle(h,w,c)
require(abs(after.sum()-(1-c)/(1+c))<1e-12,'Full switch charges two legs')
require(abs(after.sum()+fee-1)<1e-12,'Fees conserve NAV')
rng=np.random.default_rng(42)
for _ in range(100):
    h=rng.dirichlet(np.ones(13));w=rng.dirichlet(np.ones(13))
    out,fee,trade=settle(h,w,.0025)
    require(abs(out.sum()+fee-h.sum())<1e-11,'Self financing')
    require(np.allclose(out/out.sum(),w),'Post-cost target weights')
raw=np.array([.9,.1,.8,.2]);prefix=smooth(raw,4)[:2]
require(np.array_equal(prefix,smooth(raw[:2],4)),'Smoothing future dependence')
over=np.array([[1.,1.],[1.1,1.01],[.95,1.]])
day=np.array([[1.02,1.],[1.,1.],[.98,1.001]])
p=np.array([.9,.1,.9]);r=binary_returns(p,over,day,c)
h=np.array([0.,1.]);actual=[]
for i in range(3):
    prior=h.sum();at=h*over[i];target=np.array([1.,0.]) if p[i]>=.5 else np.array([0.,1.])
    after,_,_=settle(at,target,c);h=after*day[i];actual.append(h.sum()/prior-1)
require(np.allclose(r,actual,atol=1e-12),'Validation and account disagree across overnight switch')
dates=pd.bdate_range('2000-01-01',periods=400)
x=pd.DataFrame(rng.normal(0,.01,(400,12)),index=dates)
cov=covariance_series(x)[dates[-1]]
weights=2.**(-np.arange(399,-1,-1)/252);weights/=weights.sum()
centered=x.to_numpy()-weights@x.to_numpy()
expected=(centered.T*weights)@centered/(1-weights@weights)
require(np.allclose(cov,expected,atol=1e-15),'Covariance recurrence and weighted reference disagree')
require(np.array_equal(covariance_series(x.iloc[:300])[dates[299]],covariance_series(x)[dates[299]]),'Covariance future dependence')
opt=Optimizer(10,1,5)
cash=opt.solve(np.full(12,-.001),cov,np.zeros(12))
require(cash[-1]==1 and np.count_nonzero(cash[:12])==0,'Cash optimum')
w=opt.solve(np.full(12,.01),np.eye(12)*.0001,np.zeros(12))
require(np.all(w>=0) and w[:12].max()<=.4+1e-8 and abs(w.sum()-1)<1e-8,'12-asset optimizer feasibility')
require(np.allclose(w[:12],np.full(12,1/12),atol=1e-6),'Symmetric optimizer solution')
print('PASS: hash, two-leg costs, 13-asset self-financing, causal smoothing/covariance, next-open accounting, covariance reference, cash optimum, 12-asset optimizer')
if (ROOT/'data/prepared.pkl').exists():
    import tempfile, shutil
    source_root=ROOT;data=load_data();job=json.loads((ROOT/'results/jobs.json').read_text())[0]
    with tempfile.TemporaryDirectory(prefix='etf-future-isolation-') as temporary:
        ROOT=Path(temporary)
        for relative in ['configs/protocol.json','src/kernel.ts','uv.lock']:
            target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_root/relative,target)
        fit_job(job,data)
        path=next((ROOT/'results/cache').rglob('*.pkl'));baseline=pickle.loads(path.read_bytes());path.unlink()
        altered=dict(data);altered['features']=dict(data['features']);asset=job['asset']
        altered['features'][asset]=data['features'][asset].copy()
        altered['features'][asset].loc[lambda f:f.index>pd.Timestamp(job['asof'])]*=-1000000
        altered['excess']=data['excess'].copy()
        altered['excess'].loc[lambda f:f.index>pd.Timestamp(job['asof'])]*=-1000000
        fit_job(job,altered);changed=pickle.loads(path.read_bytes())
        for col in ['lr','xgb','jm','prior','mu0','mu1']:
            left=baseline['predictions'].loc[lambda f:f.date==pd.Timestamp(job['first']),col].to_numpy()
            right=changed['predictions'].loc[lambda f:f.date==pd.Timestamp(job['first']),col].to_numpy()
            require(len(left)==len(C['jump_penalties']),'Missing initial forecast in mutation test')
            require(np.array_equal(left,right),f'Future mutation changed initial {col}')
        require([f['train_labels_sha256'] for f in baseline['fits']]==[f['train_labels_sha256'] for f in changed['fits']], 'Future mutation changed training labels')
    ROOT=source_root
    print('PASS: actual JM/LR/XGB training and first forecast unchanged after future-feature/return mutation')
`);
