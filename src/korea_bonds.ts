import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { root, sha256 } from './study.ts';
interface Receipt {path:string;url:string;sha256:string;fetched_at_utc:string;bytes:number}
const raw=resolve(root,'data/private/korea/raw');
async function exists(p:string):Promise<boolean>{try{await access(p);return true;}catch{return false;}}
export async function acquireBonds():Promise<void>{
  await mkdir(raw,{recursive:true});await mkdir(resolve(raw,'kis'),{recursive:true});
  const c=JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as {raw_start:string;end:string};
  const receipts:Receipt[]=[];let cookie='';let minimumInterval=2600,lastRequestAt=0;
  async function save(name:string,url:string,kis=false):Promise<string>{
    const path=resolve(raw,name),meta=path+'.meta.json';
    if(await exists(path)&&await exists(meta)){const m=JSON.parse(await readFile(meta,'utf8')) as Receipt;const b=await readFile(path);
      assert.equal(m.url,url);assert.equal(sha256(b),m.sha256);receipts.push(m);return b.toString('utf8');}
    for(let attempt=0;attempt<8;attempt++)try{
      if(kis){const wait=minimumInterval-(Date.now()-lastRequestAt);if(wait>0)await new Promise<void>(r=>setTimeout(r,wait));lastRequestAt=Date.now();}
      const response=await fetch(url,{headers:kis?{cookie,Referer:'https://www.bond.co.kr/conts/10050','X-Requested-With':'XMLHttpRequest'}:{'User-Agent':'Mozilla/5.0'},signal:AbortSignal.timeout(30000)});
      if(response.status===429){
        const wait=Math.max(60,Number(response.headers.get('retry-after')??60));
        minimumInterval=Math.min(5000,minimumInterval*2);
        console.log(`KIS rate limit: respecting ${wait}s cooldown, slowing to ${minimumInterval}ms`);await new Promise<void>(r=>setTimeout(r,wait*1000));continue;
      }
      assert.equal(response.status,200);const bytes=Buffer.from(await response.arrayBuffer());const text=bytes.toString('utf8');
      if(kis){const j:unknown=JSON.parse(text);assert.ok(Array.isArray(j)&&j.length>=6,'KIS response missing indices');}
      const m={path:name,url,sha256:sha256(bytes),bytes:bytes.length,fetched_at_utc:new Date().toISOString()};
      await writeFile(path,bytes);await writeFile(meta,JSON.stringify(m,null,2)+'\n');receipts.push(m);return text;
    }catch(error){if(attempt===7)throw error;await new Promise<void>(r=>setTimeout(r,(attempt+1)*1000));}
    throw new Error('Unreachable');
  }
  let sessions:string[]=[];
  for(const symbol of ['KOSPI','KOSDAQ']){
    const url='https://api.finance.naver.com/siseJson.naver?'+new URLSearchParams({symbol,requestType:'1',startTime:'20000101',endTime:c.end.replaceAll('-',''),timeframe:'day'});
    const text=await save('naver-'+symbol+'.txt',url);
    const dates=[...text.matchAll(/^\["(\d{8})",/gm)].map(m=>m[1]).filter((d):d is string=>!!d&&d>=c.raw_start.replaceAll('-',''));
    assert.ok(dates.length>6000);if(symbol==='KOSPI')sessions=dates;else assert.deepEqual(dates,sessions,'KOSPI/KOSDAQ source sessions differ');
  }
  const home=await fetch('https://www.bond.co.kr/conts/10050');assert.equal(home.status,200);
  cookie=home.headers.getSetCookie().map(x=>x.split(';')[0]).join('; ');
  let next=0,done=0;
  await Promise.all(Array.from({length:1},async()=>{while(next<sessions.length){const date=sessions[next++];assert.ok(date);
    await save(`kis/${date}.json`,'https://www.bond.co.kr/kisnet/100?baseDate='+date,true);done++;
    if(done%100===0)console.log(`KIS full-calendar acquisition ${done}/${sessions.length}`);
  }}));
  receipts.sort((a,b)=>a.path.localeCompare(b.path));await mkdir(resolve(root,'results/korea'),{recursive:true});
  await writeFile(resolve(root,'results/korea/bond-source-manifest.json'),JSON.stringify({sources:receipts},null,2)+'\n');
  console.log(`KIS acquisition complete: ${sessions.length} stock-market sessions`);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))acquireBonds().catch((e:unknown)=>{console.error(e);process.exitCode=1;});
