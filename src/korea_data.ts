import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { root, sha256 } from './study.ts';

interface EcosSpec {table:string;item:string;name:string}
interface Config {raw_start:string;end:string;ecos:Record<string,EcosSpec>;series:Record<string,{ticker?:string;provider:string}>}
interface EcosRow {STAT_CODE:string;ITEM_CODE1:string;ITEM_NAME1:string;TIME:string;DATA_VALUE:string}
interface EcosResponse {StatisticSearch?:{list_total_count:number;row:EcosRow[]};RESULT?:{CODE:string;MESSAGE:string}}
interface Receipt {path:string;url:string;sha256:string;fetched_at_utc:string;bytes:number}
const directory=resolve(root,'data/private/korea/raw');
async function exists(path:string):Promise<boolean>{try{await access(path);return true;}catch{return false;}}
export async function acquireKorea():Promise<void>{
  const c=JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as Config;
  await mkdir(directory,{recursive:true});const receipts:Receipt[]=[];
  async function fetchSaved(name:string,url:string):Promise<unknown>{
    const path=resolve(directory,name),meta=path+'.meta.json';
    if(await exists(path)&&await exists(meta)){
      const r=JSON.parse(await readFile(meta,'utf8')) as Receipt;
      const bytes=await readFile(path);assert.equal(sha256(bytes),r.sha256);assert.equal(r.url,url);receipts.push(r);
      return JSON.parse(bytes.toString('utf8')) as unknown;
    }
    for(let attempt=0;attempt<4;attempt++){
      try{
        const response=await fetch(url,{headers:{'User-Agent':'Mozilla/5.0 (academic historical-data research)'},signal:AbortSignal.timeout(30000)});
        assert.equal(response.status,200,`HTTP ${response.status}: ${name}`);
        const bytes=Buffer.from(await response.arrayBuffer());const parsed:unknown=JSON.parse(bytes.toString('utf8'));
        assert.ok(!(parsed&&typeof parsed==='object'&&'RESULT' in parsed),'ECOS returned error; do not cache it');
        const r={path:name,url,sha256:sha256(bytes),fetched_at_utc:new Date().toISOString(),bytes:bytes.length};
        await writeFile(path,bytes);await writeFile(meta,JSON.stringify(r,null,2)+'\n');receipts.push(r);return parsed;
      }catch(e){if(attempt===3)throw e;await new Promise<void>(r=>setTimeout(r,500*(attempt+1)));}
    }throw new Error('Unreachable');
  }
  const bondUrl='https://datacenter.hankyung.com/data/datafile/stock/json/kisBondData.json';
  await fetchSaved('hankyung-kis.json',bondUrl);
  for(const symbol of ['^KS11','^KQ11'])for(let year=2000;year<=2024;year+=4){
    const start=Date.parse(`${year}-01-01`)/1000;
    const end=Date.parse(year===2024?'2025-10-01':`${year+4}-01-01`)/1000;
    const url='https://query1.finance.yahoo.com/v8/finance/chart/'+encodeURIComponent(symbol)+'?'+new URLSearchParams({period1:String(start),period2:String(end),interval:'1d',events:'div,splits',includeAdjustedClose:'true'});
    await fetchSaved(`${symbol.slice(1)}_${year}.json`,url);
  }
  const compact=(s:string):string=>s.replaceAll('-','');
  const tasks:Array<()=>Promise<void>>=[];
  for(const [name,spec] of Object.entries(c.ecos)){
    const base=`https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/`;
    const tail=`/${spec.table}/D/${compact(c.raw_start)}/${compact(c.end)}/${spec.item}`;
    const first=await fetchSaved(`${name}_1.json`,base+'1/10'+tail) as EcosResponse;
    assert.ok(first.StatisticSearch);const total=first.StatisticSearch.list_total_count;
    for(let start=11;start<=total;start+=10){const from=start;
      tasks.push(async()=>{await fetchSaved(`${name}_${from}.json`,base+`${from}/${Math.min(from+9,total)}`+tail);});
    }
    console.log(`ECOS ${name}: ${total} observations, sample API pages of at most 10`);
  }
  let next=0,done=0;
  await Promise.all(Array.from({length:3},async()=>{while(next<tasks.length){const task=tasks[next++];assert.ok(task);await task();done++;
    if(done%100===0)console.log(`ECOS acquisition ${done}/${tasks.length}`);
    await new Promise<void>(r=>setTimeout(r,100));
  }}));
  receipts.sort((a,b)=>a.path.localeCompare(b.path));
  await mkdir(resolve(root,'results/korea'),{recursive:true});
  await writeFile(resolve(root,'results/korea/source-manifest.json'),JSON.stringify({sources:receipts,
    note:'Raw responses are retained locally under data/private/korea. End-date filtering precedes preparation; the KIS chart also contains later observations.'},null,2)+'\n');
  console.log(`Acquisition complete: ${receipts.length} immutable cached responses`);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))acquireKorea().catch((e:unknown)=>{console.error(e);process.exitCode=1;});
