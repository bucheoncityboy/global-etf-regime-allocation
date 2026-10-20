import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { root, sha256 } from './study.ts';
import { common } from './kernel.ts';
import { allocation } from './portfolio.ts';
import { koreaNumerics } from './korea_kernel.ts';
import { koreaReporting } from './korea_report.ts';
import { acquireKorea } from './korea_data.ts';
import { acquireBonds } from './korea_bonds.ts';
export interface KoreaConfig {
  id:string;raw_start:string;train_start:string;validation_start:string;evaluation_start:string;end:string;
  train_years:number;validation_years:number;refit_months:number;assets:string[];cash:string;
  jump_penalties:number[];jump_n_init:number;[key:string]:unknown;
}
export interface KoreaJob {asset:string;boundary:string;asof:string;train_start:string;first:string;last:string}
export function validateKorea(c:KoreaConfig):void{
  assert.equal(c.train_years,11);assert.equal(c.validation_years,5);assert.equal(c.refit_months,6);
  assert.equal(c.assets.length,8);assert.equal(new Set(c.assets).size,8);
  assert.equal(c.train_start,'2002-07-01');assert.equal(c.validation_start,'2013-07-01');
  assert.equal(c.evaluation_start,'2018-07-01');assert.equal(c.end,'2025-09-30');
  assert.deepEqual([...c.jump_penalties].sort((a,b)=>a-b),c.jump_penalties);
}
export function koreaProcess(c:KoreaConfig,action:string,capture=false){
  const code=common+allocation+koreaNumerics+koreaReporting+'\nset_korea(json.loads('+JSON.stringify(JSON.stringify(c))+'))\n'+action+'\n';
  const child=spawn('uv',['run','--frozen','python','-u','-c',"import sys; exec(compile(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())), '<korea-adapter>', 'exec'))"],
    {cwd:root,env:{...process.env,PYTHONUTF8:'1',OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1'},stdio:capture?['pipe','pipe','inherit']:['pipe','inherit','inherit']});
  child.stdin?.write(`${Buffer.byteLength(code,'utf8')}\n`);child.stdin?.write(code);return child;
}
export async function koreaRun(c:KoreaConfig,action:string,capture=false):Promise<string>{
  return new Promise<string>((ok,fail)=>{const child=koreaProcess(c,action,capture);let output='';
    child.stdout?.on('data',(b:Buffer)=>{output+=b.toString('utf8');});child.on('error',fail);
    child.on('exit',code=>code===0?ok(output):fail(new Error(`Korea numerical adapter exited ${code}`)));child.stdin?.end();});
}
const sources=['src/korea.ts','src/korea_kernel.ts','src/korea_data.ts','src/korea_bonds.ts','src/korea_harness.ts',
  'src/korea_report.ts','src/korea_pipeline.ts','src/kernel.ts','src/portfolio.ts','configs/korea.json','uv.lock','pyproject.toml','.python-version','package-lock.json'];
async function fingerprint():Promise<Record<string,string>>{
  return Object.fromEntries(await Promise.all(sources.map(async p=>[p,sha256(await readFile(resolve(root,p)))])));
}
async function seal():Promise<void>{
  const files=await fingerprint();const input=JSON.parse(await readFile(resolve(root,'results/korea/input-receipt.json'),'utf8')) as unknown;
  await writeFile(resolve(root,'results/korea/design-freeze.json'),JSON.stringify({sources:files,input},null,2)+'\n');
}
async function check():Promise<void>{
  const f=JSON.parse(await readFile(resolve(root,'results/korea/design-freeze.json'),'utf8')) as {sources:Record<string,string>};
  assert.deepEqual(await fingerprint(),f.sources,'Korean experiment sources changed; start at prepare');
  await koreaRun(JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as KoreaConfig,'korea_check_inputs()');
}
async function fit(c:KoreaConfig):Promise<void>{
  await check();const jobs=JSON.parse(await readFile(resolve(root,'results/korea/jobs.json'),'utf8')) as KoreaJob[];
  const children:ReturnType<typeof koreaProcess>[]=[];let next=0,done=0;
  const workers=Number(process.env.STUDY_WORKERS??'4');assert.ok(Number.isInteger(workers)&&workers>=1&&workers<=16);
  try{await Promise.all(Array.from({length:workers},()=>new Promise<void>((ok,fail)=>{
    const child=koreaProcess(c,'korea_worker()',true);children.push(child);
    const send=():void=>{const job=jobs[next++];if(job)child.stdin?.write(JSON.stringify(job)+'\n');else child.stdin?.end();};
    assert.ok(child.stdout);createInterface({input:child.stdout}).on('line',line=>{try{
      const result=JSON.parse(line) as {done:KoreaJob;status:string};assert.ok(result.done);done++;
      console.log(`[${done}/${jobs.length}] ${result.done.asset} ${result.done.asof} ${result.status}`);send();
    }catch(e){fail(e);child.kill();}});child.on('error',fail);child.on('exit',code=>code===0?ok():fail(new Error(`Korea worker exited ${code}`)));send();
  })));}catch(e){for(const child of children)child.kill();throw e;}
}
async function main():Promise<void>{
  const c=JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as KoreaConfig;validateKorea(c);
  const command=process.argv[2]??'all';assert.ok(['acquire','prepare','fit','allocate','verify','all'].includes(command));
  await mkdir(resolve(root,'results/korea'),{recursive:true});
  if(command==='acquire'){await acquireKorea();await acquireBonds();return;}
  if(command==='prepare'||command==='all'){await koreaRun(c,'korea_prepare()');await seal();}
  if(command==='fit'||command==='all')await fit(c);
  if(command==='allocate'||command==='all'){await check();await koreaRun(c,'d=korea_load(); p=korea_select(d); korea_accounts(d,p); korea_verify(); korea_report()');}
  if(command==='verify'){await check();await koreaRun(c,'korea_verify()');}
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch((e:unknown)=>{console.error(e);process.exitCode=1;});
