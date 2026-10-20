import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { resolve, relative } from 'node:path';
import { common, preparation, fitting } from './kernel.ts';
import { assembly, allocation } from './portfolio.ts';
import { reporting, verification } from './report.ts';

export const root = fileURLToPath(new URL('../', import.meta.url));
export type Job = { profile: string; asset: string; boundary: string; asof: string; first: string; last: string };
type Completion = { done: Job; status: 'cached' | 'fit' };
export const sha256 = (data: string | Buffer): string => createHash('sha256').update(data).digest('hex');
const sourceFiles = ['src/kernel.ts','src/portfolio.ts','src/report.ts','src/study.ts','src/harness.ts',
  'configs/protocol.json','.python-version','pyproject.toml','uv.lock','package.json','package-lock.json','tsconfig.json'];
export async function sourceFingerprint(): Promise<Record<string,string>> {
  const entries = await Promise.all(sourceFiles.map(async path => [path,sha256(await readFile(resolve(root,path)))] as const));
  return Object.fromEntries(entries);
}
export function python(action: string, inherit = true) {
  const code = common + preparation + fitting + assembly + allocation + reporting + verification + '\n' + action + '\n';
  const child = spawn('uv', ['run','--frozen','python','-u','-c',
    "import sys; exec(compile(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())), '<scientific-adapter>', 'exec'))"], {
    cwd: root, shell: false,
    env: {...process.env, PYTHONUTF8:'1', OMP_NUM_THREADS:'1', OPENBLAS_NUM_THREADS:'1', MKL_NUM_THREADS:'1'},
    stdio: inherit ? ['pipe','inherit','inherit'] : ['pipe','pipe','inherit'],
  });
  child.stdin?.write(`${Buffer.byteLength(code,'utf8')}\n`);
  child.stdin?.write(code);
  return child;
}
export async function run(action: string): Promise<void> {
  await new Promise<void>((ok,fail) => {
    const child=python(action);
    child.on('error',fail); child.on('exit',code=>code===0?ok():fail(new Error(`Scientific backend exited ${code}`)));
    child.stdin?.end();
  });
}
async function seal(): Promise<void> {
  const sources=await sourceFingerprint();
  await mkdir(resolve(root,'results'),{recursive:true});
  await writeFile(resolve(root,'results/design-freeze.json'),JSON.stringify({sources,
    input_snapshot_sha256:sha256(await readFile(resolve(root,'data/input_snapshot.zip')))},null,2)+'\n');
}
async function checkSeal(): Promise<void> {
  const frozen=JSON.parse(await readFile(resolve(root,'results/design-freeze.json'),'utf8')) as {sources:Record<string,string>};
  const now=await sourceFingerprint();
  for (const [path,hash] of Object.entries(frozen.sources)) {
    if(now[path]!==hash) throw new Error(`Design/source changed: ${path}. Start a fresh run with prepare.`);
  }
}
async function fit(): Promise<void> {
  await checkSeal();
  const jobs=JSON.parse(await readFile(resolve(root,'results/jobs.json'),'utf8')) as Job[];
  let next=0,completed=0;
  const workers=Number(process.env.STUDY_WORKERS??'4');
  if(!Number.isInteger(workers)||workers<1||workers>16) throw new Error('STUDY_WORKERS must be 1..16');
  await Promise.all(Array.from({length:workers},(_,worker)=>new Promise<void>((ok,fail)=>{
    const child=python('worker()',false);
    const send=():void=>{const job=jobs[next++]; if(job)child.stdin?.write(JSON.stringify(job)+'\n');else child.stdin?.end();};
    if(!child.stdout) throw new Error('Missing worker output');
    createInterface({input:child.stdout}).on('line',(line:string)=>{
      try {
        const result=JSON.parse(line) as Completion;
        if(!result.done)throw new Error(line);
        completed++;
        console.log(`[${completed}/${jobs.length}] w${worker} ${result.done.profile} ${result.done.asset} ${result.done.asof} ${result.status}`);
        send();
      } catch(error) {fail(error);child.kill();}
    });
    child.on('error',fail);child.on('exit',code=>code===0?ok():fail(new Error(`Worker ${worker} exited ${code}`)));
    send();
  })));
}
async function manifest(): Promise<void> {
  const entries:Record<string,string>={};
  async function scan(dir:string):Promise<void>{
    for(const item of await readdir(dir,{withFileTypes:true})){
      if(item.name==='cache'||item.name==='artifact-manifest.json')continue;
      const path=resolve(dir,item.name);
      if(item.isDirectory())await scan(path);else entries[relative(root,path).replaceAll('\\','/')]=sha256(await readFile(path));
    }
  }
  await scan(resolve(root,'results'));
  await writeFile(resolve(root,'results/artifact-manifest.json'),JSON.stringify(entries,null,2)+'\n');
}
async function verifyManifest():Promise<void>{
  await checkSeal();
  const entries=JSON.parse(await readFile(resolve(root,'results/artifact-manifest.json'),'utf8')) as Record<string,string>;
  for(const [path,hash] of Object.entries(entries))if(sha256(await readFile(resolve(root,path)))!==hash)throw new Error(`Artifact changed: ${path}`);
}
async function main():Promise<void>{
  const command=process.argv[2]??'all';
  if(command==='prepare'||command==='all'){await run('prepare()');await seal();}
  if(command==='restore'){
    const preserved=await Promise.all(['results/jobs.json','data/manifest.json'].map(async path=>({path,bytes:await readFile(resolve(root,path))})));
    try {await run('prepare()');} finally {for(const item of preserved)await writeFile(resolve(root,item.path),item.bytes);}
  }
  if(command==='fit'||command==='all')await fit();
  if(command==='allocate'||command==='all'){await checkSeal();await run('assemble(); backtest()');}
  if(command==='report'||command==='all'){await checkSeal();await run('report(); verify(save=True)');await manifest();}
  if(command==='verify'||command==='all'){await verifyManifest();if(command==='verify')await run('verify()');}
  if(!['prepare','restore','fit','allocate','report','verify','all'].includes(command))throw new Error(`Unknown command ${command}`);
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch((error:unknown)=>{console.error(error);process.exitCode=1;});
