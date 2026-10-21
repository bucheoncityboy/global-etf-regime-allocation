/** Resume frozen Korean inputs and completed fits after an external interruption. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { root, sha256 } from './study.ts';
import { koreaAtomicWrite } from './korea_bonds.ts';

const statusPath=resolve(root,'data/private/korea/pipeline.json');
const started=new Date().toISOString();
let owned=false;
async function status(stage:string,error?:string):Promise<void>{
  await koreaAtomicWrite(statusPath,JSON.stringify({pid:process.pid,started_at_utc:started,
    updated_at_utc:new Date().toISOString(),stage,driver:'src/korea_resume.ts',error},null,2)+'\n');
}
async function step(stage:string,command:string):Promise<void>{
  await status(stage);console.log(`${new Date().toISOString()} ${stage}`);
  await new Promise<void>((ok,fail)=>{
    const child=spawn(process.execPath,[resolve(root,'node_modules/tsx/dist/cli.mjs'),'src/korea.ts',command],
      {cwd:root,env:process.env,stdio:'inherit',windowsHide:true});
    child.on('error',fail);child.on('exit',code=>code===0?ok():fail(new Error(`${stage} exited ${code}`)));
  });
}
try{
  const old=JSON.parse(await readFile(statusPath,'utf8')) as {pid:number;stage:string};
  assert.notEqual(old.stage,'completed','Korean experiment is already complete; inspect its verification and report');
  let live=false;try{process.kill(old.pid,0);live=true;}catch{/* Previous owner no longer exists. */}
  assert.ok(!live,`Korean pipeline is already running as PID ${old.pid}`);
  const freeze=await readFile(resolve(root,'results/korea/design-freeze.json'));
  await readFile(resolve(root,'data/private/korea/prepared.pkl'));
  await mkdir(resolve(root,'data/private/korea'),{recursive:true});
  owned=true;
  await status('resuming_frozen_inputs');
  await koreaAtomicWrite(resolve(root,'results/korea/resume-provenance.json'),JSON.stringify({
    started_at_utc:started,driver:'src/korea_resume.ts',
    driver_sha256:sha256(await readFile(resolve(root,'src/korea_resume.ts'))),
    original_design_freeze_sha256:sha256(freeze),inputs_reprepared:false,
    note:'The fit and allocate commands check every originally frozen source and input; existing fit caches are reused.'
  },null,2)+'\n');
  await step('training_from_frozen_inputs','fit');
  await step('allocating_verifying_reporting','allocate');
  const result=JSON.parse(await readFile(resolve(root,'results/korea/status.json'),'utf8')) as {status:string;strategy_performance_computed:boolean};
  assert.equal(result.status,'executed_and_independently_verified');
  assert.equal(result.strategy_performance_computed,true);
  await status('completed');console.log('Korean experiment completed from frozen inputs; results/korea/README.md is verified.');
}catch(error){if(owned)await status('failed',String(error));console.error(error);process.exitCode=1;}
