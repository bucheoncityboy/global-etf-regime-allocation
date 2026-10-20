/** Resumable acquisition followed by the complete, verified Korean experiment. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { root } from './study.ts';
const statusPath=resolve(root,'data/private/korea/pipeline.json');
const started=new Date().toISOString();
let owned=false;
async function status(stage:string,error?:string):Promise<void>{
  await mkdir(resolve(root,'data/private/korea'),{recursive:true});
  await writeFile(statusPath,JSON.stringify({pid:process.pid,started_at_utc:started,updated_at_utc:new Date().toISOString(),stage,error},null,2)+'\n');
}
async function step(stage:string,args:string[]):Promise<void>{
  await status(stage);console.log(`${new Date().toISOString()} ${stage}`);
  await new Promise<void>((ok,fail)=>{const p=spawn(process.execPath,[resolve(root,'node_modules/tsx/dist/cli.mjs'),...args],{cwd:root,env:process.env,stdio:'inherit',windowsHide:true});
    p.on('error',fail);p.on('exit',code=>code===0?ok():fail(new Error(`${stage} exited ${code}`)));});
}
try{
  try{const old=JSON.parse(await readFile(statusPath,'utf8')) as {pid:number;stage:string};
    if(old.stage!=='completed'&&old.stage!=='failed'){
      let live=false;try{process.kill(old.pid,0);live=true;}catch{/* Previous owner no longer exists. */}
      assert.ok(!live,`Korean pipeline is already running as PID ${old.pid}`);
    }
  }catch(e){if((e as NodeJS.ErrnoException).code!=='ENOENT')throw e;}
  owned=true;
  await step('acquiring_market_inputs',['src/korea_data.ts']);
  await step('acquiring_daily_bond_panel',['src/korea_bonds.ts']);
  await step('validating_causal_implementation',['src/korea_harness.ts']);
  await step('preparing_training_allocating_verifying',['src/korea.ts','all']);
  await status('completed');console.log('Korean experiment completed; results/korea/README.md is ready for review.');
}catch(e){if(owned)await status('failed',String(e));console.error(e);process.exitCode=1;}
