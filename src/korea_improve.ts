import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { root, sha256 } from './study.ts';
import { koreaAtomicWrite } from './korea_bonds.ts';
import { koreaRun, type KoreaConfig } from './korea.ts';
import { koreaImprovement } from './korea_improve_kernel.ts';
import { koreaImprovementReport } from './korea_improve_report.ts';

export interface Variant {
  id:string; probability_weighted:boolean; mean_multiplier:number; turnover_buffer:number;
}
export interface ImprovementConfig {
  id:string;scope:string;original_commit:string;evaluation_start:string;end:string;
  classifiers:string[];variants:Variant[];baselines:string[];cost_bp:number;delays:number[];
  primary_delay:number;primary_variant:string;bootstrap:{risk_aversion:number;replications:number;block_lengths:number[];seed:number};
  no_retraining_or_parameter_search:boolean;
}
export const output='results/korea_improvement';
const newSources=['src/korea_improve.ts','src/korea_improve_kernel.ts','src/korea_improve_report.ts',
  'src/korea_improve_harness.ts','configs/korea-improvement.json','docs/korea-improvement-design.md'];
const publicInputs=['results/korea/design-freeze.json','results/korea/input-receipt.json','results/korea/status.json',
  'results/korea/verification.json','results/korea/predictions.parquet','results/korea/nav.parquet',
  'results/korea/targets.parquet','results/korea/trades.parquet','results/korea/fits.json','results/korea/tuning.json',
  'results/korea/summary.csv'];
interface Freeze {
  created_at_utc:string;scope:string;sources:Record<string,string>;public_inputs:Record<string,string>;prepared_sha256:string;
}
export function validateImprovement(c:ImprovementConfig):void {
  assert.equal(c.id,'korea-allocation-improvement-v1');
  assert.deepEqual(c.classifiers,['lr','xgb']);assert.deepEqual(c.delays,[0,1]);
  assert.deepEqual(c.baselines,['mv','minvar','ew','cash']);assert.equal(c.cost_bp,5);
  assert.equal(c.evaluation_start,'2018-07-01');assert.equal(c.end,'2025-09-30');
  assert.equal(c.primary_delay,1);assert.equal(c.primary_variant,'buffer');
  assert.deepEqual(c.variants,[
    {id:'hard',probability_weighted:false,mean_multiplier:1,turnover_buffer:0},
    {id:'probability',probability_weighted:true,mean_multiplier:1,turnover_buffer:0},
    {id:'shrink',probability_weighted:true,mean_multiplier:0.5,turnover_buffer:0},
    {id:'buffer',probability_weighted:true,mean_multiplier:0.5,turnover_buffer:0.02},
  ]);assert.equal(c.no_retraining_or_parameter_search,true);
}
export async function improvementRun(c:KoreaConfig,q:ImprovementConfig,action:string,capture=false):Promise<string> {
  return koreaRun(c,koreaImprovement+koreaImprovementReport+'\nset_improvement(json.loads('+JSON.stringify(JSON.stringify(q))+'))\n'+action,capture);
}
async function hashes(paths:string[]):Promise<Record<string,string>> {
  return Object.fromEntries(await Promise.all(paths.map(async p=>[p,sha256(await readFile(resolve(root,p)))] as const)));
}
async function checkOriginal():Promise<Record<string,string>> {
  const frozen=JSON.parse(await readFile(resolve(root,'results/korea/design-freeze.json'),'utf8')) as {sources:Record<string,string>};
  assert.deepEqual(await hashes(Object.keys(frozen.sources)),frozen.sources,'Original Korean sources changed');
  return frozen.sources;
}
async function check(privateInput:boolean):Promise<void> {
  const frozen=JSON.parse(await readFile(resolve(root,output,'design-freeze.json'),'utf8')) as Freeze;
  assert.deepEqual(await hashes(Object.keys(frozen.sources)),frozen.sources,'Improvement sources changed after freeze');
  assert.deepEqual(await hashes(Object.keys(frozen.public_inputs)),frozen.public_inputs,'Original public inputs changed');
  if(privateInput)assert.equal(sha256(await readFile(resolve(root,'data/private/korea/prepared.pkl'))),frozen.prepared_sha256,'Prepared input changed');
}
async function main():Promise<void> {
  const q=JSON.parse(await readFile(resolve(root,'configs/korea-improvement.json'),'utf8')) as ImprovementConfig;
  const c=JSON.parse(await readFile(resolve(root,'configs/korea.json'),'utf8')) as KoreaConfig;
  validateImprovement(q);const command=process.argv[2]??'run';
  assert.ok(['freeze','run','verify','verify-public'].includes(command),'Unknown improvement command');
  await mkdir(resolve(root,output),{recursive:true});
  if(command==='freeze'){
    const original=await checkOriginal();const receipt=JSON.parse(await readFile(resolve(root,'results/korea/input-receipt.json'),'utf8')) as {prepared_sha256:string};
    const freeze:Freeze={created_at_utc:new Date().toISOString(),scope:q.scope,sources:{...original,...await hashes(newSources)},
      public_inputs:await hashes(publicInputs),prepared_sha256:receipt.prepared_sha256};
    const path=resolve(root,output,'design-freeze.json');
    try{await readFile(path);throw new Error('Freeze already exists; do not overwrite a design after results');}
    catch(e:unknown){if((e as NodeJS.ErrnoException).code!=='ENOENT')throw e;}
    await koreaAtomicWrite(path,JSON.stringify(freeze,null,2)+'\n');await check(true);
    console.log('Exploratory design frozen; original results preserved.');return;
  }
  await check(command!=='verify-public');
  if(command==='verify-public'){
    await improvementRun(c,q,'improve_verify_public()');return;
  }
  await koreaRun(c,'korea_check_inputs()');
  if(command==='verify'){await improvementRun(c,q,'improve_verify(); improve_verify_public()');return;}
  const progressPath=resolve(root,'data/private/korea/improvement-process.json');
  await koreaAtomicWrite(progressPath,JSON.stringify({pid:process.pid,stage:'allocation',started_at_utc:new Date().toISOString()})+'\n');
  try {
    await improvementRun(c,q,'improve_run(); improve_verify(); improve_statistics(); improve_report(); improve_manifest(); improve_verify_public()');
    await check(true);
    await koreaAtomicWrite(progressPath,JSON.stringify({pid:process.pid,stage:'completed',updated_at_utc:new Date().toISOString()})+'\n');
  } catch(e:unknown) {
    await koreaAtomicWrite(progressPath,JSON.stringify({pid:process.pid,stage:'failed',updated_at_utc:new Date().toISOString(),error:String(e)})+'\n');throw e;
  }
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch((e:unknown)=>{console.error(e);process.exitCode=1;});
