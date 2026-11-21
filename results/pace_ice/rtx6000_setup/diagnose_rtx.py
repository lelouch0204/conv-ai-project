"""Reproduce the repeated invitation/meeting divergence on RTX 6000."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path.cwd()))
from src.data import load_traces
from src.engine import VLLMEngine
revision=Path('results/pace_setup/model_revision.txt').read_text().strip()
engine=VLLMEngine(dtype='float16',revision=revision)
engine.warmup()
trace=next(t for t in load_traces('data/fdb_v2_sample/dev.jsonl') if t.conversation_id=='dev_daily_slow_003' and t.turn_id==1)
reference=engine.generate_greedy_reference(trace.history,trace.final.text,48)
seq,_=engine.prefill_prompt('diagnostic',trace.history,trace.final.text,True,48)
while not seq.finished:engine.decode_step([seq])
draft=engine.generate_greedy_reference(trace.history,trace.partials[-1].text,20)
ver,state=engine.verify_and_prefill_final('verify',trace.history,trace.final.text,draft,48)
while not state.finished:engine.decode_step([state])
report={'revision':revision,'reference_tokens':reference,'reference_text':engine.tokens_to_text(reference),
        'incremental_tokens':seq.generated_tokens,'incremental_text':engine.tokens_to_text(seq.generated_tokens),
        'draft_tokens':draft,'accepted':ver.accepted_count,'bonus_token':ver.bonus_token,
        'verified_tokens':state.generated_tokens,'verified_text':engine.tokens_to_text(state.generated_tokens)}
base=engine._format_prompt_ids(trace.history,trace.final.text)
position=5
prefix=base+reference[:position]
def describe(entry):
 return [{'id':int(k),'text':engine.tokens_to_text([k]),'rank':getattr(v,'rank',None),'logprob':float(v.logprob)} for k,v in (entry or {}).items()]
outputs,_=engine._timed_generate([{'prompt_token_ids':prefix}],engine._greedy_params(max_tokens=1,logprobs=5))
report['canonical_next_token']=int(outputs[0].outputs[0].token_ids[0])
report['canonical_next_scores']=describe(outputs[0].outputs[0].logprobs[0])
report['prompt_score_probes']=[]
for suffix in (draft,reference[:5]+[reference[0]]+reference[6:12],reference[:12]):
 outputs,_=engine._timed_generate([{'prompt_token_ids':base+suffix}],engine._greedy_params(max_tokens=1,prompt_logprobs=5))
 entry=outputs[0].prompt_logprobs[len(base)+position]
 report['prompt_score_probes'].append({'candidate_token':suffix[position],'suffix_length':len(suffix),'scores':describe(entry)})
report['batch_probes']=[]
for size in (1,2,4,8):
 outputs,_=engine._timed_generate([{'prompt_token_ids':prefix} for _ in range(size)],engine._greedy_params(max_tokens=1,logprobs=5))
 report['batch_probes'].append({'size':size,'tokens':[int(out.outputs[0].token_ids[0]) for out in outputs]})
Path('results/pace_setup/rtx_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
