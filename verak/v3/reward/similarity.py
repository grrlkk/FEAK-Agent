"""Local, frozen CLS embeddings and reconstruction-label ROC calibration."""
from collections import Counter
import gc
from pathlib import Path
import time

import numpy as np

from ..common import read_json,write_json

MODELS={
    'BAAI/bge-m3':('models--BAAI--bge-m3','5617a9f61b028005a4858fdac845db406aefb181'),
    'nlpai-lab/KURE-v1':('models--nlpai-lab--KURE-v1','d14c8a9423946e268a0c9952fecf3a7aabd73bd9')}


class SentenceSimilarity:
    def __init__(self,model_name,*,device='cuda:1',batch_size=16):
        import torch
        from transformers import AutoModel,AutoTokenizer
        folder,revision=MODELS[model_name]
        self.path=Path.home()/'.cache/huggingface/hub'/folder/'snapshots'/revision
        pooling=read_json(self.path/'1_Pooling/config.json')
        if not pooling['pooling_mode_cls_token'] or pooling['pooling_mode_mean_tokens']:
            raise ValueError('Unexpected embedding pooling; do not silently change model')
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.path),local_files_only=True)
        self.model=AutoModel.from_pretrained(str(self.path),local_files_only=True).to(device).eval()
        self.device,self.batch_size=device,batch_size
        self.cache={}
        self.revision=revision

    def encode(self,texts):
        import torch
        missing=list(dict.fromkeys(s for s in texts if s not in self.cache))
        for start in range(0,len(missing),self.batch_size):
            batch=missing[start:start+self.batch_size]
            encoded=self.tokenizer(batch,padding=True,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>8192:
                raise ValueError('Embedding input exceeds 8192; no truncation')
            with torch.inference_mode():
                hidden=self.model(**{k:v.to(self.device) for k,v in encoded.items()}).last_hidden_state[:,0]
                vectors=torch.nn.functional.normalize(hidden,dim=1).float().cpu().numpy()
            self.cache.update(zip(batch,vectors))
        return np.array([self.cache[s] for s in texts])

    def __call__(self,left,right):
        a,b=self.encode([left,right])
        return float(np.clip(np.dot(a,b),-1,1))

    def close(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def roc_metrics(labels,scores):
    from sklearn.metrics import roc_auc_score,roc_curve
    labels=np.asarray(labels,dtype=bool)
    scores=np.asarray(scores,dtype=float)
    if len(labels)!=len(scores) or not np.isfinite(scores).all() or len(set(labels))!=2:
        raise ValueError('ROC needs finite scores and both label classes')
    fpr,tpr,thresholds=roc_curve(labels,scores,drop_intermediate=False)
    # Ignore the ROC infinity sentinel. On equal J, prefer higher specificity.
    choices=[i for i,t in enumerate(thresholds) if np.isfinite(t) and -1<=t<=1]
    chosen=max(choices,key=lambda i:(tpr[i]-fpr[i],-fpr[i],thresholds[i]))
    return {'n':len(labels),'positive':int(labels.sum()),'negative':int((~labels).sum()),
            'roc_auc':float(roc_auc_score(labels,scores)),'tau':float(thresholds[chosen]),
            'youden_j':float(tpr[chosen]-fpr[chosen]),'tpr':float(tpr[chosen]),'fpr':float(fpr[chosen])}


def make_pairs(cases,reconstructions):
    by_id={c['item_id']:c for c in cases}
    sources=sorted({r['source'] for r in reconstructions})
    if len(by_id)!=200:
        raise ValueError('Calibration requires 200 unique deletion sites')
    for source in sources:
        ids=[r['item_id'] for r in reconstructions if r['source']==source]
        if len(ids)!=200 or set(ids)!=set(by_id):
            raise ValueError('Each reconstruction source needs 200 distinct labeled cases')
    if not {'kanana','luna'}<=set(sources):
        raise ValueError('Both current reconstruction sources are required')
    if any(type(r['same_role']) is not bool for r in reconstructions):
        raise ValueError('same_role must be a verified boolean')
    pairs=[{'item_id':r['item_id'],'source':r['source'],'left':by_id[r['item_id']]['deleted_sentence'],
            'right':r['reconstruction'],'label':r['same_role'],'kind':'reconstruction'} for r in reconstructions]
    pairs += [{'item_id':c['item_id'],'source':'negative','left':c['deleted_sentence'],
               'right':c['negative_sentence'],'label':False,'kind':'same_essay_negative'} for c in cases]
    return pairs


def calibrate(cases,reconstructions,output,*,device='cuda:1'):
    pairs=make_pairs(cases,reconstructions)
    sources=sorted({r['source'] for r in reconstructions})
    metrics={}
    for name in MODELS:
        encoder=SentenceSimilarity(name,device=device)
        start=time.monotonic()
        vectors=encoder.encode([r[key] for r in pairs for key in ('left','right')])
        scores=np.clip(np.sum(vectors[::2]*vectors[1::2],axis=1),-1,1)
        result={'revision':encoder.revision,'elapsed_s':time.monotonic()-start,'by_source':{}}
        for source in sources+['union']:
            idx=[i for i,r in enumerate(pairs) if source=='union' or r['source'] in {source,'negative'}]
            result['by_source'][source]=roc_metrics([pairs[i]['label'] for i in idx],scores[idx])
            tau=result['by_source'][source]['tau']
            rec_idx=[i for i in idx if pairs[i]['kind']=='reconstruction']
            regions=Counter('zero' if scores[i]<=tau-.05 else 'one' if scores[i]>=tau+.05 else 'linear' for i in rec_idx)
            result['by_source'][source]['regions_at_own_tau']={k:{'count':regions[k],'share':regions[k]/len(rec_idx)}
                                                                            for k in ('zero','linear','one')}
        np.savez_compressed(output/(name.replace('/','__')+'_scores.npz'),scores=scores)
        encoder.close()
        metrics[name]=result
    chosen=max(metrics,key=lambda name:metrics[name]['by_source']['union']['roc_auc'])
    tau=metrics[chosen]['by_source']['union']['tau']
    scores=np.load(output/(chosen.replace('/','__')+'_scores.npz'))['scores']
    for source in sources+['union']:
        idx=[i for i,r in enumerate(pairs) if r['kind']=='reconstruction' and (source=='union' or r['source']==source)]
        regions=Counter('zero' if scores[i]<=tau-.05 else 'one' if scores[i]>=tau+.05 else 'linear' for i in idx)
        metrics[chosen]['by_source'][source]['regions_at_selected_union_tau']={
            k:{'count':regions[k],'share':regions[k]/len(idx)} for k in ('zero','linear','one')}
    result={'models':metrics,'selected_model':chosen,'tau':tau,'margin':.05,'pairs':len(pairs),
            'reconstruction_sources':sources,'unique_negatives':200,
            'limitations':['same-essay negatives are rule-labeled, not judge-verified',
                          'threshold fitted and ROC reported on the same dev calibration sample']}
    write_json(output/'similarity_metrics.json',result)
    return result
