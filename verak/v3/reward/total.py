"""Training-only answer-based rewards; no action gate or in-loop verifier."""
import math
from statistics import mean

from .overedit import overedit
from .recovery import recover_records

LOCAL_LEVELS = ('WORD', 'SENTENCE', 'TEXT')


def quality_reward(before, after, genre, config):
    if not all(math.isfinite(v) for v in (before,after)):
        raise ValueError('Quality scores must be finite')
    settings = config['quality']
    floor = settings['noise_floor_by_genre'].get(genre, settings['noise_floor'])
    delta = after-before
    value = math.copysign(max(0.,abs(delta)-floor)/2., delta) if delta else 0.
    return {'value': value, 'delta': delta, 'noise_floor': floor, 'before': before, 'after': after}


def _mean(values):
    return mean(values) if values else 0.


def cohesion_changes(before,after):
    """Training penalty proxy from frozen observations, not semantic judgments.

    The obsolete REF-identity term becomes the accepted positional DEP hint.
    Closed-set conjunction changes have weight 1, predecessor hints weight .5.
    WORD/SENTENCE/TEXT aggregate weights are all 1.
    """
    old=before.structure()
    new=after.structure()
    earlier={a.sid:a for a in old.annotations}
    counts={'WORD':0.,'SENTENCE':0.,'TEXT':0.}
    changes=[]
    for ann in new.annotations:
        source=earlier.get(ann.sid)
        if source is not None:
            fields=[('polarity',source.polarity!=ann.polarity,'WORD',1.),
                    ('modality',source.modality!=ann.modality,'WORD',1.)]
            def ec(a):
                return [c['coarse_class'] for c in a.connectives if c['eligible']]
            def conj(a):
                return a.initial_conj['coarse_class'] if a.initial_conj and a.initial_conj['eligible'] else None
            fields += [('ec_relation',ec(source)!=ec(ann),'WORD',1.),
                       ('conjunction',conj(source)!=conj(ann),'SENTENCE',1.),
                       ('predecessor_hint',source.subject_omitted and ann.subject_omitted and
                        source.predecessor_id!=ann.predecessor_id,'SENTENCE',.5)]
        else:
            fields=[]
        fields.append(('off_style',old.dominant_style not in {'unknown','mixed'} and
                       ann.style not in {'unknown','mixed',old.dominant_style},'TEXT',1.))
        for kind,changed,level,weight in fields:
            if changed:
                counts[level]+=weight
                changes.append({'sid':ann.sid,'kind':kind,'level':level,'weight':weight})
    denominator=max(1,len(old.annotations))
    return {'value':sum(counts.values())/denominator,'counts_by_level':counts,
            'sentence_count':denominator,'changes':changes,'interpretation':'structural_change_proxy'}


def real_reward(before,after,*,q_before,q_after,genre,actions,config):
    """Spec 9.3 real-essay reward; no answer recovery or verifier call."""
    quality=quality_reward(q_before,q_after,genre,config)
    cohesion=cohesion_changes(before,after)
    terms={'quality':config.get('w_q',.3)*quality['value'],
           'cohesion':-config.get('w_coh',1.)*cohesion['value'],
           'steps':-config.get('w_step',.01)*len(actions)}
    return {'R':sum(terms.values()),'R_q':quality['value'],'R_step':len(actions),
            'CohesionDamage':cohesion['value'],'quality':quality,'cohesion':cohesion,
            'weighted_components':terms}


def _breakdown(rec_values, per_record, quality, over, steps, weights):
    rec = _mean(rec_values)
    terms = {'recovery': weights.get('w_rec',1.)*rec,
             'quality': weights.get('w_q',.3)*quality['value'],
             'overedit': -weights.get('w_over',.5)*over['value'],
             'steps': -weights.get('w_step',.01)*steps}
    return {'R': sum(terms.values()), 'R_rec': rec, 'R_q': quality['value'],
            'R_over': over['value'], 'R_step': steps, 'weighted_components': terms,
            'quality': quality, 'overedit': over, 'recovery_terms': rec_values,
            'per_record': per_record,
            'per_level_main': {level: _mean([r['main'] for r in per_record if r['level']==level])
                               for level in ('GLOBAL',)+LOCAL_LEVELS}}


def rewards(source, corrupted, final, records, *, genre, q_corrupted, q_final, config,
            mode='single', stage1=None, q_stage1=None, stage1_actions=(), stage2_actions=(),
            actions=(), similarity=None, tau=None, preexisting_spell_spans=()):
    """Return GLOBAL/KOREAN/combined, keeping final-stage recovery out of GLOBAL.

    Korean recovery is a mean of local-record main terms and one normalized
    coupled term per GLOBAL record with coupled damage. Each local record has
    equal weight, irrespective of level (the corpus is already level-balanced).
    No-record roles get zero recovery and still pay their action/over-edit costs.
    """
    if mode not in {'single','two_stage'}:
        raise ValueError('Unknown episode mode')
    weight = config.get('dependents_weight',.3)
    end = recover_records(source, final, records, similarity=similarity, tau=tau, coupled_weight=weight)
    all_actions = list(stage1_actions)+list(stage2_actions) if mode=='two_stage' else list(actions)
    combined = _breakdown([r['recovery'] for r in end], end,
        quality_reward(q_corrupted,q_final,genre,config),
        overedit(source,source,final,records,preexisting_spell_spans=preexisting_spell_spans),
        len(all_actions),config)
    if mode=='single':
        return {'mode': mode, 'global': None, 'korean': None, 'combined': combined}
    if stage1 is None or q_stage1 is None:
        raise ValueError('two_stage rewards require actual stage-1 text and score')
    middle = recover_records(source,stage1,records,similarity=similarity,tau=tau,coupled_weight=weight)
    global_records = [r for r in middle if r['level']=='GLOBAL']
    korean_records = [r for r in end if r['level'] in LOCAL_LEVELS or r['coupled'] is not None]
    korean_terms = [r['main'] for r in end if r['level'] in LOCAL_LEVELS] + [
        r['coupled'] for r in end if r['level']=='GLOBAL' and r['coupled'] is not None]
    global_result = _breakdown([r['main'] for r in global_records],global_records,
        quality_reward(q_corrupted,q_stage1,genre,config),
        overedit(source,corrupted,stage1,records,actions=stage1_actions,
                 preexisting_spell_spans=preexisting_spell_spans),len(stage1_actions),config)
    korean_result = _breakdown(korean_terms,korean_records,
        quality_reward(q_stage1,q_final,genre,config),
        overedit(source,stage1,final,records,actions=stage2_actions,
                 preexisting_spell_spans=preexisting_spell_spans),len(stage2_actions),config)
    return {'mode': mode, 'global': global_result, 'korean': korean_result, 'combined': combined}
