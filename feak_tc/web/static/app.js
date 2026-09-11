"use strict";
const $ = (id) => document.getElementById(id);
const state = {id:null, events:[], result:null, status:null, request:null, meta:null, activeId:null, cursor:0, view:"text", selection:null, token:0, timer:null, createdAt:null, model:null};
const emptyMarkup={paper:$("essay-view").innerHTML,timeline:$("timeline").innerHTML,candidates:$("candidates-list").innerHTML};
const rubricNames = {task_1:"과제 충실성",content_1:"설명 명료성",content_2:"설명 구체성",content_3:"설명 적절성",organization_1:"문장 연결성",organization_2:"글 통일성",expression_1:"어휘 적절성",expression_2:"어법 적절성"};
const actions = {ADD_DETAIL:"설명 보충",DELETE_OR_FOCUS:"삭제·초점 조정",COMPRESS:"중복 압축",RESTRUCTURE:"흐름 재구성",STYLE_REFINE:"표현 다듬기"};
const reasons = {no_improvement_over_noop:"현재 글을 유지하는 것보다 유리하지 않음",quality_drop:"전체 진단 점수 하락",non_target_drop:"다른 평가 항목 하락",rv_target_fulfillment:"수정 목표 달성 불충분",rv_preservation:"내용 보존 불충분",candidate_error:"수정 응답을 확인할 수 없음",duplicate_candidate:"다른 후보와 같은 수정",visited_state:"이미 검토한 글",no_effect:"실질적인 변화 없음",guard_preservation:"경로 가드의 내용 보존 실패 판정",guard_coherence:"전체 흐름 훼손 판정",guard_unavailable:"경로 검증을 완료하지 못함",checkpoint_quality_drop:"이전 checkpoint 대비 점수 하락",repeated_failed_plan:"같은 상태에서 이미 실패한 계획",max_steps:"설정한 수정 단계에 도달했습니다",no_progress:"추가로 채택할 수 있는 수정이 없습니다",planner_no_request:"더 고칠 근거가 없어 마쳤습니다",rollback_limit:"설정한 복구 횟수에 도달했습니다",llm_call_budget:"모델 호출 예산을 모두 사용했습니다",runtime_error:"실행 중 오류가 발생했습니다",user_cancelled:"실행을 중단하고 마지막 검증본을 보관했습니다",process_error:"실행이 중단되었습니다",target_span_not_found:"수정 위치를 원문에서 찾을 수 없음",ambiguous_target_span:"수정 위치를 하나로 특정할 수 없음",protected_definition:"핵심 정의 문장 보호"};
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
Object.assign(reasons,{edit_ratio:"허용한 편집량 초과",target_gain:"대상 평가 항목의 개선 기준 미달",goal_preservation:"원문 의미 유사도 기준 미달",evidence_match:"대상 부분과 수정 내용의 연결 기준 미달"});
const friendly = (value) => reasons[value] || (value?.startsWith("validity:") ? "수정 형식 검사 미통과" : value || "");
const labelText = (value) => ({pass:"통과",partial:"부분 충족",fail:"미통과"}[value] || "판정 불가");
const modelName = (model) => model?.toLowerCase().includes("kanana") ? "Kanana 8B" : model?.includes("Qwen") ? "Qwen 7B · 이전 실행" : model || "로컬 모델";
function notice(message=""){ $("notice").textContent=message; $("notice").hidden=!message; }
async function api(path, options={}){
  const response=await fetch(path, options);
  const body=await response.json();
  if(!response.ok) throw new Error(body.error || "요청을 처리하지 못했습니다.");
  return body;
}
function setBusy(){
  $("run-button").disabled=!!state.activeId;
  $("run-button").firstElementChild.textContent=state.activeId?"에이전트 실행 중":"수정 과정 시작";
  $("cancel-button").hidden=!(state.id===state.activeId && state.status==="running");
  $("active-button").hidden=!state.activeId || state.activeId===state.id;
}
async function refreshHistory(){
  const data=await api("/api/runs");
  state.activeId=data.active_id;
  $("history").replaceChildren(new Option("실행 기록 선택", ""));
  for(const run of data.runs){
    const mode=run.request.offline_smoke?"데모":modelName(run.model);
    const status={running:"진행 중",completed:"완료",cancelled:"중단",error:"오류"}[run.status] || run.status;
    const title=run.request.question || "제목 없는 실행";
    const option=new Option(`${status} · ${title.slice(0,24)} · ${mode}`,run.id);
    option.title=title;
    $("history").add(option);
  }
  $("history").value=state.id || "";
  setBusy();
  return data;
}
async function loadRun(id){
  if(!id) return;
  clearTimeout(state.timer);
  state.token++;
  Object.assign(state,{id,events:[],result:null,status:"loading",request:null,cursor:0,selection:null});
  render();
  localStorage.setItem("feak_run_id",id);
  notice();
  await poll(id,state.token);
}
async function poll(id,token){
  try{
    const data=await api(`/api/runs/${encodeURIComponent(id)}?cursor=${state.cursor}`);
    if(token!==state.token) return;
    state.events.push(...data.events);
    Object.assign(state,{cursor:data.cursor,status:data.status,result:data.result,request:data.request,model:data.model,createdAt:data.created_at});
    if(state.result?.events) state.events=state.result.events;
    if(data.error) notice(data.error); else notice();
    render();
    if(data.status==="running") state.timer=setTimeout(()=>poll(id,token),1300);
    else await refreshHistory();
  }catch(error){
    if(token!==state.token) return;
    notice("서버 연결을 확인하고 있습니다. " + error.message);
    state.timer=setTimeout(()=>poll(id,token),3000);
  }
}
function currentSnapshot(){
  const original=state.result?.original_text ?? state.request?.text ?? $("essay").value;
  let text=original, diagnosis=null, id=0, provisional=false;
  const checkpoints=new Map([[0,{text:original,diagnosis:null}]]);
  for(const row of state.events){
    if(row.event==="start"){diagnosis=row.diagnosis;checkpoints.set(0,{text:original,diagnosis});}
    if(row.event==="accept"){text=row.text;diagnosis=row.diagnosis;id=row.checkpoint_id;provisional=true;checkpoints.set(id,{text,diagnosis});}
    if(row.event==="guard") provisional=!!row.reasons?.length || !!row.error;
    if(row.event==="rollback"){const saved=checkpoints.get(row.to_checkpoint);text=row.text;diagnosis=saved?.diagnosis;id=row.to_checkpoint;provisional=false;}
  }
  if(state.result){text=state.result.final_text;id=state.result.final_checkpoint_id;diagnosis=state.result.checkpoints?.find(x=>x.id===id)?.diagnosis || diagnosis;provisional=false;}
  return {original,text,diagnosis,id,provisional,checkpoints};
}
function candidateRows(){
  const rows=new Map();
  for(const event of state.events){
    if(event.event==="patch" || event.event==="candidate") rows.set(`${event.step}:${event.index}`,event);
  }
  return [...rows.values()];
}
function selectedSnapshot(){
  const snapshot=currentSnapshot();
  if(state.selection?.type==="candidate"){
    const row=candidateRows().find(x=>x.step===state.selection.step && x.index===state.selection.index);
    if(row?.candidate?.new_text) return {...snapshot,text:row.candidate.new_text,diagnosis:row.after,label:`${row.step}단계 · 후보 ${String.fromCharCode(65+row.index)}`,preview:true};
  }
  if(state.selection?.type==="checkpoint"){
    const entry=snapshot.checkpoints.get(state.selection.id);
    if(entry) return {...snapshot,...entry,label:`저장 지점 ${state.selection.id}`,preview:true};
  }
  return {...snapshot,label:snapshot.id===0?"원문":`저장 지점 ${snapshot.id}${snapshot.provisional?" · 검증 중":""}`,preview:false};
}
function diffHTML(before,after){
  if(before===after) return escapeHTML(after);
  const a=before.match(/\S+\s*|\s+/g)||[], b=after.match(/\S+\s*|\s+/g)||[];
  if(a.length*b.length>300000){
    let start=0,end=0;while(start<Math.min(before.length,after.length)&&before[start]===after[start])start++;
    while(end<Math.min(before.length,after.length)-start&&before[before.length-1-end]===after[after.length-1-end])end++;
    return escapeHTML(before.slice(0,start))+`<del>${escapeHTML(before.slice(start,before.length-end))}</del><ins>${escapeHTML(after.slice(start,after.length-end))}</ins>`+escapeHTML(after.slice(after.length-end));
  }
  const matrix=Array.from({length:a.length+1},()=>new Uint16Array(b.length+1));
  const sameToken=(x,y)=>x.trimEnd()===y.trimEnd();
  for(let i=a.length-1;i>=0;i--)for(let j=b.length-1;j>=0;j--)matrix[i][j]=sameToken(a[i],b[j])?matrix[i+1][j+1]+1:Math.max(matrix[i+1][j],matrix[i][j+1]);
  let i=0,j=0,html="";
  while(i<a.length||j<b.length){
    if(i<a.length&&j<b.length&&sameToken(a[i],b[j])){html+=escapeHTML(b[j]);i++;j++;}
    else if(j<b.length&&(i===a.length||matrix[i][j+1]>=matrix[i+1][j]))html+=`<ins>${escapeHTML(b[j++])}</ins>`;
    else html+=`<del>${escapeHTML(a[i++])}</del>`;
  }
  return html;
}
function renderPaper(){
  const selected=selectedSnapshot();
  $("essay-view").innerHTML=selected.text?(state.view==="diff"?diffHTML(selected.original,selected.text):escapeHTML(selected.text)):emptyMarkup.paper;
  $("version-badge").textContent=selected.label;
  $("preview-banner").hidden=!selected.preview;
  $("preview-label").textContent=selected.preview?`${selected.label} 미리보기 · 최종 글과 다를 수 있어요`:"";
  $("diff-legend").hidden=state.view!=="diff" || !selected.text;
  $("paper-chars").textContent=`${selected.text.length.toLocaleString()}자`;
  $("download-button").disabled=!state.result;
  $("copy-button").disabled=!selected.text;
  renderRubrics(selected.diagnosis);
}
function scores(diagnosis){
  if(!diagnosis) return null;
  const continuous=diagnosis.metadata?.rf_corrected_score;
  return Array.isArray(continuous)&&continuous.length===8?Object.fromEntries(Object.keys(rubricNames).map((key,i)=>[key,continuous[i]])):diagnosis.rubrics;
}
function renderRubrics(diagnosis){
  const original=scores(state.events.find(x=>x.event==="start")?.diagnosis), current=scores(diagnosis);
  if(!original){$("rubric-chart").innerHTML='<p class="empty-small">첫 진단이 끝나면 8개 평가 항목의 점수가 표시됩니다.</p>';return;}
  if(!current){$("rubric-chart").innerHTML='<p class="empty-small">선택한 후보의 재진단 결과가 아직 없거나, 재진단 전에 거절된 후보입니다.</p>';return;}
  $("rubric-chart").innerHTML=Object.entries(rubricNames).map(([key,label])=>{
    const value=Number(current[key]),base=Number(original[key]),delta=value-base;
    if(!Number.isFinite(value)||!Number.isFinite(base)) return "";
    return `<div class="rubric-row"><span class="rubric-label">${label}</span><div class="rubric-track" aria-label="${label}: 원문 ${base.toFixed(2)}, 현재 ${value.toFixed(2)}"><span class="rubric-fill" style="width:${Math.max(0,Math.min(100,value/9*100))}%"></span><i class="rubric-original" style="left:${Math.max(0,Math.min(100,base/9*100))}%"></i></div><span class="rubric-score">${value.toFixed(1)}</span><span class="rubric-delta ${delta>.005?"up":delta<-.005?"down":""}">${Math.abs(delta)<.005?"—":(delta>0?"+":"")+delta.toFixed(2)}</span></div>`;
  }).join("");
}
function renderTimeline(){
  const rows=state.events.filter(x=>["start","plan","accept","reject","guard","rollback","replan","stop","error"].includes(x.event));
  $("timeline-count").textContent=rows.length;
  if(!rows.length){$("timeline").innerHTML=emptyMarkup.timeline;return;}
  $("timeline").innerHTML=rows.map(row=>{
    let title="",detail="",button="";
    if(row.event==="start"){title="첫 진단 완료";detail="수정 전 글과 평가 점수를 보관했습니다.";}
    if(row.event==="plan"){title=row.response.plan?`${row.step}단계 · ${actions[row.response.plan.action_type]||"수정 계획"}`:"추가 수정 없음";detail=row.response.plan?.problem || row.response.reason;}
    if(row.event==="accept"){title=`후보 ${String.fromCharCode(65+row.chosen_index)} 채택`;detail="경로 가드에서 글 전체를 다시 확인합니다.";button=`<button data-checkpoint="${Number(row.checkpoint_id)}">저장 지점 ${Number(row.checkpoint_id)} 보기 ↗</button>`;}
    if(row.event==="reject"){title="후보 채택 보류";detail=friendly(row.reason)==="no_viable_candidate"?"현재 글을 유지하고 다른 수정을 검토합니다.":friendly(row.reason);}
    if(row.event==="guard"){title=row.reasons?.length?"경로 가드 · 복구 필요":"경로 가드 통과";detail=row.verdict?.preservation?.reason || row.error || "글 전체의 내용과 흐름을 확인했습니다.";}
    if(row.event==="rollback"){title=`저장 지점 ${row.to_checkpoint}로 복구`;detail=(row.reasons||[]).map(friendly).join(" · ");button=`<button data-checkpoint="${Number(row.to_checkpoint)}">복구된 글 보기 ↗</button>`;}
    if(row.event==="replan"){title="다른 수정 계획 검토";detail="남은 실행 예산 안에서 다음 단계를 준비합니다.";}
    if(row.event==="stop"){title="수정 과정 종료";detail=friendly(row.reason);}
    if(row.event==="error"){title="실행 오류";detail=row.message;}
    return `<div class="timeline-item ${escapeHTML(row.event)}"><small>${row.step?`${row.step}단계` : "FEAK"}</small><strong>${escapeHTML(title)}</strong><p>${escapeHTML(detail)}</p>${button}</div>`;
  }).join("");
}
function renderCandidates(){
  const rows=candidateRows();
  $("candidate-count").textContent=rows.length?`총 ${rows.length}개 · 클릭해서 비교해 보세요`:"후보가 만들어지면 이곳에 나타납니다.";
  if(!rows.length){$("candidates-list").innerHTML=emptyMarkup.candidates;return;}
  const expanded=new Set([...$("candidates-list").querySelectorAll("details[open]")].map(x=>x.closest("article").dataset.key));
  $("candidates-list").innerHTML=rows.map(row=>{
    const candidate=row.candidate || {}, patch=candidate.patch || {};
    const beforeClass=patch.operation==="insert_after"?"patch-anchor":"patch-before";
    const accepted=state.events.find(x=>x.event==="accept" && x.step===row.step && x.chosen_index===row.index);
    const restored=accepted && state.events.some(x=>x.event==="rollback"&&x.from_checkpoint===accepted.checkpoint_id);
    const decision=restored?"채택 후 복구":accepted?"채택":row.rejected?"거절":row.event==="patch"?"검증 중":"통과 · 비교 후보";
    const klass=restored?"rolled-back":row.rejected?"rejected":"";
    const selected=state.selection?.type==="candidate"&&state.selection.step===row.step&&state.selection.index===row.index;
    const axes=[["target_fulfillment","목표 달성"],["preservation","내용 보존"]];
    const verdicts=axes.map(([key,label])=>`<span class="verdict ${escapeHTML(row.rv?.[key]?.label || "unknown")}">${label} · ${row.rv?labelText(row.rv[key]?.label):row.event==="patch"?"대기":"미실행"}</span>`).join("");
    const details=axes.map(([key,label])=>`<p><strong>${label}</strong><br>${escapeHTML(row.rv?.[key]?.reason || "아직 판정하지 않았습니다.")}</p>`).join("");
    return `<article data-key="${Number(row.step)}:${Number(row.index)}" class="candidate-card ${selected?"selected":""}"><div class="candidate-card-header"><strong>${Number(row.step)}단계 · 후보 ${String.fromCharCode(65+row.index)}</strong><span class="decision-badge ${klass}">${decision}</span></div><div class="candidate-body"><div class="action-label">${escapeHTML(actions[candidate.action_type]||candidate.action_type||"수정 응답")} · ${escapeHTML(rubricNames[candidate.target_rubric]||"")}</div><div class="candidate-patch">${patch.before?`<span class="${beforeClass}">${escapeHTML(patch.before)}</span>`:""}<span class="patch-after">${escapeHTML(patch.after || (patch.operation==="delete"?"대상 문장을 삭제합니다.":row.error||"수정 내용을 기다리고 있습니다."))}</span></div><div class="verdicts">${verdicts}</div>${row.reject_reasons?.length?`<p class="rejection-note">${escapeHTML(row.reject_reasons.map(friendly).join(" · "))}</p>`:""}<details ${expanded.has(`${row.step}:${row.index}`)?"open":""}><summary>수정 요구와 판정 근거</summary><p><strong>수정 요구</strong><br>${escapeHTML(row.request?.instruction||candidate.instruction||"")}</p><p><strong>보존 조건</strong><br>${escapeHTML((row.request?.preserve||[]).join("\n"))}</p>${details}</details></div><button class="candidate-preview" data-candidate="${Number(row.step)}:${Number(row.index)}">전체 글에서 비교하기 ↗</button></article>`;
  }).join("");
}
function renderStatus(){
  const phase=[...state.events].reverse().find(x=>x.event==="phase"), last=state.events.at(-1);
  const currentStep=Math.max(0,...state.events.map(x=>x.step||0));
  const running=state.status==="running";
  const stageText={diagnose:"글을 진단하고 있어요",plan:"수정 계획을 세우고 있어요",generate:"수정 후보를 만들고 있어요",rediagnose:"수정 후보를 다시 진단하고 있어요",verify:"목표 달성과 내용 보존을 검증하고 있어요",guard:"글 전체의 변화와 흐름을 확인하고 있어요"};
  $("status-text").textContent=running?`${currentStep?`${currentStep}단계 · `:""}${stageText[phase?.stage]||"실행을 준비하고 있어요"}`:state.status==="cancelled"?"실행이 중단됐어요":state.status==="error"?"실행 기록에서 오류를 확인해 주세요":state.status==="completed"?friendly(state.result?.stop_reason||last?.reason)||"수정 과정을 마쳤어요":"시작할 준비가 됐어요";
  $("status-dot").className=running?"running":state.status==="error"?"error":"";
  const stage=phase?.stage==="rediagnose"?"verify":phase?.stage;
  const order=["diagnose","plan","generate","verify","guard"];
  document.querySelectorAll("[data-stage]").forEach(el=>{el.classList.toggle("active",running&&el.dataset.stage===stage);el.classList.toggle("done",!!state.id&&order.indexOf(el.dataset.stage)<order.indexOf(stage));});
  $("stat-steps").textContent=state.id?`${currentStep} / ${state.request?.max_steps||currentStep||"—"}`:"—";
  $("stat-candidates").textContent=state.id?candidateRows().length:"—";
  $("stat-accepted").textContent=state.id?state.events.filter(x=>x.event==="accept").length:"—";
  $("stat-rollbacks").textContent=state.id?state.events.filter(x=>x.event==="rollback").length:"—";
  $("model-badge").textContent=state.request?.offline_smoke?"화면 데모 · 모의 판정":`${modelName(state.model||state.meta?.model)} · 로컬`;
  $("workspace-title").textContent=state.id?(state.request?.offline_smoke?"모델 없이 과정을 둘러보는 중.":running?"한 단계씩, 글을 다듬고 있어요.":"수정의 결과와 이유를 확인하세요."):"글이 바뀌는 과정을 한눈에.";
  $("workspace-subtitle").textContent=state.request?.question || "작은 수정부터 마지막 선택까지, 판단의 근거를 함께 확인하세요.";
  setBusy();
  updateElapsed();
}
function updateElapsed(){
  if(state.status==="running"&&state.createdAt){const s=Math.max(0,Math.floor((Date.now()-Date.parse(state.createdAt))/1000));$("elapsed").textContent=`${Math.floor(s/60)}분 ${String(s%60).padStart(2,"0")}초 경과`;}
  else $("elapsed").textContent=state.id?"SAVED LOCALLY":"LOCAL WORKSPACE";
}
function render(){renderStatus();renderPaper();renderTimeline();renderCandidates();}
$("essay").addEventListener("input",()=>{$("char-count").textContent=`${$("essay").value.length.toLocaleString()} / 4,000자`;if(!state.id)renderPaper();});
$("example-button").addEventListener("click",()=>{$("essay").value=state.meta.example;$("question").value=state.meta.example_question;$("essay").dispatchEvent(new Event("input"));});
$("text-file").addEventListener("change",async event=>{const file=event.target.files[0];if(!file)return;if(file.size>24000){notice("최대 4,000자의 텍스트 파일을 선택해 주세요.");return;}const text=await file.text();if(text.length>4000){notice("글은 최대 4,000자까지 입력할 수 있습니다.");return;}$("essay").value=text;$("essay").dispatchEvent(new Event("input"));});
$("run-form").addEventListener("submit",async event=>{event.preventDefault();notice();$("run-button").disabled=true;try{const data=await api("/api/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:$("essay").value,question:$("question").value,max_steps:Number($("max-steps").value),candidates:Number($("candidates").value),offline_smoke:$("mode").value==="demo"})});state.activeId=data.id;await loadRun(data.id);await refreshHistory();}catch(error){notice(error.message);setBusy();}});
$("cancel-button").addEventListener("click",async()=>{try{$("cancel-button").disabled=true;await api(`/api/runs/${state.id}/cancel`,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});}catch(error){notice(error.message);}finally{$("cancel-button").disabled=false;}});
$("history").addEventListener("change",event=>loadRun(event.target.value));
$("history-refresh").addEventListener("click",()=>refreshHistory().catch(error=>notice(error.message)));
$("active-button").addEventListener("click",()=>loadRun(state.activeId));
document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{state.view=button.dataset.view;document.querySelectorAll("[data-view]").forEach(x=>x.setAttribute("aria-selected",String(x===button)));renderPaper();}));
$("return-button").addEventListener("click",()=>{state.selection=null;renderPaper();renderCandidates();});
$("candidates-list").addEventListener("click",event=>{const button=event.target.closest("[data-candidate]");if(!button)return;const [step,index]=button.dataset.candidate.split(":").map(Number);state.selection={type:"candidate",step,index};renderPaper();renderCandidates();$("essay-view").scrollIntoView({behavior:"smooth",block:"center"});});
$("timeline").addEventListener("click",event=>{const button=event.target.closest("[data-checkpoint]");if(!button)return;state.selection={type:"checkpoint",id:Number(button.dataset.checkpoint)};renderPaper();renderCandidates();});
$("copy-button").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(selectedSnapshot().text);$("copy-button").textContent="복사됨 ✓";setTimeout(()=>$("copy-button").textContent="복사",1500);}catch{notice("브라우저에서 복사 권한을 확인해 주세요.");}});
$("download-button").addEventListener("click",()=>{if(!state.result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.result,null,2)],{type:"application/json"}));const link=document.createElement("a");link.href=url;link.download=`feak-${state.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
setInterval(updateElapsed,1000);
(async()=>{try{state.meta=await api("/api/meta");$("max-steps").value=state.meta.max_steps;$("candidates").value=state.meta.candidates;const data=await refreshHistory();const saved=localStorage.getItem("feak_run_id");const id=data.active_id || (data.runs.some(x=>x.id===saved)?saved:null);if(id)await loadRun(id);else renderStatus();}catch(error){notice("웹 서버에 연결하지 못했습니다. "+error.message);}})();
