"use strict";

const $ = id => document.getElementById(id);
const criteria = [
  ["goal_achievement", "목표 달성", "수정 목표에 해당하는 문제가 실제로 해결되었나요?"],
  ["necessity", "수정 필요성", "원래 글에 고칠 필요가 있었나요? 이미 충분한 내용을 불필요하게 바꾸지는 않았나요?"],
  ["preservation", "의미 보존", "목표와 관계없는 유효한 주장·근거·조건과 작성자의 입장을 유지했나요?"],
  ["global_benefit", "전체 글의 이익", "문장 연결·중복·문체 등에 새로운 문제를 만들거나 중요한 내용을 삭제하지 않았나요?"],
];
const S = {token: "", session: null, index: -1, packet: null, review: null, dirty: false,
  serial: 0, inFlight: null, timer: null, busy: false, conflict: false};
let toastTimer;

function buildCriteria() {
  criteria.forEach(([key, title, description], index) => {
    const field = document.createElement("fieldset"); field.className = "criterion";
    const legend = document.createElement("legend");
    const number = document.createElement("span"); number.className = "criterion-number"; number.textContent = `0${index+1}`;
    legend.append(number, document.createTextNode(title)); field.append(legend);
    const desc = document.createElement("p"); desc.className = "criterion-description"; desc.textContent = description; field.append(desc);
    const choices = document.createElement("div"); choices.className = "choices";
    ["PASS", "FAIL", "UNCERTAIN"].forEach(value => {
      const label = document.createElement("label"); label.className = "choice";
      const input = document.createElement("input"); input.type = "radio"; input.name = key; input.value = value; input.required = true;
      input.setAttribute("aria-label", `${title}: ${value}`);
      const span = document.createElement("span"); span.textContent = value; label.append(input,span); choices.append(label);
    });
    field.append(choices);
    const label = document.createElement("label"); label.className = "reason-label"; label.htmlFor = `${key}-reason`; label.textContent = "판단 근거";
    const reason = document.createElement("textarea"); reason.id = `${key}-reason`; reason.rows = 2; reason.maxLength = 3000; reason.required = true;
    reason.placeholder = "글의 구체적인 표현을 바탕으로 이유를 적어 주세요.";
    field.append(label,reason); $("criteria-grid").append(field);
  });
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {...options, headers: {"Authorization": `Bearer ${S.token}`,
      ...(options.body ? {"Content-Type": "application/json"} : {}), ...options.headers}});
  } catch (_) { throw new Error("서버에 연결하지 못했습니다. 작성 내용은 화면에 남아 있습니다. 연결 후 다시 저장해 주세요."); }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.error || "요청을 처리하지 못했습니다."); error.status = response.status; throw error;
  }
  return response;
}

function notice(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3200);
}
function showError(error) {
  $("error-message").textContent = error.message; $("error-banner").hidden = false;
  if (error.status === 409) S.conflict = true;
  $("reload-case").hidden = !S.conflict;
  setSaveStatus("저장하지 못했습니다", "failed");
}
function clearError() { $("error-banner").hidden = true; $("reload-case").hidden = true; }
function setSaveStatus(message, state = "") {
  $("save-status").textContent = message; $("save-indicator").className = `save-dot ${state}`;
}
function updateControls() {
  document.querySelectorAll(".case-nav, #previous, #next, #submit-review, #save-draft, #download, #logout").forEach(el => { el.disabled = S.busy; });
  document.querySelectorAll("#review-form input, #review-form textarea").forEach(el => { el.disabled = S.busy; });
  if (S.session) {
    $("previous").disabled = S.busy || S.index === 0;
    $("next").disabled = S.busy || S.index >= S.session.total - 1;
    $("submit-review").textContent = S.index === S.session.total - 1 ? "평가 저장 ✓" : "저장하고 다음 →";
  }
}
async function busy(action) {
  if (S.busy) return;
  S.busy = true; updateControls();
  try { await action(); } catch (error) { showError(error); }
  finally { S.busy = false; updateControls(); }
}

function renderNavigation() {
  const completed = S.session.cases.filter(c => c.status === "submitted").length;
  $("progress-text").textContent = `${completed} / ${S.session.total}`;
  $("progress").max = S.session.total; $("progress").value = completed;
  $("case-total").textContent = `${S.session.total}개`;
  $("progress-hint").textContent = `${S.session.total - completed}개 항목이 남아 있습니다.`;
  $("complete-banner").hidden = completed !== S.session.total;
  $("case-list").replaceChildren();
  S.session.cases.forEach((item, index) => {
    const button = document.createElement("button"); button.type = "button";
    button.className = `case-nav${index === S.index ? " active" : ""}${item.status === "submitted" ? " done" : ""}`;
    if (index === S.index) button.setAttribute("aria-current", "step");
    button.setAttribute("aria-label", `항목 ${index + 1}, ${{submitted:"평가 완료",draft:"임시 저장",empty:"미작성"}[item.status]}`);
    const title = document.createElement("span"); title.textContent = `항목 ${String(index + 1).padStart(2,"0")}`;
    const status = document.createElement("span"); status.textContent = {submitted:"완료 ✓",draft:"작성 중",empty:"·"}[item.status];
    button.append(title,status); button.addEventListener("click", () => navigate(index)); $("case-list").append(button);
  });
  updateControls();
}

function collect() {
  const ratings = {};
  for (const [key] of criteria) {
    const selected = document.querySelector(`input[name="${key}"]:checked`);
    ratings[key] = {label: selected ? selected.value : null, reason: $(`${key}-reason`).value};
  }
  return {ratings, notes: $("notes").value};
}

async function loadCase(index) {
  const data = await (await api(`/api/cases/${S.session.cases[index].case_id}`)).json();
  S.index = index; S.packet = data.case; S.review = data.review; S.dirty = false; S.conflict = false; S.serial++;
  clearError();
  $("case-title").textContent = `평가 항목 ${String(index+1).padStart(2,"0")}`;
  $("writing-prompt").textContent = data.case.writing_prompt; $("planner-goal").textContent = data.case.goal;
  $("preserve-list").replaceChildren();
  data.case.must_preserve.forEach(value => { const li = document.createElement("li"); li.textContent = value; $("preserve-list").append(li); });
  $("before-text").textContent = data.case.before; $("after-text").textContent = data.case.after;
  $("before-length").textContent = `${[...data.case.before].length}자`; $("after-length").textContent = `${[...data.case.after].length}자`;
  for (const [key] of criteria) {
    document.querySelectorAll(`input[name="${key}"]`).forEach(input => { input.checked = input.value === data.review.ratings[key].label; });
    $(`${key}-reason`).value = data.review.ratings[key].reason; $(`${key}-reason`).setCustomValidity("");
  }
  $("notes").value = data.review.notes;
  S.session.cases[index].status = data.review.status;
  showReviewState(); renderNavigation();
}
function showReviewState() {
  const status = S.review.status;
  $("case-state").textContent = {empty:"미작성",draft:"작성 중",submitted:"평가 완료"}[status];
  $("case-state").className = `state-tag ${status}`;
  setSaveStatus(status === "empty" ? "아직 작성하지 않았습니다" : status === "submitted" ? "평가를 저장했습니다" : "임시 저장했습니다", status === "empty" ? "" : "saved");
}

async function persist(status = "draft") {
  clearTimeout(S.timer);
  while (S.inFlight) await S.inFlight;
  if (S.conflict) throw Object.assign(new Error("다른 창의 저장과 충돌했습니다. 메모를 복사한 뒤 현재 항목을 다시 불러와 주세요."), {status:409});
  if (!S.dirty && status === "draft") return;
  const serial = S.serial, index = S.index;
  const body = {expected_version:S.review.version, status, ...collect()};
  setSaveStatus("저장 중…", "dirty");
  S.inFlight = (async () => {
    const data = await (await api(`/api/cases/${S.packet.case_id}/review`, {method:"PUT", body:JSON.stringify(body)})).json();
    S.review = data.review;
    S.session.cases[index].status = data.review.status;
    S.dirty = S.serial !== serial;
    showReviewState(); renderNavigation(); clearError();
    if (S.dirty) { setSaveStatus("변경 내용 저장 대기", "dirty"); scheduleSave(); }
  })();
  try { await S.inFlight; } finally { S.inFlight = null; }
}
function scheduleSave() {
  clearTimeout(S.timer);
  S.timer = setTimeout(() => { if (!S.busy && !S.conflict) persist().catch(showError); }, 900);
}
function changed() {
  if (!S.packet) return;
  S.dirty = true; S.serial++;
  for (const [key] of criteria) $(`${key}-reason`).setCustomValidity("");
  setSaveStatus("변경 내용 저장 대기", "dirty"); scheduleSave();
}
function validate() {
  for (const [key] of criteria) {
    const el = $(`${key}-reason`); el.setCustomValidity(el.value.trim() ? "" : "판단 근거를 작성해 주세요.");
  }
  return $("review-form").reportValidity();
}
async function navigate(index) {
  if (index === S.index || index < 0 || index >= S.session.total) return;
  await busy(async () => { await persist(); await loadCase(index); $("case-title").focus(); window.scrollTo({top:0, behavior:"instant"}); });
}

async function login(token) {
  $("login-error").textContent = ""; $("login-button").disabled = true; S.token = token.trim();
  try {
    S.session = await (await api("/api/session")).json();
    sessionStorage.setItem("feak-review-token", S.token);
    $("study-title").textContent = S.session.title; $("rater-label").textContent = S.session.rater_label;
    const next = S.session.cases.findIndex(c => c.status !== "submitted");
    await loadCase(next === -1 ? 0 : next);
    $("login").hidden = true; $("workspace").hidden = false; $("access-code").value = "";
  } catch (error) { $("login-error").textContent = error.message; }
  finally { $("login-button").disabled = false; }
}

buildCriteria();
$("login-form").addEventListener("submit", event => { event.preventDefault(); login($("access-code").value); });
$("review-form").addEventListener("input", changed);
$("review-form").addEventListener("submit", event => event.preventDefault());
$("previous").addEventListener("click", () => navigate(S.index-1));
$("next").addEventListener("click", () => navigate(S.index+1));
$("save-draft").addEventListener("click", () => busy(async () => { await persist(); notice("임시 저장했습니다."); }));
$("submit-review").addEventListener("click", () => {
  if (!validate()) return;
  busy(async () => { await persist("submitted"); notice("평가를 저장했습니다.");
    if (S.index < S.session.total-1) { await loadCase(S.index+1); $("case-title").focus(); window.scrollTo({top:0,behavior:"instant"}); }
  });
});
$("reload-case").addEventListener("click", () => {
  if (window.confirm("화면의 미저장 내용을 내려놓고 서버에 저장된 평가를 불러올까요? 필요한 메모는 먼저 복사해 주세요.")) {
    busy(async () => { clearTimeout(S.timer); await loadCase(S.index); });
  }
});
$("download").addEventListener("click", () => busy(async () => {
  await persist(); const blob = await (await api("/api/export")).blob(); const url = URL.createObjectURL(blob);
  const link = document.createElement("a"); link.href = url; link.download = "my-reviews.jsonl"; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url),1000); notice("현재까지 저장된 내 평가를 내보냈습니다.");
}));
$("logout").addEventListener("click", () => busy(async () => {
  await persist(); clearTimeout(S.timer); sessionStorage.removeItem("feak-review-token");
  S.token = ""; S.session = null; S.packet = null; S.review = null; S.index = -1; S.dirty = false;
  $("review-form").reset(); $("case-list").replaceChildren(); $("preserve-list").replaceChildren();
  ["before-text","after-text","writing-prompt","planner-goal","study-title","rater-label"].forEach(id => { $(id).textContent = ""; });
  $("workspace").hidden = true; $("login").hidden = false; $("access-code").focus();
}));
window.addEventListener("beforeunload", event => { if (S.dirty || S.inFlight) { event.preventDefault(); event.returnValue = ""; } });
const params = new URLSearchParams(location.hash.slice(1));
const initialToken = params.get("token") || sessionStorage.getItem("feak-review-token");
if (location.hash) history.replaceState(null,"",location.pathname + location.search);
if (initialToken) login(initialToken);
