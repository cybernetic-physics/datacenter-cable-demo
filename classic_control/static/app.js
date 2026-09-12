const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
let socket;
let telemetry = null;
let uiConfig = null;
let grasps = {};
const handTouched = {left:false, right:false};

function toast(message) {
  const node = $("#toast"); node.textContent = message; node.classList.remove("hidden");
  clearTimeout(node.timer); node.timer = setTimeout(() => node.classList.add("hidden"), 3500);
}

function send(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return toast("Control connection is not open");
  socket.send(JSON.stringify(message));
}

function updateMotionButtons() { $$(".motion").forEach(button => button.disabled = !socket || socket.readyState !== WebSocket.OPEN); }
function selectedArm() { return $("input[name=arm]:checked").value; }
function n(value, digits=3) { return Number(value).toFixed(digits); }
function poseText(pose) { return pose ? `XYZ  ${pose.xyz.map(v=>n(v)).join("  ")}\nRPY  ${pose.rpy_deg.map(v=>n(v,1)).join("  ")}` : "—"; }

function setCameraStatus(text, kind="") {
  const status = $("#cameraStatus"); status.textContent = text; status.className = `pill ${kind}`;
}

async function startCamera() {
  const image = $("#headCamera"), message = $("#cameraMessage");
  image.removeAttribute("src"); image.classList.remove("visible");
  await new Promise(resolve => setTimeout(resolve, 250));
  message.classList.remove("hidden"); message.textContent = "Looking for the head camera…";
  setCameraStatus("Checking…");
  try {
    const response = await fetch("/api/cameras", {cache:"no-store"});
    const catalog = await response.json(), select = $("#cameraSelect");
    const previous = select.value || catalog.default; select.innerHTML = "";
    for (const [id,state] of Object.entries(catalog.sources)) {
      const option=document.createElement("option"); option.value=id;
      option.textContent=`${state.name}${state.available ? "" : " · offline"}`; select.append(option);
    }
    select.value = catalog.sources[previous] ? previous : catalog.default;
    const source = select.value, state = catalog.sources[source];
    if (!state.available) throw new Error(state.error || `${state.name} is unavailable`);
    setCameraStatus("Starting…");
    image.onload = () => setCameraStatus("Live", "good");
    image.onerror = () => { image.classList.remove("visible"); message.classList.remove("hidden"); message.textContent = "Camera stream stopped"; setCameraStatus("Offline", "bad"); };
    image.classList.add("visible"); message.classList.add("hidden");
    image.src = `/api/cameras/${encodeURIComponent(source)}.mjpg?t=${Date.now()}`;
  } catch (error) {
    message.textContent = error.message; setCameraStatus("Unavailable", "bad");
  }
}

function renderState(state) {
  telemetry = state;
  $("#connection").textContent = state.connected ? "Robot connected" : "Disconnected";
  $("#connection").className = `pill ${state.connected ? "good" : "bad"}`;
  $("#mode").textContent = state.acquired ? "Debug control" : "Read only";
  $("#activeCommand").textContent = state.active_command || "Idle";
  $("#fault").textContent = state.fault || ""; $("#fault").classList.toggle("hidden", !state.fault);
  if (state.arms) {
    $("#leftPose").textContent = poseText(state.arms.left.measured);
    $("#rightPose").textContent = poseText(state.arms.right.measured);
    $("#waistPose").textContent = `YAW  ${n(state.arms.waist_q[0])}\nROLL ${n(state.arms.waist_q[1])}\nPITCH ${n(state.arms.waist_q[2])}`;
  }
  if (state.hands) for (const side of ["left","right"]) {
    if (!state.hands[side]) continue;
    if (!handTouched[side]) { loadHand(side, state.hands[side]); handTouched[side] = true; }
    for (const [name,value] of Object.entries(state.hands[side])) {
      const readout = $(`#${side}-${name}-measured`); if (readout) readout.textContent = n(value);
    }
  }
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/api/control`);
  socket.onopen = () => { toast("Control session connected"); updateMotionButtons(); };
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === "telemetry") renderState(message.state);
    if (message.type === "error") toast(message.message);
  };
  socket.onclose = event => { updateMotionButtons(); toast(event.code === 4001 ? "Another browser owns control" : "Control connection closed"); };
}

function buildHand(side) {
  const root = $(`#${side}Hand`); root.innerHTML = "";
  for (const name of uiConfig.joint_names) {
    const [low, high] = uiConfig.dex3_limits[side][name];
    const row = document.createElement("div"); row.className = "joint";
    row.innerHTML = `<label for="${side}-${name}">${name.replaceAll("_"," ")}</label><input id="${side}-${name}" type="range" min="${low}" max="${high}" step="0.01" value="0"><span><b id="${side}-${name}-value">0.000</b><small id="${side}-${name}-measured"> —</small></span>`;
    root.append(row);
    row.querySelector("input").addEventListener("input", event => { handTouched[side]=true; $(`#${side}-${name}-value`).textContent = n(event.target.value); });
  }
}

function staged(side) { return Object.fromEntries(uiConfig.joint_names.map(name => [name, Number($(`#${side}-${name}`).value)])); }
function loadHand(side, values) { if (!values) return; for (const [name,value] of Object.entries(values)) { $(`#${side}-${name}`).value=value; $(`#${side}-${name}-value`).textContent=n(value); } }

async function refreshGrasps() {
  grasps = await fetch("/api/grasps").then(response => response.json());
  const select = $("#graspSelect"); const previous = select.value; select.innerHTML = "";
  for (const [name,grasp] of Object.entries(grasps)) { const option=document.createElement("option"); option.value=name; option.textContent=`${name} · ${Object.keys(grasp.hands).join("+")}`; select.append(option); }
  if (grasps[previous]) select.value=previous; select.dispatchEvent(new Event("change"));
}

function jog(axis, sign, coarse=false) {
  const mode=$("#jogMode").value; let delta=Number($("#jogStep").value)*sign*(coarse?3:1);
  if (mode === "rotation") delta *= Math.PI/180;
  send({type:"jog", side:selectedArm(), mode, axis, delta, duration_s:Number($("#jogDuration").value), elbow:$("#poseForm [name=elbow]").value});
}

async function initialize() {
  uiConfig = await fetch("/api/config").then(response => response.json());
  buildHand("left"); buildHand("right"); await refreshGrasps(); connect(); updateMotionButtons();
  startCamera();
  $("#jogMode").onchange = event => {
    const rotation=event.target.value==="rotation", input=$("#jogStep");
    $("#jogStepLabel").textContent=rotation?"Step (°)":"Step (m)";
    input.value=rotation?"3":"0.01"; input.min=rotation?"0.1":"0.001"; input.max=rotation?"15":"0.05"; input.step=rotation?"0.1":"0.001";
  };
  $("#normal").onclick = () => send({type:"normal", duration_s:20});
  $("#stop").onclick = () => send({type:"stop"});
  $("#release").onclick = () => send({type:"release"});
  $$("[data-axis]").forEach(button => button.onclick = event => jog(Number(button.dataset.axis),Number(button.dataset.sign),event.shiftKey));
  $$("[data-apply-hand]").forEach(button => button.onclick = () => { const side=button.dataset.applyHand; send({type:"hand",targets:{[side]:staged(side)},duration_s:Number($("#handDuration").value)}); });
  $("#poseForm").onsubmit = event => { event.preventDefault(); const form=new FormData(event.target); send({type:"pose",side:selectedArm(),xyz:["x","y","z"].map(k=>Number(form.get(k))),rpy_deg:["roll","pitch","yaw"].map(k=>Number(form.get(k))),duration_s:Number(form.get("duration")),elbow:form.get("elbow")}); };
  $("#copyPose").onclick = () => { if (!telemetry?.arms) return; const pose=telemetry.arms[selectedArm()].measured; const form=$("#poseForm"); ["x","y","z"].forEach((k,i)=>form.elements[k].value=pose.xyz[i]); ["roll","pitch","yaw"].forEach((k,i)=>form.elements[k].value=pose.rpy_deg[i]); };
  $("#graspSelect").onchange = event => { const grasp=grasps[event.target.value]; $("#graspDescription").textContent=grasp?.description||""; };
  $("#loadGrasp").onclick = () => { const grasp=grasps[$("#graspSelect").value]; if (!grasp) return; for (const side of ["left","right"]) loadHand(side,grasp.hands[side]); };
  $("#applyGrasp").onclick = () => send({type:"grasp",name:$("#graspSelect").value,sides:[...($("#applyLeft").checked?["left"]:[]),...($("#applyRight").checked?["right"]:[])],duration_s:Number($("#handDuration").value)});
  $("#refreshGrasps").onclick = refreshGrasps;
  $("#retryCamera").onclick = startCamera;
  $("#cameraSelect").onchange = startCamera;
  $("#saveGrasp").onsubmit = async event => { event.preventDefault(); const form=new FormData(event.target); const hands={}; if(form.get("left"))hands.left=staged("left"); if(form.get("right"))hands.right=staged("right"); const response=await fetch(`/api/grasps/${encodeURIComponent(form.get("name"))}`,{method:"PUT",headers:{"content-type":"application/json"},body:JSON.stringify({description:form.get("description"),hands})}); const body=await response.json(); if(!response.ok)return toast(body.detail); toast(`Saved ${body.saved}`); await refreshGrasps(); };
  $("#deleteGrasp").onclick = async () => { const name=$("#graspSelect").value; const response=await fetch(`/api/grasps/${encodeURIComponent(name)}`,{method:"DELETE"}); const body=await response.json(); if(!response.ok)return toast(body.detail); toast(`Deleted ${name}`); await refreshGrasps(); };
}

document.addEventListener("keydown", event => {
  if (event.code === "Escape") { send({type:"release"}); return; }
  if (event.repeat || event.target.matches("input,select,textarea")) return;
  const keys={w:[0,1],s:[0,-1],a:[1,1],d:[1,-1],r:[2,1],f:[2,-1]}; const command=keys[event.key.toLowerCase()]; if(command){event.preventDefault();jog(command[0],command[1],event.shiftKey);}
});
window.addEventListener("beforeunload", () => { $("#headCamera").removeAttribute("src"); socket?.close(); });
initialize().catch(error => toast(error.message));
