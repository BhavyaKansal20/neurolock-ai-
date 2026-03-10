/**
 * NeuroLock AI v2 — Main Application
 * WebSocket client + All UI logic
 */

// ── Constants ────────────────────────────────────────────────
const EMOTIONS = ['happy','sad','angry','fearful','disgusted','surprised','neutral'];
const E_COLORS = {
  happy:'#fbbf24', sad:'#60a5fa', angry:'#fb7185',
  fearful:'#a78bfa', disgusted:'#34d399', surprised:'#f97316', neutral:'#64748b'
};

// ── State ─────────────────────────────────────────────────────
let socket         = null;
let currentView    = 'standard';
let cameraStream   = null;
let isRunning      = false;
let frameTimer     = null;
let gradcamOn      = false;
let frameCount     = 0;
let fpsStart       = Date.now();
let fpsFrames      = 0;
let sessionStart   = null;
let sessionTimer   = null;
let detectionCount = 0;

// Timeline data
const TIMELINE_MAX = 60;
let timelineData   = [];

// Classroom
let clsStream      = null;
let clsRunning     = false;
let clsTimer       = null;
let activeSession  = null;
let detectedStudents = {};  // student_id → {name, emo, conf, last_seen}

// ── DOM helpers ───────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Init ──────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', init);

async function init() {
  initBg();
  buildEmotionBars();
  drawGauge(0, '#38bdf8');

  setLoader(20, 'Connecting to NeuroLock server...');

  await connectSocket();
  await sleep(300);

  setLoader(70, 'Loading interface...');
  await sleep(300);

  setLoader(100, 'Ready');
  await sleep(400);
  $('loader').classList.add('out');
  setTimeout(() => $('loader').style.display = 'none', 500);

  // Camera source change
  $('cam-source').addEventListener('change', onCamSourceChange);
  $('cls-cam-src').addEventListener('change', onClsCamSrcChange);

  // Load students list
  loadStudents();
  loadSessions();

  // Session timer
  setInterval(() => {
    if (sessionStart) {
      const s = Math.floor((Date.now() - sessionStart) / 1000);
      const m = Math.floor(s/60), ss = s%60;
      $('st-session').textContent = `${String(m).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
    }
  }, 1000);
}

// ── WebSocket ─────────────────────────────────────────────────
async function connectSocket() {
  return new Promise(resolve => {
    socket = io(window.location.origin, { transports: ['websocket'], reconnection: true });

    socket.on('connect', () => {
      setConnStatus(true);
      setStatus('ready', 'Server connected');
      resolve();
    });

    socket.on('disconnect', () => {
      setConnStatus(false);
      setStatus('error', 'Disconnected from server');
    });

    socket.on('server_info', data => {
      $('m-model').textContent  = data.models + ' model(s)';
      $('m-students').textContent = data.students;
      setStatus('ready', `Server online — ${data.models} model(s) loaded`);
    });

    socket.on('detection_result', onDetectionResult);
    socket.on('session_started', onSessionStarted);
    socket.on('session_ended', onSessionEnded);
    socket.on('phase_changed', data => setPhaseUI(data.phase));
    socket.on('error', data => showToast('Server: ' + data.message));

    // Timeout after 4s
    setTimeout(resolve, 4000);
  });
}

// ── View switching ────────────────────────────────────────────
window.switchView = function(view) {
  currentView = view;
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  $('view-' + view).classList.add('active');
  document.querySelector(`[data-view="${view}"]`).classList.add('active');

  if (view === 'students') loadStudents();
  if (view === 'analytics') loadSessions();
};

// ── Standard mode ─────────────────────────────────────────────
function onCamSourceChange() {
  const val = $('cam-source').value;
  $('url-input-wrap').style.display = (val === 'url') ? 'flex' : 'none';
  if (val === 'upload') document.getElementById('file-input').click();
}

window.handleStart = async function() {
  const src = $('cam-source').value;
  if (src === 'upload') { $('file-input').click(); return; }

  setStatus('busy', 'Starting camera...');
  try {
    const constraint = src === 'url'
      ? { video: { facingMode: 'user' } }
      : { video: { deviceId: src === '0' ? undefined : { exact: src },
                   width: {ideal:1280}, height: {ideal:720} } };

    cameraStream = await navigator.mediaDevices.getUserMedia(constraint);
    $('video').srcObject = cameraStream;
    $('video').style.display = 'block';
    $('feed-placeholder').style.display = 'none';
    $('start-btn').style.display = 'none';
    $('stop-btn').style.display  = 'inline-flex';
    $('live-pill').classList.add('show');

    isRunning = true;
    sessionStart = Date.now();
    frameCount = 0; detectionCount = 0;
    startFrameLoop($('video'), $('det-canvas'), false);
    setStatus('ready', 'Live detection active');
  } catch (err) {
    setStatus('error', 'Camera error: ' + err.message);
    showToast('Camera access denied');
  }
};

window.handleStop = function() {
  isRunning = false;
  if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
  if (cameraStream) { cameraStream.getTracks().forEach(t=>t.stop()); cameraStream = null; }
  $('video').srcObject = null;
  $('video').style.display = 'none';
  $('feed-placeholder').style.display = 'flex';
  $('start-btn').style.display = 'inline-flex';
  $('stop-btn').style.display  = 'none';
  $('live-pill').classList.remove('show');
  $('m-fps').textContent = '—';
  clearCanvas('det-canvas');
  sessionStart = null;
  setStatus('ready', 'Camera stopped');
};

window.handleUpload = function(e) {
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = async ev => {
    $('upload-img').src = ev.target.result;
    $('upload-img').onload = async () => {
      $('feed-placeholder').style.display = 'none';
      $('upload-wrap').style.display = 'flex';
      setStatus('busy', 'Analyzing image...');
      socket.emit('frame', { image: ev.target.result, gradcam: gradcamOn });
    };
  };
  reader.readAsDataURL(file);
  e.target.value = '';
};

window.toggleGradcam = function() {
  gradcamOn = !gradcamOn;
  $('gradcam-btn').innerHTML = `
    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/>
    <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>
    <line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg>
    Grad-CAM: ${gradcamOn ? 'ON' : 'OFF'}
  `;
  $('gradcam-btn').style.color = gradcamOn ? 'var(--accent)' : '';
  $('gradcam-btn').style.borderColor = gradcamOn ? 'var(--border-hi)' : '';
};

window.captureSnapshot = function() {
  if (!isRunning) { showToast('Start camera first'); return; }
  const c = document.createElement('canvas');
  c.width = $('video').videoWidth; c.height = $('video').videoHeight;
  c.getContext('2d').drawImage($('video'), 0, 0);
  const a = document.createElement('a');
  a.download = `neurolock_${Date.now()}.png`;
  a.href = c.toDataURL();
  a.click();
  showToast('Snapshot saved!');
};

// ── Frame loop ────────────────────────────────────────────────
function startFrameLoop(videoEl, canvas, isClassroom) {
  const INTERVAL = 200; // 5fps to server
  const timer = setInterval(async () => {
    if (videoEl.readyState < 2) return;
    const b64 = captureFrame(videoEl);
    frameCount++;
    fpsFrames++;
    $('st-frames').textContent = frameCount;
    const now = Date.now();
    if (now - fpsStart >= 1000) {
      const fps = Math.round(fpsFrames * 1000 / (now - fpsStart));
      $('m-fps').textContent = fps;
      $('fps-badge').textContent = fps + ' fps';
      fpsFrames = 0; fpsStart = Date.now();
    }
    if (isClassroom) {
      socket.emit('frame', { image: b64, gradcam: false });
    } else {
      socket.emit('frame', { image: b64, gradcam: gradcamOn });
    }
  }, INTERVAL);

  if (isClassroom) clsTimer = timer;
  else frameTimer = timer;
}

function captureFrame(video) {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 640;
  c.height = video.videoHeight || 480;
  c.getContext('2d').drawImage(video, 0, 0);
  return c.toDataURL('image/jpeg', 0.75);
}

// ── Detection results ─────────────────────────────────────────
function onDetectionResult(result) {
  const faces = result.faces || [];
  $('st-faces').textContent  = faces.length;
  $('cls-face-count').textContent = faces.length + ' student' + (faces.length !== 1 ? 's' : '') + ' detected';

  // Draw on correct canvas
  const video  = clsRunning ? $('cls-video')  : $('video');
  const canvas = clsRunning ? $('cls-canvas') : $('det-canvas');

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (faces.length === 0) {
    clearEmotionBars(); return;
  }

  faces.forEach((face, i) => drawFaceBox(ctx, face, i));

  const main = faces[0];
  updateEmotionBars(main.emotions);
  updateDominant(main.dominant, main.confidence);

  // Timeline
  addTimelinePoint(main.dominant);

  detectionCount += faces.length;
  $('st-det').textContent = detectionCount;

  // Classroom student cards
  if (clsRunning) updateStudentCards(faces);

  // Grad-CAM overlay
  if (main.gradcam_b64 && !clsRunning) {
    const box = main.box;
    const img = new Image();
    img.onload = () => {
      ctx.save();
      ctx.globalAlpha = 0.6;
      ctx.drawImage(img, box.x, box.y, box.w, box.h);
      ctx.restore();
    };
    img.src = 'data:image/jpeg;base64,' + main.gradcam_b64;
  }
}

function drawFaceBox(ctx, face, index) {
  const {x, y, w, h} = face.box;
  const color = E_COLORS[face.dominant] || '#38bdf8';

  ctx.save();
  ctx.fillStyle = hexAlpha(color, 0.06);
  ctx.fillRect(x, y, w, h);

  // Corner brackets
  const cs = Math.min(w, h) * 0.18;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  [[x+cs,y,x,y,x,y+cs],[x+w-cs,y,x+w,y,x+w,y+cs],
   [x,y+h-cs,x,y+h,x+cs,y+h],[x+w-cs,y+h,x+w,y+h,x+w,y+h-cs]
  ].forEach(([ax,ay,bx,by,cx,cy]) => {
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.lineTo(cx,cy); ctx.stroke();
  });

  // Label
  ctx.shadowBlur = 0;
  const name = face.is_registered ? face.student_name : face.dominant.toUpperCase();
  const conf = Math.round(face.confidence * 100) + '%';
  ctx.font = 'bold 11px "DM Mono",monospace';
  const tw = ctx.measureText(name + '  ' + conf).width;
  const lx = x, ly = y > 22 ? y - 22 : y + h + 4;
  ctx.fillStyle = hexAlpha(color, 0.85);
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(lx, ly, tw+10, 18, 3);
  else ctx.rect(lx, ly, tw+10, 18);
  ctx.fill();
  ctx.fillStyle = '#000';
  ctx.fillText(name + '  ' + conf, lx+5, ly+13);
  ctx.restore();
}

// ── Classroom ─────────────────────────────────────────────────
function onClsCamSrcChange() {
  const val = $('cls-cam-src').value;
  $('cls-url-wrap').style.display = (val === 'url') ? 'flex' : 'none';
}

window.toggleClassroomCamera = async function() {
  if (clsRunning) {
    clsRunning = false;
    if (clsTimer) { clearInterval(clsTimer); clsTimer = null; }
    if (clsStream) { clsStream.getTracks().forEach(t=>t.stop()); clsStream = null; }
    $('cls-video').srcObject = null;
    $('cls-video').style.display = 'none';
    $('cls-placeholder').style.display = 'flex';
    clearCanvas('cls-canvas');
    $('cls-cam-btn').innerHTML = '<svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Camera';
    return;
  }

  try {
    const src = $('cls-cam-src').value;
    clsStream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: src === '0' ? undefined : {exact: src},
               width: {ideal:1280}, height: {ideal:720} }
    });
    $('cls-video').srcObject = clsStream;
    $('cls-video').style.display = 'block';
    $('cls-placeholder').style.display = 'none';
    clsRunning = true;
    startFrameLoop($('cls-video'), $('cls-canvas'), true);
    $('cls-cam-btn').innerHTML = '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/></svg> Stop Camera';
  } catch (err) {
    showToast('Camera error: ' + err.message);
  }
};

// ── Session management ────────────────────────────────────────
window.openSessionModal = function() {
  $('session-modal').style.display = 'flex';
};

window.createSession = async function() {
  const name     = $('sess-input-name').value || 'Session ' + new Date().toLocaleTimeString();
  const teacher  = $('sess-input-teacher').value;
  const subject  = $('sess-input-subject').value;
  const location = $('sess-input-location').value;

  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({name, teacher, subject, location})
  });
  const data = await res.json();
  if (data.success) {
    activeSession = data;
    closeModal('session-modal');
    showToast('Session started!');
  }
};

function onSessionStarted(data) {
  activeSession = data;
  $('session-bar').style.display = 'flex';
  $('sess-name').textContent = data.name;
  $('sess-meta').textContent = 'Session ID: ' + data.session_id;
  $('cls-start-btn').style.display = 'none';
  setPhaseUI('before');
}

window.setPhase = function(phase) {
  socket.emit('set_phase', {phase});
  setPhaseUI(phase);
};

function setPhaseUI(phase) {
  ['before','during','after'].forEach(p => {
    $('phase-' + p).classList.toggle('active', p === phase);
  });
}

window.endSession = async function() {
  if (!confirm('End this session and generate report?')) return;
  const res = await fetch('/api/sessions/end', {method:'POST'});
  const data = await res.json();
  if (data.success) {
    showToast('Session ended — generating report...');
    loadSessions();
    switchView('analytics');
  }
};

function onSessionEnded(data) {
  activeSession = null;
  $('session-bar').style.display = 'none';
  $('cls-start-btn').style.display = 'inline-flex';
  detectedStudents = {};
  $('student-cards-grid').innerHTML = '<div class="empty-students"><svg viewBox="0 0 24 24" width="32" height="32"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><p>No students detected yet</p></div>';
}

// ── Student cards (classroom) ─────────────────────────────────
function updateStudentCards(faces) {
  faces.forEach(face => {
    const sid = face.student_id;
    detectedStudents[sid] = {
      name:     face.student_name,
      dominant: face.dominant,
      conf:     face.confidence,
      reg:      face.is_registered,
      last:     Date.now(),
    };
  });

  const grid = $('student-cards-grid');
  const entries = Object.entries(detectedStudents);
  $('cls-registered-count').textContent = entries.filter(([,v])=>v.reg).length + ' registered';

  if (entries.length === 0) {
    grid.innerHTML = '<div class="empty-students"><svg viewBox="0 0 24 24" width="32" height="32"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><p>No students detected</p></div>';
    return;
  }

  grid.innerHTML = entries.map(([sid, s]) => {
    const color = E_COLORS[s.dominant] || '#64748b';
    return `<div class="student-face-card detected">
      <div class="student-face-placeholder"><svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="1.5"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>
      <div class="student-info">
        <div class="student-face-name">${s.name}</div>
        <div class="student-face-emo" style="color:${color}">${s.dominant} · ${Math.round(s.conf*100)}%</div>
        <div class="student-face-conf">${s.reg ? '✓ Registered' : 'Unknown'}</div>
      </div>
    </div>`;
  }).join('');

  // Update engagement chart
  updateEngagementChart();
}

// ── Students view ─────────────────────────────────────────────
async function loadStudents() {
  const res = await fetch('/api/students');
  const students = await res.json();
  const grid = $('student-list-grid');

  $('m-students').textContent = students.length;

  if (students.length === 0) {
    grid.innerHTML = '<div class="empty-card" style="grid-column:1/-1"><svg viewBox="0 0 24 24" width="40" height="40" stroke="currentColor" fill="none" stroke-width="1.2" stroke-linecap="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><p>No students registered yet.<br>Click Register Student to add.</p></div>';
    return;
  }

  grid.innerHTML = students.map(s => `
    <div class="student-card">
      ${s.image_b64
        ? `<img class="student-card-img" src="data:image/jpeg;base64,${s.image_b64}" alt="${s.name}"/>`
        : `<div class="student-card-img" style="background:var(--raised);display:grid;place-items:center"><svg viewBox="0 0 24 24" width="28" height="28" stroke="var(--text-lo)" fill="none" stroke-width="1.2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`
      }
      <div class="student-card-name">${s.name}</div>
      <div class="student-card-meta">${s.id}${s.class_name ? ' · ' + s.class_name : ''}${s.roll_no ? ' · Roll ' + s.roll_no : ''}</div>
      <div class="student-card-actions">
        <button class="btn-danger btn-sm" onclick="deleteStudent('${s.id}','${s.name}')">Remove</button>
      </div>
    </div>
  `).join('');
}

window.openRegisterModal = function() { $('register-modal').style.display = 'flex'; };

window.previewRegPhoto = function(e) {
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    $('reg-photo-preview').src = ev.target.result;
    $('reg-photo-preview').style.display = 'block';
    $('photo-upload-hint').style.display = 'none';
  };
  reader.readAsDataURL(file);
};

window.registerStudent = async function() {
  const id       = $('reg-id').value.trim();
  const name     = $('reg-name').value.trim();
  const cls      = $('reg-class').value.trim();
  const roll     = $('reg-roll').value.trim();
  const photoSrc = $('reg-photo-preview').src;

  if (!name) { showToast('Name is required'); return; }
  if (!photoSrc || !photoSrc.startsWith('data:')) { showToast('Photo is required'); return; }

  const res = await fetch('/api/students', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({id: id || undefined, name, class_name: cls, roll_no: roll, image: photoSrc})
  });
  const data = await res.json();
  if (data.success) {
    showToast(`${name} registered successfully!`);
    closeModal('register-modal');
    // Reset form
    $('reg-id').value = ''; $('reg-name').value = '';
    $('reg-class').value = ''; $('reg-roll').value = '';
    $('reg-photo-preview').style.display = 'none';
    $('photo-upload-hint').style.display = 'flex';
    loadStudents();
  } else {
    showToast('Error: ' + data.error);
  }
};

window.deleteStudent = async function(id, name) {
  if (!confirm(`Remove ${name}?`)) return;
  await fetch('/api/students/' + id, {method:'DELETE'});
  showToast(name + ' removed');
  loadStudents();
};

// ── Sessions / Analytics ──────────────────────────────────────
async function loadSessions() {
  const res = await fetch('/api/sessions');
  const sessions = await res.json();
  const list = $('sessions-list');

  if (sessions.length === 0) {
    list.innerHTML = '<div class="empty-card"><svg viewBox="0 0 24 24" width="40" height="40" stroke="currentColor" fill="none" stroke-width="1.2" stroke-linecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg><p>No sessions yet.</p></div>';
    return;
  }

  list.innerHTML = sessions.map(s => `
    <div class="session-row">
      <div>
        <div class="sess-row-name">${s.name}</div>
        <div class="sess-row-meta">${s.teacher ? s.teacher + ' · ' : ''}${s.subject || ''} · ${new Date(s.start_time).toLocaleString()}</div>
      </div>
      <div class="sess-row-badges">
        <span class="badge ${s.status}">${s.status}</span>
        ${s.status === 'completed'
          ? `<button class="btn-outline btn-sm" onclick="viewReport('${s.id}')">View Report</button>`
          : ''}
      </div>
    </div>
  `).join('');
}

window.viewReport = async function(sessionId) {
  const res = await fetch('/api/sessions/' + sessionId + '/report');
  const data = await res.json();
  if (!data.report) { showToast('Report not found'); return; }
  renderReport(data.report);
  $('report-modal').style.display = 'flex';
};

function renderReport(report) {
  const meta = report.meta || {};
  const cs   = report.class_summary || {};
  const students = report.students || {};

  $('report-body').innerHTML = `
    <div class="report-section">
      <h3>Session Overview</h3>
      <div style="margin-bottom:10px;font-family:'DM Mono',monospace;font-size:.7rem;color:var(--text-lo)">
        ${meta.name} · ${meta.teacher || ''} · ${meta.subject || ''} · ${new Date(meta.start_time).toLocaleString()}
      </div>
      <div class="report-grid">
        <div class="report-metric"><div class="val">${cs.total_students || 0}</div><div class="lbl">Students</div></div>
        <div class="report-metric"><div class="val" style="color:var(--green)">${cs.avg_engagement || 0}%</div><div class="lbl">Avg Engagement</div></div>
        <div class="report-metric"><div class="val" style="color:var(--accent)">${cs.avg_comprehension || 0}%</div><div class="lbl">Avg Comprehension</div></div>
      </div>
    </div>
    <div class="report-section">
      <h3>Student Performance</h3>
      ${Object.values(students).map(s => {
        const eng  = s.engagement?.overall || 0;
        const comp = s.comprehension?.score || 0;
        const trend = s.trend || 'stable';
        return `<div class="student-report-row">
          <div class="srow-img" style="background:var(--raised);display:grid;place-items:center;border-radius:50%">
            <svg viewBox="0 0 24 24" width="20" height="20" stroke="var(--text-lo)" fill="none" stroke-width="1.2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          </div>
          <div class="srow-info">
            <div class="srow-name">${s.student_id}</div>
            <div class="srow-trend trend-${trend}">Trend: ${trend} · ${s.total_readings || 0} readings</div>
            <div style="font-size:.72rem;color:var(--text-lo);margin-top:3px">${s.recommendation || ''}</div>
          </div>
          <div class="srow-score">
            <div class="num" style="color:var(--green)">${eng}%</div>
            <div class="sub">Engaged</div>
          </div>
          <div class="srow-score">
            <div class="num" style="color:var(--accent)">${comp}%</div>
            <div class="sub">Comprehend</div>
          </div>
        </div>`;
      }).join('')}
    </div>
  `;
}

window.printReport = function() { window.print(); };

// ── Emotion UI ────────────────────────────────────────────────
function buildEmotionBars() {
  const container = $('emotion-bars');
  container.innerHTML = EMOTIONS.map(e => `
    <div class="emo-row">
      <div class="emo-lbl" id="lbl-${e}">${e.charAt(0).toUpperCase()+e.slice(1)}</div>
      <div class="emo-track"><div class="emo-fill fill-${e}" id="bar-${e}"></div></div>
      <div class="emo-pct" id="pct-${e}">0%</div>
    </div>
  `).join('');
}

function updateEmotionBars(emotions) {
  let dom = '', max = 0;
  EMOTIONS.forEach(e => { if ((emotions[e]||0) > max) { max = emotions[e]; dom = e; }});
  EMOTIONS.forEach(e => {
    const v = Math.round((emotions[e]||0) * 100);
    $('bar-' + e).style.width = v + '%';
    $('pct-' + e).textContent = v + '%';
    $('lbl-' + e).classList.toggle('active', e === dom);
    $('pct-' + e).classList.toggle('active', e === dom);
  });
}

function clearEmotionBars() {
  EMOTIONS.forEach(e => {
    $('bar-'+e).style.width = '0%';
    $('pct-'+e).textContent = '0%';
    $('lbl-'+e).classList.remove('active');
    $('pct-'+e).classList.remove('active');
  });
}

function updateDominant(emotion, confidence) {
  const color = E_COLORS[emotion] || '#38bdf8';
  const el = $('dominant-emo');
  el.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
  el.style.color = color;
  $('dominant-conf').textContent = Math.round(confidence*100) + '% confidence';
  drawGauge(confidence, color);
}

// ── Gauge ─────────────────────────────────────────────────────
function drawGauge(value, color) {
  const c = $('gauge-canvas'); if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width = 100, H = c.height = 56;
  const cx = W/2, cy = H-4, r = 38;
  ctx.clearRect(0, 0, W, H);
  ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0);
  ctx.strokeStyle = 'rgba(255,255,255,.05)'; ctx.lineWidth = 5; ctx.lineCap = 'round'; ctx.stroke();
  if (value > 0) {
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, Math.PI + value * Math.PI);
    ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 8; ctx.stroke();
  }
  ctx.shadowBlur = 0;
  ctx.fillStyle = 'rgba(226,232,240,.85)';
  ctx.font = 'bold 13px "DM Mono",monospace';
  ctx.textAlign = 'center';
  ctx.fillText(Math.round(value * 100) + '%', cx, cy - 5);
}

// ── Timeline chart ────────────────────────────────────────────
function addTimelinePoint(emotion) {
  timelineData.push({ emotion, time: Date.now(), color: E_COLORS[emotion] || '#64748b' });
  if (timelineData.length > TIMELINE_MAX) timelineData.shift();
  drawTimeline();
}

window.clearTimeline = function() { timelineData = []; drawTimeline(); };

function drawTimeline() {
  const c = $('timeline-canvas'); if (!c) return;
  const ctx = c.getContext('2d');
  c.width = c.offsetWidth || 260; const W = c.width, H = 80;
  ctx.clearRect(0, 0, W, H);

  if (timelineData.length < 2) return;

  const step = W / (TIMELINE_MAX - 1);
  const emotionToY = e => {
    const idx = EMOTIONS.indexOf(e);
    return H - 10 - (idx / (EMOTIONS.length - 1)) * (H - 20);
  };

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.lineWidth = 1;
  EMOTIONS.forEach((e, i) => {
    const y = emotionToY(e);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  });

  // Emotion path
  ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.lineCap = 'round';

  // Draw path segments colored by emotion
  for (let i = 1; i < timelineData.length; i++) {
    const prev = timelineData[i-1], curr = timelineData[i];
    const x1 = (i-1) * step, y1 = emotionToY(prev.emotion);
    const x2 = i * step,     y2 = emotionToY(curr.emotion);
    ctx.strokeStyle = curr.color;
    ctx.shadowColor = curr.color; ctx.shadowBlur = 3;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }

  // Dots
  ctx.shadowBlur = 0;
  timelineData.forEach((d, i) => {
    const x = i * step, y = emotionToY(d.emotion);
    ctx.fillStyle = d.color;
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI*2); ctx.fill();
  });

  // Current label
  const last = timelineData[timelineData.length - 1];
  const ly = emotionToY(last.emotion);
  ctx.fillStyle = last.color;
  ctx.font = '9px "DM Mono",monospace';
  ctx.fillText(last.emotion, W - 48, ly - 4);
}

// ── Engagement chart ──────────────────────────────────────────
const engagementData = { before: [], during: [], after: [] };

function updateEngagementChart() {
  const total = Object.values(detectedStudents).length;
  if (total === 0) return;
  const engaged = Object.values(detectedStudents).filter(s =>
    ['happy', 'surprised', 'neutral'].includes(s.dominant)
  ).length;
  const pct = Math.round(engaged / total * 100);
  const phase = ['before','during','after'].find(p =>
    document.getElementById('phase-'+p)?.classList.contains('active')
  ) || 'during';
  engagementData[phase].push(pct);
  if (engagementData[phase].length > 30) engagementData[phase].shift();
  drawEngagementChart();
}

function drawEngagementChart() {
  const c = $('engagement-chart'); if (!c) return;
  const ctx = c.getContext('2d');
  c.width = c.offsetWidth || 600; const W = c.width, H = 100;
  ctx.clearRect(0, 0, W, H);

  const colors = { before: '#60a5fa', during: '#34d399', after: '#f97316' };
  const allData = Object.entries(engagementData);
  const maxLen = Math.max(...allData.map(([,d]) => d.length), 1);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.lineWidth = 1;
  [25, 50, 75].forEach(y => {
    const py = H - (y/100) * (H-16) - 4;
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.2)';
    ctx.font = '8px "DM Mono",monospace';
    ctx.fillText(y+'%', 2, py - 2);
  });

  allData.forEach(([phase, data]) => {
    if (data.length < 2) return;
    const step = W / (maxLen - 1);
    ctx.strokeStyle = colors[phase];
    ctx.shadowColor  = colors[phase]; ctx.shadowBlur = 4;
    ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = i * step;
      const y = H - (v/100) * (H-16) - 4;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
  });
}

// ── Helpers ───────────────────────────────────────────────────
window.closeModal = function(id) { $(id).style.display = 'none'; };

function setStatus(type, msg) {
  $('status-dot').className = 'status-dot ' + type;
  $('status-text').textContent = msg;
}

function setConnStatus(connected) {
  $('conn-dot').className = 'conn-indicator' + (connected ? ' connected' : ' error');
  $('conn-text').textContent = connected ? 'Connected' : 'Disconnected';
}

function setLoader(pct, msg) {
  $('loader-bar').style.width = pct + '%';
  $('loader-status').textContent = msg;
}

function clearCanvas(id) {
  const c = $(id); if (!c) return;
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showToast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

// ── Background animation ──────────────────────────────────────
function initBg() {
  const c = $('bg-canvas'), ctx = c.getContext('2d');
  let W, H, nodes = [];
  const N = 50, MAX_D = 130;
  const resize = () => {
    W = c.width = window.innerWidth; H = c.height = window.innerHeight;
    nodes = Array.from({length:N}, () => ({
      x:Math.random()*W, y:Math.random()*H,
      vx:(Math.random()-.5)*.3, vy:(Math.random()-.5)*.3, r:Math.random()*1.5+.5
    }));
  };
  const draw = () => {
    ctx.clearRect(0,0,W,H);
    nodes.forEach(n => {
      n.x+=n.vx; n.y+=n.vy;
      if(n.x<0||n.x>W)n.vx*=-1; if(n.y<0||n.y>H)n.vy*=-1;
    });
    for(let i=0;i<nodes.length;i++) for(let j=i+1;j<nodes.length;j++) {
      const dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
      const d=Math.sqrt(dx*dx+dy*dy);
      if(d<MAX_D) {
        ctx.strokeStyle=`rgba(56,189,248,${(1-d/MAX_D)*.15})`;
        ctx.lineWidth=.6; ctx.beginPath();
        ctx.moveTo(nodes[i].x,nodes[i].y); ctx.lineTo(nodes[j].x,nodes[j].y); ctx.stroke();
      }
    }
    nodes.forEach(n => {
      ctx.fillStyle='rgba(56,189,248,.3)';
      ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,Math.PI*2); ctx.fill();
    });
    requestAnimationFrame(draw);
  };
  window.addEventListener('resize', resize);
  resize(); draw();
}
