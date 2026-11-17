const $ = (id) => document.getElementById(id);
let transcriptId = null;
let revision = 0;
let history = [];
let summaryTimer;
let chatRevision = 0;

async function api(path, body, timeout = 90000) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeout),
    cache: 'no-store',
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check your input and try again.');
  return data;
}

function errorText(error) {
  if (error.name === 'TimeoutError') return 'This request took too long. Please try again.';
  if (error instanceof TypeError) return 'Cannot reach the backend. Check that the server is running.';
  return error.message;
}

async function checkConnection() {
  $('check').disabled = true;
  try {
    const data = await api('/api/health', undefined, 10000);
    $('status').textContent = data.ai_configured ? 'AI connected' : 'AI setup needed';
  } catch (error) {
    $('status').textContent = errorText(error);
  } finally {
    $('check').disabled = false;
  }
}
$('check').addEventListener('click', checkConnection);
checkConnection();

function timestamp(seconds) {
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const sec = String(total % 60).padStart(2, '0');
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${sec}` : `${minutes}:${sec}`;
}

function sourceLink(url, label) {
  const link = document.createElement('a');
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  link.textContent = label;
  return link;
}

async function requestSummary(currentRevision) {
  clearTimeout(summaryTimer);
  $('summary-retry').hidden = true;
  $('summary-status').textContent = 'Generating your video summary… You can ask a question while it is being prepared.';
  try {
    const state = await api('/api/summary', { transcript_id: transcriptId });
    if (currentRevision !== revision) return;
    displaySummary(state, currentRevision);
  } catch (error) {
    if (currentRevision !== revision) return;
    $('summary-status').textContent = errorText(error);
    $('summary-retry').hidden = false;
  }
}

function displaySummary(state, currentRevision) {
  if (state.status === 'complete') {
    $('summary-status').textContent = 'Summary based on the video transcript.';
    $('summary-text').textContent = state.summary;
  } else if (state.status === 'failed') {
    $('summary-status').textContent = state.error;
    $('summary-retry').hidden = false;
  } else if (state.status === 'pending') {
    const elapsed = state.elapsed_seconds || 0;
    $('summary-status').textContent = `${state.progress || 'Generating your summary'}… ${elapsed}s elapsed.${elapsed >= 30 ? ' Gemini is taking longer than usual. You can still chat below.' : ''}`;
    summaryTimer = setTimeout(async () => {
      try {
        const next = await api(`/api/summary/${transcriptId}`, undefined, 15000);
        if (currentRevision === revision) displaySummary(next, currentRevision);
      } catch (error) {
        if (currentRevision !== revision) return;
        $('summary-status').textContent = errorText(error);
        $('summary-retry').hidden = false;
      }
    }, 2000);
  } else {
    $('summary-status').textContent = 'The summary job could not be confirmed. Please retry.';
    $('summary-retry').hidden = false;
  }
}
$('summary-retry').addEventListener('click', () => requestSummary(revision));

$('video-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  revision += 1;
  const currentRevision = revision;
  clearTimeout(summaryTimer);
  transcriptId = null;
  history = [];
  chatRevision += 1;
  $('question').disabled = true;
  $('chat-button').disabled = true;
  $('load').disabled = true;
  $('transcript-panel').hidden = true;
  $('summary-status').textContent = 'Loading the video transcript…';
  for (const container of ['segments', 'chat-messages', 'summary-text']) $(container).replaceChildren();
  $('question').value = '';
  $('chat-status').textContent = '';
  $('transcript-status').textContent = 'Loading video captions…';
  try {
    const data = await api('/api/transcript', { url: $('video-url').value.trim() });
    if (currentRevision !== revision) return;
    transcriptId = data.transcript_id;
    const fragment = document.createDocumentFragment();
    for (const segment of data.segments) {
      const row = document.createElement('p');
      row.className = 'segment';
      const text = document.createElement('span');
      text.textContent = segment.text;
      row.append(sourceLink(`${data.video_url}&t=${Math.floor(segment.start)}s`, timestamp(segment.start)), text);
      fragment.append(row);
    }
    $('segments').append(fragment);
    $('transcript-meta').textContent = `${data.language} · ${data.is_generated ? 'Auto-generated captions' : 'Manual captions'} · ${data.segments.length} segments`;
    for (const panel of ['transcript-panel', 'summary-panel', 'chat-panel']) $(panel).hidden = false;
    $('chat-button').disabled = false;
    $('question').disabled = false;
    $('video-player').src = `https://www.youtube-nocookie.com/embed/${data.video_id}`;
    $('video-player').hidden = false;
    $('video-placeholder').hidden = true;
    $('video-title').textContent = 'Loaded YouTube video';
    $('video-byline').textContent = `${data.language} captions · Ready to explore`;
    $('watch-video').href = data.video_url;
    $('watch-video').hidden = false;
    resetGreeting(true);
    saveRecent(data);
    loadMetadata(data, currentRevision);
    $('transcript-status').textContent = 'Video loaded. Your summary and chat are below.';
    requestSummary(currentRevision);
  } catch (error) {
    $('transcript-status').textContent = errorText(error);
    $('summary-status').textContent = 'Load a captioned video to generate a summary.';
    resetGreeting(false);
  } finally {
    $('load').disabled = false;
  }
});

function appendMessage(role, text, sources = []) {
  const message = document.createElement('article');
  message.className = `message ${role}`;
  const label = document.createElement('span');
  label.className = 'message-avatar';
  label.textContent = role === 'user' ? 'Y' : 'AI';
  label.setAttribute('aria-label', role === 'user' ? 'You' : 'Video assistant');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const content = document.createElement('p');
  content.textContent = text;
  bubble.append(content);
  message.append(label, bubble);
  if (sources.length) {
    const details = document.createElement('details');
    const heading = document.createElement('summary');
    heading.textContent = 'Transcript sources';
    details.append(heading);
    for (const source of sources) {
      const passage = document.createElement('p');
      passage.textContent = source.text;
      details.append(sourceLink(source.url, `[${source.id}] ${timestamp(source.start)}–${timestamp(source.end)}`), passage);
    }
    bubble.append(details);
  }
  $('chat-messages').append(message);
  $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
}

$('chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!transcriptId || !question) return;
  const currentRevision = revision;
  const currentChat = chatRevision;
  $('chat-button').disabled = true;
  appendMessage('user', question);
  $('question').value = '';
  $('chat-status').textContent = 'Finding source passages and generating an answer…';
  try {
    const data = await api('/api/chat', { transcript_id: transcriptId, question, history });
    if (currentRevision !== revision || currentChat !== chatRevision) return;
    appendMessage('assistant', data.answer, data.sources);
    history = [...history, { role: 'user', content: question }, { role: 'assistant', content: data.answer.slice(0, 8000) }].slice(-8);
    $('chat-status').textContent = '';
  } catch (error) {
    if (currentRevision !== revision || currentChat !== chatRevision) return;
    $('chat-status').textContent = errorText(error);
    $('question').value = question;
  } finally {
    if (currentRevision === revision && currentChat === chatRevision) $('chat-button').disabled = false;
  }
});


function resetGreeting(loaded) {
  $('chat-messages').replaceChildren();
  appendMessage('assistant', loaded
    ? 'Hi! Ask me anything about this video. I’ll answer using the video content and link you to the sources.'
    : 'Hi! Load a YouTube video to get started. Then ask me about its ideas, examples, or key takeaways.');
}
resetGreeting(false);

$('clear-chat').addEventListener('click', () => {
  chatRevision += 1;
  history = [];
  $('question').value = '';
  $('chat-status').textContent = '';
  $('chat-button').disabled = !transcriptId;
  resetGreeting(Boolean(transcriptId));
});

function readSaved(name, fallback) {
  try { return JSON.parse(localStorage.getItem(name)) ?? fallback; } catch { return fallback; }
}
function writeSaved(name, value) {
  try { localStorage.setItem(name, JSON.stringify(value)); } catch { /* App still works without storage. */ }
}
function recentVideos() {
  const saved = readSaved('yt-recent', []);
  return Array.isArray(saved) ? saved.filter(item => /^[A-Za-z0-9_-]{11}$/.test(item.id) && typeof item.title === 'string').slice(0, 12) : [];
}
function saveRecent(data, title = 'YouTube video') {
  const recent = recentVideos().filter(item => item.id !== data.video_id);
  recent.unshift({ id: data.video_id, title, date: new Date().toISOString() });
  writeSaved('yt-recent', recent.slice(0, 12));
}
async function loadMetadata(data, currentRevision) {
  try {
    const meta = await api(`/api/video-info/${data.video_id}`, undefined, 10000);
    if (currentRevision !== revision) return;
    $('video-title').textContent = meta.title;
    $('video-byline').textContent = `${meta.author} · ${data.language} captions`;
    saveRecent(data, meta.title);
  } catch { /* Captions, summary, and chat work without metadata. */ }
}
function showView(view) {
  for (const name of ['home', 'history', 'about']) $(`${name}-view`).hidden = name !== view;
  for (const button of document.querySelectorAll('[data-view]')) {
    button.classList.toggle('active', button.dataset.view === view);
    if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  }
  if (view === 'history') renderHistory();
}
for (const button of document.querySelectorAll('[data-view]')) button.addEventListener('click', () => showView(button.dataset.view));
document.querySelector('.brand').addEventListener('click', () => showView('home'));
function renderHistory() {
  const container = $('history-list');
  container.replaceChildren();
  const recent = recentVideos();
  if (!recent.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No videos yet. Load your first video on the Home page.';
    container.append(empty);
  }
  for (const video of recent) {
    const row = document.createElement('article');
    row.className = 'history-item';
    const info = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = video.title;
    const date = document.createElement('small');
    date.textContent = new Date(video.date).toLocaleDateString();
    const button = document.createElement('button');
    button.className = 'secondary';
    button.textContent = 'Load video';
    button.addEventListener('click', () => {
      showView('home');
      $('video-url').value = `https://www.youtube.com/watch?v=${video.id}`;
      if (!$('load').disabled) $('video-form').requestSubmit();
    });
    info.append(title, date);
    row.append(info, button);
    container.append(row);
  }
}
function setTheme(dark) {
  document.body.classList.toggle('dark', dark);
  $('theme-toggle').textContent = dark ? '☀' : '☾';
  $('theme-toggle').setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
}
setTheme(readSaved('yt-dark', false) === true);
$('theme-toggle').addEventListener('click', () => {
  const dark = !document.body.classList.contains('dark');
  setTheme(dark);
  writeSaved('yt-dark', dark);
});
