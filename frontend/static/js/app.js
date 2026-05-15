const els = {};

function initElements() {
  els.modeRadios = document.querySelectorAll('input[name="mode"]');
  els.modeIndicator = document.getElementById('mode-indicator');
  els.chatForm = document.getElementById('chat-form');
  els.userInput = document.getElementById('user-input');
  els.output = document.getElementById('output');
  els.sourcesList = document.getElementById('sources-list');
  els.alertBox = document.getElementById('alert');
  els.sendBtn = document.getElementById('send-btn');
}

function updateUI(mode) {
  const isWrite = mode === 'write';
  document.body.dataset.mode = mode;
  if (els.modeIndicator) els.modeIndicator.textContent = isWrite ? 'Режим: Запись' : 'Режим: Чтение';
  if (els.userInput) els.userInput.placeholder = isWrite ? 'Введи текст или путь к файлу...' : 'Спроси что-нибудь...';
  if (els.sendBtn) els.sendBtn.textContent = isWrite ? 'Сохранить' : 'Отправить';
  if (els.output) els.output.innerHTML = '<p class="placeholder">Ожидание ввода...</p>';
  if (els.sourcesList) els.sourcesList.innerHTML = '<p class="placeholder">Источники появятся здесь...</p>';
}

async function handleSubmit(e) {
  if (e) e.preventDefault();
  const text = els.userInput?.value.trim();
  if (!text) return;

  const mode = document.body.dataset.mode || 'read';
  els.userInput.disabled = true;
  els.sendBtn.disabled = true;
  els.sendBtn.textContent = '⏳';

  if (els.output) els.output.innerHTML = mode === 'read' ? '<p class="placeholder">🔍 Ищу...</p>' : '<p class="placeholder">⏳ Сохраняю...</p>';
  if (els.sourcesList) els.sourcesList.innerHTML = '';

  try {
    if (mode === 'read') await handleQueryStream(text);
    else await handleIngestRequest(text);
  } catch (err) {
    console.error('❌ Error:', err);
    showAlert(`Ошибка: ${err.message}`, 'error');
    if (els.output) els.output.innerHTML = '<p style="color: var(--text-muted);">❌ Сбой</p>';
  } finally {
    if (els.userInput) { els.userInput.disabled = false; els.userInput.value = ''; els.userInput.focus(); }
    // Разблокируем кнопку всегда, но текст меняем только если это не статус успеха
    if (els.sendBtn) els.sendBtn.disabled = false;
    if (els.sendBtn && !els.sendBtn.textContent.includes('✓')) {
      els.sendBtn.textContent = mode === 'read' ? 'Отправить' : 'Сохранить';
    }
  }
}

async function handleQueryStream(question) {
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answerBuffer = '';

  if (els.output) els.output.innerHTML = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop();

    for (const raw of events) {
      if (!raw.trim()) continue;
      const lines = raw.split('\n');
      let type = '', dataParts = [];

      for (const line of lines) {
        if (line.startsWith('event:')) type = line.slice(6).trim();
        else if (line.startsWith('data:')) dataParts.push(line.slice(5));
      }

      const data = dataParts.join('\n');
      if (!data) continue;

      if (type === 'sources') renderSources(JSON.parse(data));
      else if (type === 'answer') {
        answerBuffer += data;
        if (els.output) {
          // Передаём опции прямо в parse (современный подход для marked v5+)
          els.output.innerHTML = marked.parse(answerBuffer, { breaks: true, gfm: true });
          els.output.scrollTop = els.output.scrollHeight;
        }
      } else if (type === 'done') {
        if (els.sendBtn) els.sendBtn.textContent = 'Готово ✓';
      } else if (type === 'error') {
        showAlert(JSON.parse(data).message, 'error');
      }
    }
  }
}

function renderSources(sources) {
  if (!els.sourcesList) return;
  if (!sources?.length) {
    els.sourcesList.innerHTML = '<p class="placeholder">Ничего не найдено</p>';
    return;
  }
  // ✅ Исправлен синтаксис шаблонных строк
  els.sourcesList.innerHTML = sources.map(src => `
    <div class="source-card">
      <div class="source-topic">${src.topic || 'info'}</div>
      <div class="source-preview">${src.preview}</div>
    </div>
  `).join('');
}

async function handleIngestRequest(text) {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  const status = data.status === 'success' ? 'success' :
                 data.status === 'duplicate' ? 'duplicate' : 'error';
  showAlert(data.message, status);
  if (els.output) {
    els.output.innerHTML = `<p>${data.message}</p>${data.data?.topic ? `<p class="placeholder" style="margin-top:8px">Тема: ${data.data.topic}</p>` : ''}`;
  }
  // Добавил статус успеха и для режима записи, чтобы finally не сбрасывал его
  if (els.sendBtn) els.sendBtn.textContent = 'Готово ✓';
}

function showAlert(msg, type) {
  if (!els.alertBox) return;
  els.alertBox.className = `alert ${type}`;
  els.alertBox.textContent = msg;
  els.alertBox.classList.add('show');
  setTimeout(() => els.alertBox.classList.remove('show'), 4000);
}

document.addEventListener('DOMContentLoaded', () => {
  initElements();
  updateUI('read');
  els.modeRadios?.forEach(radio => {
    radio.addEventListener('change', e => updateUI(e.target.value));
  });
  if (els.chatForm) els.chatForm.addEventListener('submit', handleSubmit);
  els.userInput?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      els.chatForm?.requestSubmit();
    }
  });
  els.userInput?.focus();
});