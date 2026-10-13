/**
 * app.js
 * Frontend application logic for Strict Calculation Agent.
 * Manages localStorage conversation history, differentiates AI vs Human messages,
 * and formats history to pass to the agent backend.
 */

const STORAGE_KEY = 'calc_agent_conversation_history';

const chatContainer = document.getElementById('chatContainer');
const messagesList = document.getElementById('messagesList');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const welcomeCard = document.getElementById('welcomeCard');

// -------------------------------------------------------------
// Local Storage History Management
// -------------------------------------------------------------

function getStoredHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Error loading history from localStorage:', e);
    return [];
  }
}

function saveStoredHistory(history) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  } catch (e) {
    console.error('Error saving history to localStorage:', e);
  }
}

function clearStoredHistory() {
  localStorage.removeItem(STORAGE_KEY);
  messagesList.innerHTML = '';
  if (welcomeCard) {
    welcomeCard.style.display = 'block';
  }
}

// -------------------------------------------------------------
// UI Rendering
// -------------------------------------------------------------

function renderMessage(msg) {
  if (welcomeCard) {
    welcomeCard.style.display = 'none';
  }

  const isHuman = msg.role === 'human';
  const wrapper = document.createElement('div');
  wrapper.className = `message-wrapper ${isHuman ? 'role-human' : 'role-ai'}`;

  // Header with role badge
  const header = document.createElement('div');
  header.className = 'message-header';
  const badge = document.createElement('span');
  badge.className = isHuman ? 'role-badge-human' : 'role-badge-ai';
  badge.textContent = isHuman ? '👤 You' : '🤖 Calculation Agent';
  header.appendChild(badge);
  wrapper.appendChild(header);

  // Content Bubble
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = msg.content;

  // If AI invoked the calculation tool, render tool execution card
  if (!isHuman && msg.tool_called) {
    const toolCard = document.createElement('div');
    toolCard.className = 'tool-card';
    
    const cardTitle = document.createElement('div');
    cardTitle.className = 'tool-card-title';
    cardTitle.innerHTML = '⚡ <span>Tool Executed: calculate</span>';
    toolCard.appendChild(cardTitle);

    if (msg.expression) {
      const expRow = document.createElement('div');
      expRow.className = 'tool-detail';
      expRow.innerHTML = `<span class="tool-key">Expression:</span> <span class="tool-val">${escapeHtml(msg.expression)}</span>`;
      toolCard.appendChild(expRow);
    }

    if (msg.tool_result) {
      const resRow = document.createElement('div');
      resRow.className = 'tool-detail';
      resRow.innerHTML = `<span class="tool-key">Tool Output:</span> <span class="tool-val">${escapeHtml(msg.tool_result)}</span>`;
      toolCard.appendChild(resRow);
    }

    bubble.appendChild(toolCard);
  }

  wrapper.appendChild(bubble);
  messagesList.appendChild(wrapper);
  scrollToBottom();
}

function renderTypingIndicator() {
  const wrapper = document.createElement('div');
  wrapper.id = 'typingIndicator';
  wrapper.className = 'message-wrapper role-ai';
  wrapper.innerHTML = `
    <div class="message-header">
      <span class="role-badge-ai">🤖 Calculation Agent</span>
    </div>
    <div class="bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messagesList.appendChild(wrapper);
  scrollToBottom();
}

function removeTypingIndicator() {
  const indicator = document.getElementById('typingIndicator');
  if (indicator) {
    indicator.remove();
  }
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

// -------------------------------------------------------------
// Form Submit & API Communication
// -------------------------------------------------------------

async function handleSubmit(query) {
  const text = (query || userInput.value).trim();
  if (!text) return;

  userInput.value = '';
  userInput.focus();

  // 1. Store and render human message
  const history = getStoredHistory();
  const humanMsg = { role: 'human', content: text };
  history.push(humanMsg);
  saveStoredHistory(history);
  renderMessage(humanMsg);

  // Formatted history for backend (exclude current message)
  const formattedHistory = history.slice(0, -1).map(({ role, content }) => ({ role, content }));

  sendBtn.disabled = true;
  renderTypingIndicator();

  try {
    // 2. Ask LLM to classify and extract the math expression (no tool call yet)
    const thinkResp = await fetch('/api/think', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: formattedHistory })
    });

    if (!thinkResp.ok) throw new Error(`Server error ${thinkResp.status}`);
    const think = await thinkResp.json();

    // 3a. Non-math query — show refusal, no tool call
    if (!think.is_math) {
      const refusalMsg = {
        role: 'ai',
        content: think.refusal || 'I can only assist with mathematical calculations.',
        tool_called: false
      };
      history.push(refusalMsg);
      saveStoredHistory(history);
      removeTypingIndicator();
      renderMessage(refusalMsg);
      return;
    }

    // 3b. Math query — invoke the tool via WebMCP so Tool Activity captures it
    const expression = think.expression;
    let toolResult;

    const modelContext = window.document?.modelContext || window.navigator?.modelContext;
    if (modelContext && typeof modelContext.invokeTool === 'function') {
      // ✅ WebMCP path: Chrome DevTools Tool Activity will show this call
      const wcResult = await modelContext.invokeTool('calculate', { expression });
      // result.content is [{type:'text', text:'...'}]
      toolResult = wcResult?.content?.[0]?.text ?? String(wcResult);
    } else {
      // Fallback: call /api/evaluate directly (Tool Activity won't show, but still works)
      const evalResp = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression })
      });
      const evalData = await evalResp.json();
      toolResult = String(evalData.result);
    }

    // 4. Compose and render AI response
    const aiMsg = {
      role: 'ai',
      content: `The result of ${expression} is **${toolResult}**.`,
      tool_called: true,
      expression,
      tool_result: toolResult
    };
    history.push(aiMsg);
    saveStoredHistory(history);
    removeTypingIndicator();
    renderMessage(aiMsg);

  } catch (error) {
    removeTypingIndicator();
    const errorMsg = {
      role: 'ai',
      content: `Error: Unable to connect to calculation agent (${error.message}). Please ensure backend is running.`,
      tool_called: false
    };
    renderMessage(errorMsg);
  } finally {
    sendBtn.disabled = false;
  }
}

// -------------------------------------------------------------
// WebMCP Integration (Chrome DevTools Application > WebMCP)
// -------------------------------------------------------------

async function registerWebMCP() {
  const modelContext = window.document?.modelContext || window.navigator?.modelContext;
  if (!modelContext || typeof modelContext.registerTool !== 'function') {
    return;
  }

  try {
    await modelContext.registerTool({
      name: 'calculate',
      description: 'Evaluates any mathematical expression accurately (arithmetic, algebra, trigonometry, powers, roots, calculus).',
      inputSchema: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'The raw mathematical expression string to evaluate (e.g. 25 * 48, sqrt(144), sin(pi/2))'
          }
        },
        required: ['expression']
      },
      async execute({ expression }) {
        const resp = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ expression })
        });
        const data = await resp.json();
        return {
          content: [{ type: 'text', text: String(data.result) }]
        };
      }
    });
    console.log('✓ WebMCP calculate tool registered. Inspect in DevTools -> Application -> WebMCP');
  } catch (err) {
    console.warn('WebMCP registration note:', err);
  }
}

// -------------------------------------------------------------
// Initialization & Event Listeners
// -------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  // Register WebMCP tool for Chrome DevTools Application > WebMCP panel
  registerWebMCP();

  // Load messages from localStorage
  const saved = getStoredHistory();
  if (saved.length > 0) {
    saved.forEach(renderMessage);
  }

  // Form submit
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSubmit();
  });

  // Clear history button
  clearBtn.addEventListener('click', () => {
    if (confirm('Clear conversation history stored in localStorage?')) {
      clearStoredHistory();
    }
  });

  // Quick chip buttons
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query) {
        handleSubmit(query);
      }
    });
  });
});
