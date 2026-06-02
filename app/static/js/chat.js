document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const log = document.getElementById('chat-log');
  const newChatButton = document.querySelector('[data-action="new-chat"]');
  const chatsList = document.getElementById('chats-list');
  const currentChatLabel = document.getElementById('current-chat-label');

  const chatStorageKey = 'llmrag_chat_id';

  const createChatId = () => {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
      return window.crypto.randomUUID();
    }

    return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  };

  const getChatId = () => {
    const storedChatId = window.sessionStorage.getItem(chatStorageKey);
    if (storedChatId) {
      return storedChatId;
    }

    const chatId = createChatId();
    window.sessionStorage.setItem(chatStorageKey, chatId);
    return chatId;
  };

  let chatId = getChatId();

  if (!form || !input || !log) {
    return;
  }

  const scrollToBottom = () => {
    log.scrollTop = log.scrollHeight;
  };

  const setCurrentChatLabel = (label) => {
    if (currentChatLabel) {
      currentChatLabel.textContent = label;
    }
  };

  const setComposerDisabled = (disabled) => {
    input.disabled = disabled;
    const submitBtn = form.querySelector('.send-btn');
    if (submitBtn) {
      submitBtn.disabled = disabled;
      if (disabled) {
        submitBtn.style.opacity = '0.5';
        submitBtn.style.cursor = 'not-allowed';
      } else {
        submitBtn.style.opacity = '1';
        submitBtn.style.cursor = 'pointer';
      }
    }
  };

  const renderMessage = (text, role) => {
    const message = document.createElement('div');
    message.className = `message message-${role}`;

    if (role === 'bot') {
      const badge = document.createElement('div');
      badge.className = 'message-badge';
      badge.textContent = 'AI';
      message.appendChild(badge);
    }

    const body = document.createElement('div');
    body.className = 'message-body';
    if (role === 'bot' && typeof marked !== 'undefined') {
      body.innerHTML = marked.parse(text);
    } else {
      body.textContent = text;
    }
    message.appendChild(body);

    log.appendChild(message);
  };

  const addMessage = (text, role) => {
    renderMessage(text, role);
    scrollToBottom();
  };

  const clearChatView = () => {
    log.innerHTML = '';
  };

  const loadChatHistory = async (selectedChatId, fallbackTitle = 'Chat') => {
    chatId = selectedChatId;
    window.sessionStorage.setItem(chatStorageKey, chatId);
    clearChatView();
    setCurrentChatLabel(fallbackTitle);

    try {
      const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}`);
      const data = await response.json();
      const messages = data.messages || [];

      if (!messages.length) {
        renderMessage('Nuevo chat listo. Escribe tu pregunta.', 'bot');
        scrollToBottom();
        return;
      }

      messages.forEach((message) => {
        renderMessage(message.content, message.role === 'assistant' ? 'bot' : 'user');
      });
      scrollToBottom();
    } catch (error) {
      renderMessage('No se pudo cargar el historial de este chat.', 'bot');
      scrollToBottom();
    }
  };

  const refreshChatsList = async () => {
    if (!chatsList) {
      return;
    }

    try {
      const response = await fetch('/api/chats');
      const data = await response.json();
      const chats = data.chats || [];

      chatsList.innerHTML = '';

      if (!chats.length) {
        const emptyItem = document.createElement('div');
        emptyItem.className = 'history-empty';
        emptyItem.textContent = 'Sin chats guardados todavía';
        chatsList.appendChild(emptyItem);
        return;
      }

      chats.forEach((chat) => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = `history-item${chat.chat_id === chatId ? ' history-item-active' : ''}`;
        item.innerHTML = `
          <div class="history-item-main">
            <span class="material-symbols-outlined history-icon">chat_bubble</span>
            <span class="history-text">${chat.title}</span>
          </div>
          <span class="history-meta">${new Date(chat.updated_at).toLocaleDateString()}</span>
        `;
        item.addEventListener('click', () => {
          loadChatHistory(chat.chat_id, chat.title);
        });
        chatsList.appendChild(item);
      });
    } catch (error) {
      chatsList.innerHTML = '<div class="history-empty">No se pudo cargar el historial</div>';
    }
  };

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message) {
      return;
    }

    setComposerDisabled(true); // Deshabilitamos entrada para evitar doble envío
    addMessage(message, 'user');
    input.value = '';

    const loading = document.createElement('div');
    loading.className = 'message message-bot';
    loading.innerHTML = `
      <div class="message-badge">AI</div>
      <div class="message-body" style="display: flex; align-items: center; gap: 0.5rem;">
        <span class="status-spinner"></span>
        <span class="status-text">Clasificando pregunta...</span>
      </div>
    `;
    log.appendChild(loading);
    scrollToBottom();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, chat_id: chatId })
      });

      if (!response.ok) {
        loading.remove();
        let errorMsg = 'Ocurrió un error al responder.';
        try {
          const data = await response.json();
          errorMsg = data.error || errorMsg;
        } catch (_) {}
        addMessage(errorMsg, 'bot');
        setComposerDisabled(false);
        input.focus();
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let botMessageElement = null;
      let botBodyElement = null;
      let accumulatedAnswer = '';
      let isFirstToken = true;
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const event = JSON.parse(dataStr);

              if (event.type === 'status') {
                const statusText = loading.querySelector('.status-text');
                if (statusText) {
                  statusText.textContent = event.content;
                }
              }
              else if (event.type === 'token') {
                if (isFirstToken) {
                  isFirstToken = false;
                  loading.remove();

                  botMessageElement = document.createElement('div');
                  botMessageElement.className = 'message message-bot';

                  const badge = document.createElement('div');
                  badge.className = 'message-badge';
                  badge.textContent = 'AI';
                  botMessageElement.appendChild(badge);

                  botBodyElement = document.createElement('div');
                  botBodyElement.className = 'message-body streaming-text';
                  botMessageElement.appendChild(botBodyElement);

                  log.appendChild(botMessageElement);
                }

                accumulatedAnswer += event.content;
                if (typeof marked !== 'undefined') {
                  botBodyElement.innerHTML = marked.parse(accumulatedAnswer);
                } else {
                  botBodyElement.textContent = accumulatedAnswer;
                }
                scrollToBottom();
              }
              else if (event.type === 'done') {
                if (botBodyElement) {
                  botBodyElement.classList.remove('streaming-text');
                }
                await refreshChatsList();
                setComposerDisabled(false);
                input.focus();
              }
              else if (event.type === 'error') {
                loading.remove();
                addMessage(event.content || 'Error en la transmisión.', 'bot');
                setComposerDisabled(false);
                input.focus();
              }
            } catch (err) {
              console.error('Error al parsear evento SSE:', err, line);
            }
          }
        }
      }
    } catch (error) {
      loading.remove();
      addMessage('No se pudo conectar con el servidor.', 'bot');
      setComposerDisabled(false);
      input.focus();
    }
  });

  if (newChatButton) {
    newChatButton.addEventListener('click', () => {
      chatId = createChatId();
      window.sessionStorage.setItem(chatStorageKey, chatId);
      clearChatView();
      setCurrentChatLabel('Nuevo chat');
      renderMessage('Nuevo chat listo. Escribe tu pregunta.', 'bot');
      input.value = '';
      input.focus();
      refreshChatsList();
    });
  }

  loadChatHistory(chatId, 'Chat actual');
  refreshChatsList();
});
