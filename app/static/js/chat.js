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
    body.textContent = text;
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

    addMessage(message, 'user');
    input.value = '';
    input.focus();

    const loading = document.createElement('div');
    loading.className = 'message message-bot';
    loading.innerHTML = '<div class="message-badge">AI</div><div class="message-body">Pensando...</div>';
    log.appendChild(loading);
    scrollToBottom();

    try {
      const response = await fetch(window.APP_CONFIG.apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, chat_id: chatId })
      });

      const data = await response.json();
      loading.remove();

      if (!response.ok) {
        addMessage(data.error || 'Ocurrió un error al responder.', 'bot');
        return;
      }

      addMessage(data.answer || 'No hubo respuesta.', 'bot');
      await refreshChatsList();
    } catch (error) {
      loading.remove();
      addMessage('No se pudo conectar con el servidor.', 'bot');
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
