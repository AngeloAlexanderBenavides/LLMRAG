document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const log = document.getElementById('chat-log');
  const newChatButton = document.querySelector('[data-action="new-chat"]');

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

  const addMessage = (text, role) => {
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
    scrollToBottom();
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
    } catch (error) {
      loading.remove();
      addMessage('No se pudo conectar con el servidor.', 'bot');
    }
  });

  if (newChatButton) {
    newChatButton.addEventListener('click', () => {
      chatId = createChatId();
      window.sessionStorage.setItem(chatStorageKey, chatId);
      log.innerHTML = '';
      addMessage('Nuevo chat listo. Escribe tu pregunta.', 'bot');
      input.value = '';
      input.focus();
    });
  }
});
