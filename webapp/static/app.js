/**
 * MRAG — клиентская логика.
 *
 * Разделы: API-клиент → авторизация → чаты → сообщения → PDF-вьювер → презентация.
 */

"use strict";

// ============================================================
// КОНФИГУРАЦИЯ И СОСТОЯНИЕ
// ============================================================

const API = ""; // тот же origin, что и FastAPI
const TOKEN_KEY = "mrag_token";
const USER_KEY = "mrag_user";

const state = {
  token: localStorage.getItem(TOKEN_KEY),
  user: localStorage.getItem(USER_KEY),
  chats: [],
  currentChatId: null,
  lessons: [],       // [{lesson, pdf_name, pages}]
  pdfDoc: null,      // загруженный документ PDF.js
  pdfLesson: null,   // текущий урок в панели
  pdfPage: 1,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ============================================================
// API-КЛИЕНТ
// ============================================================

/**
 * Запрос к API с автоматической подстановкой JWT.
 * @param {string} path — путь, напр. "/api/chat"
 * @param {object} options — опции fetch
 * @returns {Promise<any>} разобранный JSON или null для 204
 */
async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) {
    headers.Authorization = `Bearer ${state.token}`;
  }

  const response = await fetch(API + path, { ...options, headers });

  // Токен протух — сбрасываем сессию и просим войти заново
  if (response.status === 401) {
    logout(false);
    throw new Error("Сессия истекла, войдите заново");
  }

  if (response.status === 204) return null;

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Ошибка ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

// ============================================================
// ВСПОМОГАТЕЛЬНОЕ
// ============================================================

let toastTimer = null;

/**
 * Показывает всплывающее уведомление.
 * @param {string} message — текст
 */
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("toast--visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("toast--visible"), 3600);
}

/**
 * Экранирует HTML, чтобы ответ модели не ломал разметку.
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

/** Автоувеличение высоты textarea под текст. */
function autoGrow(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
}
// ============================================================
// АВТОРИЗАЦИЯ
// ============================================================

/** Переключает табы "Вход" / "Регистрация". */
function initAuthTabs() {
  $$("[data-auth-tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.authTab;
      $$("[data-auth-tab]").forEach((t) => t.classList.toggle("tab--active", t === tab));
      $("#login-form").classList.toggle("auth-form--hidden", target !== "login");
      $("#register-form").classList.toggle("auth-form--hidden", target !== "register");
      $("#auth-error").textContent = "";
    });
  });
}

/**
 * Сохраняет сессию и показывает основной интерфейс.
 * @param {string} token
 * @param {string} username
 */
async function loginSuccess(token, username) {
  state.token = token;
  state.user = username;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, username);

  $("#auth-modal").classList.remove("modal--visible");
  $("#app").classList.remove("app--hidden");
  $("#user-chip").textContent = username;

  await Promise.all([loadLessons(), loadChats()]);
}

/**
 * Выходит из аккаунта.
 * @param {boolean} showMessage — показывать ли уведомление
 */
function logout(showMessage = true) {
  state.token = null;
  state.user = null;
  state.chats = [];
  state.currentChatId = null;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);

  $("#app").classList.add("app--hidden");
  $("#auth-modal").classList.add("modal--visible");
  $("#chat-list").innerHTML = "";
  renderMessages([]);

  if (showMessage) toast("Вы вышли из аккаунта");
}

/**
 * Отправляет форму авторизации.
 * @param {HTMLFormElement} form
 * @param {string} endpoint — "/api/auth/login" или "/register"
 */
async function submitAuth(form, endpoint) {
  const errorEl = $("#auth-error");
  errorEl.textContent = "";

  const payload = {
    username: form.username.value.trim(),
    password: form.password.value,
  };

  try {
    const data = await api(endpoint, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    form.reset();
    await loginSuccess(data.access_token, data.username);
  } catch (error) {
    errorEl.textContent = error.message;
  }
}

/** Навешивает обработчики на формы и кнопку выхода. */
function initAuth() {
  initAuthTabs();
  $("#login-form").addEventListener("submit", (e) => {
    e.preventDefault();
    submitAuth(e.target, "/api/auth/login");
  });
  $("#register-form").addEventListener("submit", (e) => {
    e.preventDefault();
    submitAuth(e.target, "/api/auth/register");
  });
  $("#logout-btn").addEventListener("click", () => logout(true));
}

/**
 * Пробует восстановить сессию из localStorage.
 * @returns {Promise<boolean>} удалось ли
 */
// ============================================================
// УРОКИ
// ============================================================

/** Загружает список уроков для навигации по PDF. */
async function loadLessons() {
  try {
    state.lessons = await api("/api/lessons");
  } catch (error) {
    state.lessons = [];
    toast("Не удалось загрузить список уроков");
  }
}

/** @returns {object|null} метаданные урока по номеру */
function lessonMeta(lesson) {
  return state.lessons.find((l) => l.lesson === lesson) || null;
}

// ============================================================
// ЧАТЫ
// ============================================================

/** Загружает историю чатов и перерисовывает сайдбар. */
async function loadChats() {
  try {
    state.chats = await api("/api/chat");
  } catch (error) {
    state.chats = [];
  }
  renderChatList();
}

/** Рисует список чатов в сайдбаре. */
function renderChatList() {
  const list = $("#chat-list");
  list.innerHTML = "";

  if (state.chats.length === 0) {
    const empty = document.createElement("li");
    empty.className = "chat-item";
    empty.style.color = "var(--fg-muted)";
    empty.style.cursor = "default";
    empty.textContent = "Пока пусто";
    list.appendChild(empty);
    return;
  }

  state.chats.forEach((chat) => {
    const li = document.createElement("li");
    li.className = "chat-item" + (chat.id === state.currentChatId ? " chat-item--active" : "");

    const title = document.createElement("span");
    title.className = "chat-item__title";
    title.textContent = chat.title || "Без названия";
    title.title = chat.title || "";

    const del = document.createElement("button");
    del.className = "chat-item__del";
    del.textContent = "\u00d7";
    del.title = "Удалить чат";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteChat(chat.id);
    });

    li.append(title, del);
    li.addEventListener("click", () => openChat(chat.id));
    list.appendChild(li);
  });
}

/** Создаёт новый пустой чат. */
async function createChat() {
  try {
    const chat = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ title: "Новый чат" }),
    });
    state.chats.unshift(chat);
    state.currentChatId = chat.id;
    renderChatList();
    renderMessages([]);
    $("#question-input").focus();
  } catch (error) {
    toast(error.message);
  }
}

/**
 * Открывает чат и показывает его историю.
 * @param {number} chatId
 */
async function openChat(chatId) {
  try {
    const chat = await api(`/api/chat/${chatId}`);
    state.currentChatId = chat.id;
    renderChatList();
    renderMessages(chat.messages || []);
  } catch (error) {
    toast(error.message);
  }
}

/**
 * Удаляет чат.
 * @param {number} chatId
 */
async function deleteChat(chatId) {
  try {
// ============================================================
// СООБЩЕНИЯ
// ============================================================

/**
 * Полностью перерисовывает ленту сообщений.
 * @param {Array} messages — список сообщений с сервера
 */
function renderMessages(messages) {
  const container = $("#messages");
  container.innerHTML = "";

  if (!messages || messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.id = "empty-state";
    empty.innerHTML = `
      <h2 class="empty-state__title">Задайте вопрос по курсу</h2>
      <p class="empty-state__text">
        Система найдёт ответ в лекциях и покажет страницы-источники справа.
      </p>
      <div class="sample-questions">
        <button class="sample-q">Что такое функция потерь?</button>
        <button class="sample-q">Как работает backpropagation?</button>
        <button class="sample-q">Что такое переобучение?</button>
      </div>`;
    container.appendChild(empty);
    bindSampleQuestions();
    return;
  }

  messages.forEach((message) => appendMessage(message.role, message.content, message.sources));
  scrollMessages();
}

/** Навешивает обработчики на кнопки примеров вопросов. */
function bindSampleQuestions() {
  $$(".sample-q").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("#question-input").value = btn.textContent;
      $("#ask-form").requestSubmit();
    });
  });
}

/**
 * Добавляет сообщение в ленту.
 * @param {"user"|"assistant"} role
 * @param {string} content
 * @param {Array} sources — [{lesson, page, score}]
 * @returns {HTMLElement} созданный узел
 */
function appendMessage(role, content, sources) {
  const emptyState = $("#empty-state");
  if (emptyState) emptyState.remove();

  const wrapper = document.createElement("div");
  wrapper.className = `msg msg--${role}`;

  const roleLabel = document.createElement("span");
  roleLabel.className = "msg__role";
  roleLabel.textContent = role === "user" ? "Вы" : "MRAG";

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";
  bubble.textContent = content || "";

  wrapper.append(roleLabel, bubble);

  if (sources && sources.length > 0) {
    const sourcesBox = document.createElement("div");
    sourcesBox.className = "sources";
    sources.forEach((src) => {
      const chip = document.createElement("button");
      chip.className = "source-chip";
      const score = typeof src.score === "number" ? src.score.toFixed(2) : "";
      chip.innerHTML =
        `Урок ${escapeHtml(src.lesson)} · стр. ${escapeHtml(src.page)}` +
        `<span class="source-chip__score">${score}</span>`;
      chip.title = "Открыть страницу презентации";
      chip.addEventListener("click", () => openPdfPage(src.lesson, src.page));
      sourcesBox.appendChild(chip);
    });
    wrapper.appendChild(sourcesBox);
  }

  $("#messages").appendChild(wrapper);
  return wrapper;
}

/** Показывает индикатор ожидания ответа. */
function showTyping() {
  const wrapper = document.createElement("div");
  wrapper.className = "msg msg--assistant";
  wrapper.id = "typing-indicator";
  wrapper.innerHTML = `
    <span class="msg__role">MRAG</span>
    <div class="msg__bubble typing"><span></span><span></span><span></span></div>`;
  $("#messages").appendChild(wrapper);
  scrollMessages();
}

/** Убирает индикатор ожидания. */
function hideTyping() {
  const el = $("#typing-indicator");
  if (el) el.remove();
}

/** Прокручивает ленту к последнему сообщению. */
function scrollMessages() {
  const container = $("#messages");
  container.scrollTop = container.scrollHeight;
}

/**
 * Отправляет вопрос на сервер и показывает ответ.
 * @param {string} question
 */
async function sendQuestion(question) {
  const sendBtn = $("#send-btn");
  sendBtn.disabled = true;

  appendMessage("user", question, null);
  scrollMessages();
  showTyping();

  try {
    const data = await api("/api/chat/ask", {
      method: "POST",
      body: JSON.stringify({
        question,
        chat_id: state.currentChatId,
      }),
    });

    hideTyping();
    state.currentChatId = data.chat_id;
    appendMessage("assistant", data.answer, data.sources);
    scrollMessages();

    // Новый чат мог появиться — обновляем историю
    await loadChats();

    // Автоматически открываем первую страницу-источник
    if (data.sources && data.sources.length > 0) {
      openPdfPage(data.sources[0].lesson, data.sources[0].page);
    }
  } catch (error) {
    hideTyping();
    appendMessage("assistant", `Ошибка: ${error.message}`, null);
    scrollMessages();
  } finally {
    sendBtn.disabled = false;
  }
}

/** Навешивает обработчики на форму вопроса. */
function initComposer() {
  const form = $("#ask-form");
  const input = $("#question-input");

  input.addEventListener("input", () => autoGrow(input));

  // Enter отправляет, Shift+Enter переносит строку
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    autoGrow(input);
// ============================================================
// PDF-ВЬЮВЕР
// ============================================================

/**
 * Открывает панель PDF и показывает нужную страницу урока.
 * @param {number} lesson
 * @param {number} page
 */
async function openPdfPage(lesson, page) {
  $("#pdf-panel").classList.remove("pdf-panel--hidden");

  const switchedLesson = state.pdfLesson !== lesson;
  state.pdfLesson = lesson;
  state.pdfPage = page || 1;

  if (switchedLesson || !state.pdfDoc) {
    await loadPdfDocument(lesson);
  }
  renderPdfPage(state.pdfPage);
}

/**
 * Загружает PDF урока через PDF.js.
 * @param {number} lesson
 */
async function loadPdfDocument(lesson) {
  const placeholder = $("#pdf-placeholder");
  placeholder.textContent = "Загрузка презентации…";

  try {
    state.pdfDoc = await pdfjsLib.getDocument(`/api/lessons/${lesson}/pdf`).promise;
  } catch (error) {
    state.pdfDoc = null;
    placeholder.textContent = "Не удалось загрузить презентацию";
    return;
  }

  const meta = lessonMeta(lesson);
  if (meta) {
    state.pdfPage = Math.min(state.pdfPage, meta.pages || 1);
  }
}

/**
 * Рисует страницу PDF на canvas.
 * @param {number} pageNumber — номер страницы (с 1)
 */
async function renderPdfPage(pageNumber) {
  if (!state.pdfDoc) return;

  const total = state.pdfDoc.numPages;
  const page = Math.max(1, Math.min(pageNumber, total));
  state.pdfPage = page;

  const pdfPage = await state.pdfDoc.getPage(page);

  // Масштабируем под ширину панели, но не мельче 1.0
  const viewportWidth = $("#pdf-viewport").clientWidth - 40;
  const baseViewport = pdfPage.getViewport({ scale: 1 });
  const scale = Math.max(1.0, viewportWidth / baseViewport.width);
  const viewport = pdfPage.getViewport({ scale });

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  canvas.width = viewport.width;
  canvas.height = viewport.height;

  await pdfPage.render({ canvasContext: context, viewport }).promise;

  const viewportEl = $("#pdf-viewport");
  viewportEl.innerHTML = "";
  viewportEl.appendChild(canvas);

  const meta = lessonMeta(state.pdfLesson);
  $("#pdf-lesson-label").textContent = `Урок ${state.pdfLesson}`;
  $("#pdf-page-label").textContent = `стр. ${page} из ${total}`;
  $("#pdf-page-input").value = page;
  $("#pdf-page-input").max = total;

  // Красим заголовок цветом из темы
  $("#pdf-lesson-label").style.color = "var(--blue)";
  $("#pdf-page-label").style.color = "var(--orange)";

  const hasMeta = Boolean(meta);
  $("#pdf-prev-lesson").disabled = !hasMeta;
  $("#pdf-next-lesson").disabled = !hasMeta;
}

/** Переходит к соседнему уроку. @param {number} delta — -1 или +1 */
async function shiftLesson(delta) {
  if (state.lessons.length === 0) return;
  const numbers = state.lessons.map((l) => l.lesson).sort((a, b) => a - b);
  const index = numbers.indexOf(state.pdfLesson);
  const nextIndex = index === -1 ? 0 : index + delta;

  if (nextIndex < 0 || nextIndex >= numbers.length) {
    toast(delta < 0 ? "Это первый урок" : "Это последний урок");
    return;
  }

  const targetLesson = numbers[nextIndex];
  state.pdfDoc = null;
  await openPdfPage(targetLesson, 1);
}

/** Навешивает обработчики на панель PDF. */
function initPdfPanel() {
  $("#pdf-prev").addEventListener("click", () => renderPdfPage(state.pdfPage - 1));
  $("#pdf-next").addEventListener("click", () => renderPdfPage(state.pdfPage + 1));
  $("#pdf-prev-lesson").addEventListener("click", () => shiftLesson(-1));
  $("#pdf-next-lesson").addEventListener("click", () => shiftLesson(1));

  $("#pdf-page-input").addEventListener("change", (event) => {
    const value = parseInt(event.target.value, 10);
    if (!Number.isNaN(value)) renderPdfPage(value);
  });

  // Стрелки листают страницы, если фокус не в поле ввода
  document.addEventListener("keydown", (event) => {
    if ($("#pdf-panel").classList.contains("pdf-panel--hidden")) return;
    const tag = document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;

    if (event.key === "ArrowRight") renderPdfPage(state.pdfPage + 1);
    if (event.key === "ArrowLeft") renderPdfPage(state.pdfPage - 1);
  });

  // Пересчитываем масштаб при изменении размера окна
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if ($("#pdf-panel").classList.contains("pdf-panel--hidden")) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => renderPdfPage(state.pdfPage), 250);
  });
}
    sendQuestion(question);
  });

  $("#new-chat-btn").addEventListener("click", createChat);
}
    await api(`/api/chat/${chatId}`, { method: "DELETE" });
    state.chats = state.chats.filter((c) => c.id !== chatId);
    if (state.currentChatId === chatId) {
      state.currentChatId = null;
      renderMessages([]);
    }
    renderChatList();
    toast("Чат удалён");
  } catch (error) {
    toast(error.message);
  }
}
async function restoreSession() {
// ============================================================
// ПРЕЗЕНТАЦИЯ ПРОЕКТА
// ============================================================

/** Этапы проекта: кратко в шапке, подробности — в раскрывающемся блоке. */
const STAGES = [
  {
    badge: "blue",
    title: "Сбор данных",
    summary: "59 PDF с Яндекс Диска",
    details: [
      "Публичный API Яндекс Диска: обход папок Lessons 1-59.",
      "Фильтрация служебных файлов (домашки, учебные кейсы).",
      "Переименование в lesson_N.pdf для сквозной нумерации.",
      "Докачка через .part — обрыв связи не оставляет битых файлов.",
    ],
  },
  {
    badge: "green",
    title: "Парсинг и чанкинг",
    summary: "973 страницы → 1034 чанка",
    details: [
      "pdfplumber извлекает текст постранично с citation-метаданными.",
      "EDA на реальных данных подобрал размер чанка.",
      "chunk_size=1000: 93% страниц укладываются в один чанк.",
      "Против 500 симв. — там рвалось 72% страниц.",
    ],
  },
  {
    badge: "orange",
    title: "Эмбеддинги",
    summary: "sentence-transformers, 384d",
    details: [
      "Модель paraphrase-multilingual-MiniLM-L12-v2 (RU/EN).",
      "L2-нормализация: cosine = скалярное произведение.",
      "Батчи по 64 текста, dtype float32.",
      "Апгрейд до BAAI/bge-m3 (1024d) — в планах.",
    ],
  },
  {
    badge: "pink",
    title: "Векторная БД",
    summary: "Chroma (по умолчанию) / Qdrant",
    details: [
      "Абстракция VectorStore с двумя реализациями.",
      "Переключение бэкенда одной строкой в vector_store.yaml.",
      "Метаданные: lesson, page, chunk_id, pdf_name.",
      "Тексты хранятся в documents, метаданные — отдельно.",
    ],
  },
  {
    badge: "blue",
    title: "RAG и генерация",
    summary: "retrieve → prompt → LLM",
    details: [
      "top_k=8 чанков по косинусной близости.",
      "Prompt требует отвечать только по чанкам и cited источник.",
      "LLM через OpenRouter: цепочка бесплатных моделей.",
      "FallbackLLMClient переключает модель при rate-limit.",
    ],
  },
  {
    badge: "green",
    title: "API и веб-интерфейс",
    summary: "FastAPI + JWT + SQLite",
    details: [
      "JWT-авторизация, пароли хешируются bcrypt.",
      "SQLAlchemy: users, chats, messages, sources.",
      "REST: /api/auth, /api/chat, /api/lessons.",
      "Фронтенд: PDF.js, навигация по страницам и урокам.",
    ],
  },
  {
    badge: "orange",
    title: "Оценка качества",
    summary: "gold-набор, Hit@k, MRR",
    details: [
      "63 вопроса сгенерированы LLM по 21 уроку.",
      "Метрики: hit@1=0.365, hit@5=0.667, MRR=0.491.",
      "Диагностика: 100% провалов — страницы с мусорным текстом.",
      "Вывод: узкое место — парсинг таблиц и схем, не модель.",
    ],
  },
];

/** Рисует аккордеон этапов проекта. */
function renderStages() {
  const container = $("#stages");
  container.innerHTML = "";

  STAGES.forEach((stage, index) => {
    const li = document.createElement("li");
    li.className = "stage";

    const head = document.createElement("div");
    head.className = "stage__head";

    const num = document.createElement("span");
    num.className = "stage__num";
    num.textContent = String(index + 1);

    const title = document.createElement("span");
    title.className = "stage__title";
    title.textContent = stage.title;

    const badge = document.createElement("span");
    badge.className = `stage__badge stage__badge--${stage.badge}`;
    badge.textContent = stage.summary;

    const arrow = document.createElement("span");
    arrow.className = "stage__arrow";
    arrow.textContent = "\u25b6";

    head.append(num, title, badge, arrow);

    const body = document.createElement("div");
    body.className = "stage__body";
    const list = document.createElement("ul");
    stage.details.forEach((detail) => {
      const item = document.createElement("li");
      item.textContent = detail;
      list.appendChild(item);
    });
    body.appendChild(list);

    head.addEventListener("click", () => li.classList.toggle("stage--open"));

    li.append(head, body);
    container.appendChild(li);
  });
}

/** Навешивает обработчики модалок (презентация, закрытие). */
function initModals() {
  renderStages();

  $("#open-about").addEventListener("click", () => {
    $("#about-modal").classList.add("modal--visible");
  });

  $$("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.getElementById(btn.dataset.closeModal);
      if (!target) return;

      // Модалка прячется снятием modal--visible, а панель PDF — добавлением
      // pdf-panel--hidden (она встроена в layout, а не накрывает экран)
      if (target.id === "pdf-panel") {
        target.classList.add("pdf-panel--hidden");
      } else {
        target.classList.remove("modal--visible");
      }
    });
  });

  // Клик по затемнению закрывает модалку
  $$(".modal").forEach((modal) => {
    modal.addEventListener("click", (event) => {
      if (event.target === modal && modal.id !== "auth-modal") {
        modal.classList.remove("modal--visible");
      }
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      $("#about-modal").classList.remove("modal--visible");
      $("#pdf-panel").classList.add("pdf-panel--hidden");
    }
  });
}

// ============================================================
// ИНИЦИАЛИЗАЦИЯ
// ============================================================

/** Точка входа: настраивает всё и восстанавливает сессию. */
async function init() {
  if (window.pdfjsLib) {
    pdfjsLib.GlobalWorkerOptions.workerSrc =
      "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
  }

  initAuth();
  initComposer();
  initPdfPanel();
  initModals();
  bindSampleQuestions();

  const restored = await restoreSession();
  if (!restored) {
    $("#auth-modal").classList.add("modal--visible");
  }
}

document.addEventListener("DOMContentLoaded", init);
  if (!state.token) return false;
  try {
    const user = await api("/api/auth/me");
    $("#auth-modal").classList.remove("modal--visible");
    $("#app").classList.remove("app--hidden");
    $("#user-chip").textContent = user.username;
    await Promise.all([loadLessons(), loadChats()]);
    return true;
  } catch (error) {
    return false;
  }
}