/* ═══════════════════════════════════════════════════════════════
   GPA Chatbot – RAG Backend + Multi-LLM Engine
   Server-side RAG · Gemini via Proxy · Professional Chatbot
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────
const CHATBOT_CONFIG = {
    // ┌──────────────────────────────────────────────────────┐
    // │  🔑  API & BACKEND CONFIGURATION                     │
    // └──────────────────────────────────────────────────────┘

    // Cloudflare Worker proxy (handles Gemini & OpenRouter keys securely)
    API_PROXY_URL: 'https://gp2.anantanand259.workers.dev',

    // RAG Backend URL — Python server running locally or on a server
    // Preference: localStorage (set via Admin) > hardcoded default
    RAG_BACKEND_URL: localStorage.getItem('gpa_rag_url') || 'https://gp2.anantanand259.workers.dev',

    // ┌──────────────────────────────────────────────────────┐
    // │  🤖  MODEL CONFIGURATION                            │
    // └──────────────────────────────────────────────────────┘

    // Models: tried in order via proxy
    OPENROUTER_MODEL: 'meta-llama/llama-3.3-70b-instruct',
    GEMINI_MODELS: ['gemini-2.5-flash', 'gemini-2.5-flash-lite'],

    // ┌──────────────────────────────────────────────────────┐
    // │  ⚙️  ENGINE SETTINGS                                │
    // └──────────────────────────────────────────────────────┘
    MAX_RETRIES: 3,
    RETRY_BASE_DELAY: 2000,

    MAX_HISTORY: 10,
    TYPING_SPEED: 2,
    STORAGE_KEY_HISTORY: 'gpa_chatbot_history',

    // RAG settings
    RAG_TIMEOUT: 90000,  // Includes retrieval, grounded generation and provider retry
};

// System prompt shared by all LLM calls
const SYSTEM_PROMPT = `You are "GPA Assistant" — an intelligent, friendly chatbot for Government Polytechnic Adityapur (GPA), Jamshedpur, Jharkhand, India.

ROLE & BEHAVIOR:
- You help students, parents, and visitors with information about GPA college.
- When CONTEXT is provided from the knowledge base, use it to answer accurately. Cite the context.
- For college-related questions, if you don't have specific data in the context or quick facts, YOU MUST POLITELY DECLINE. Say "This specific information is not available in our knowledge base. Please contact the college directly or visit gpa.ac.in."
- You ARE ALLOWED to answer simple, general questions (math, science, general knowledge) using your own knowledge. 
- Always be professional, concise, and helpful.
- Format responses with markdown when helpful (bold, lists, etc.) but keep it readable.
- Keep responses under 300 words unless the question demands detail.

COLLEGE QUICK FACTS (always available):
- Name: Government Polytechnic Adityapur
- Established: 1980
- Location: Adityapur Industrial Area, Jamshedpur, Jharkhand – 832109
- Affiliation: JUT Ranchi (Jharkhand University of Technology)
- Approval: AICTE, New Delhi
- Departments: CSE, Mechanical, Electrical, Metallurgical (4 depts, 45 seats each)
- Duration: 3-year diploma programs
- Email: gpa2010@rediffmail.com
- Faculty count: 11
- Hostel capacity: 100 students
- Placement partners: Tata Steel, Wipro, JSPL, L&T
- Highest package: 7 LPA
- Placement rate: ~90%

IMPORTANT: Do NOT make up specific data about GPA (fees, exact dates, specific notices) unless it's in the provided context. For such queries without context, direct users to the official website gpa.ac.in or contact the college.`;

// Keywords that trigger RAG lookup
const COLLEGE_KEYWORDS = [
    'placement', 'notice', 'syllabus', 'department', 'faculty', 'admission',
    'hostel', 'fee', 'fees', 'scholarship', 'exam', 'result', 'calendar',
    'attendance', 'semester', 'subject', 'lab', 'library', 'ragging',
    'grievance', 'principal', 'lecturer', 'teacher', 'class', 'timetable',
    'branch', 'cse', 'mechanical', 'electrical', 'metallurgical', 'met',
    'gpa', 'polytechnic', 'adityapur', 'jamshedpur', 'jharkhand',
    'jut', 'aicte', 'sbte', 'pece', 'e-kalyan', 'ekalyan',
    'tender', 'circular', 'download', 'alumni', 'training', 'workshop',
    'college', 'institute', 'campus', 'infrastructure', 'about'
];


// ─────────────────────────────────────────────────────────────
// CHATBOT UI CONTROLLER
// ─────────────────────────────────────────────────────────────
class ChatbotUI {
    constructor() {
        this.isOpen = false;
        this.isProcessing = false;
        this.conversationHistory = [];
        this.ragAvailable = false;

        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        console.log('[Chatbot] Initializing GPA Assistant...');
        
        // Refresh RAG URL from localStorage in case it was changed in Admin
        CHATBOT_CONFIG.RAG_BACKEND_URL = localStorage.getItem('gpa_rag_url') || CHATBOT_CONFIG.RAG_BACKEND_URL;
        console.log(`[Chatbot] Using RAG Backend: ${CHATBOT_CONFIG.RAG_BACKEND_URL}`);

        this.bindElements();
        this.bindEvents();
        this.loadHistory();
        this.checkRAGHealth();
    }

    // ─── Check if RAG backend is available ───
    async checkRAGHealth() {
        let timeout;
        try {
            const controller = new AbortController();
            timeout = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(`${CHATBOT_CONFIG.RAG_BACKEND_URL}/api/health`, {
                signal: controller.signal
            });
            clearTimeout(timeout);
            this.ragAvailable = false;

            if (response.ok) {
                const data = await response.json();
                this.ragAvailable = data.retriever_ready === true;
                console.log(`[Chatbot] ✅ RAG backend connected — ${data.total_chunks} chunks, ${data.total_documents} docs`);
            }
        } catch (e) {
            this.ragAvailable = false;
            console.log('[Chatbot] ⚠️ RAG backend unavailable; queries will retry the backend');
        } finally {
            clearTimeout(timeout);
            this.updateServiceStatus();
        }
    }

    bindElements() {
        this.trigger = document.getElementById('chatbotTrigger');
        this.window = document.getElementById('chatbotWindow');
        this.overlay = document.getElementById('chatbotOverlay');
        this.messagesContainer = document.getElementById('chatbotMessages');
        this.input = document.getElementById('chatbotInput');
        this.sendBtn = document.getElementById('chatbotSendBtn');
        this.closeBtn = document.getElementById('chatbotCloseBtn');
        this.clearBtn = document.getElementById('chatbotClearBtn');
        this.adminBtn = document.getElementById('chatbotAdminBtn');
        this.badge = document.getElementById('chatbotBadge');
        this.statusText = document.getElementById('chatbotStatusText');
        this.serviceStatus = document.getElementById('chatbotServiceStatus');
    }

    bindEvents() {
        if (!this.trigger || !this.window) return;

        // Open/close
        this.trigger.addEventListener('click', () => this.toggle());
        if (this.closeBtn) this.closeBtn.addEventListener('click', () => this.close());
        if (this.overlay) this.overlay.addEventListener('click', () => this.close());

        // Send message
        if (this.sendBtn) this.sendBtn.addEventListener('click', () => this.handleSend());
        if (this.input) {
            this.input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
                    e.preventDefault();
                    this.handleSend();
                }
            });
            this.input.addEventListener('input', () => this.autoResizeInput());
        }

        // Clear chat
        if (this.clearBtn) this.clearBtn.addEventListener('click', () => this.clearChat());

        // Admin portal
        if (this.adminBtn) {
            this.adminBtn.addEventListener('click', () => {
                window.open('admin.html', '_blank');
            });
        }

        // Escape to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) this.close();
            if (e.key === 'Tab' && this.isOpen) {
                const controls = [...this.window.querySelectorAll('button, a[href], textarea')]
                    .filter(el => !el.disabled && el.getClientRects().length);
                const first = controls[0], last = controls[controls.length - 1];
                if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
                else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
            }
        });
        window.addEventListener('resize', () => {
            if (!this.isOpen) return;
            const mobile = window.innerWidth <= 576;
            this.trigger.style.opacity = mobile ? '0' : '';
            this.trigger.style.pointerEvents = mobile ? 'none' : '';
        });

        // Quick action buttons (delegated)
        if (this.messagesContainer) {
            this.messagesContainer.addEventListener('click', (e) => {
                const btn = e.target.closest('.quick-action-btn');
                if (btn) {
                    const query = btn.dataset.query;
                    if (query) {
                        this.input.value = query;
                        this.handleSend();
                    }
                }
                const copy = e.target.closest('.msg-copy');
                if (copy) this.copyAnswer(copy);
            });
        }
    }

    // ─── Open / Close ───
    toggle() {
        if (this.isOpen) this.close();
        else this.open();
    }

    open() {
        if (this.isOpen) return;
        this.isOpen = true;
        this.window.inert = false;
        this.window.classList.add('open');
        this.trigger.classList.add('active');
        this.trigger.setAttribute('aria-expanded', 'true');
        this.trigger.setAttribute('aria-label', 'Close GPA Assistant chatbot');
        this.previousOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        this.backgroundInertState = [...document.body.children]
            .filter(el => ![this.window, this.overlay, this.trigger].includes(el) && !['SCRIPT', 'STYLE', 'LINK'].includes(el.tagName))
            .map(el => ({ el, inert: el.inert }));
        this.backgroundInertState.forEach(({ el }) => { el.inert = true; });
        if (this.overlay) this.overlay.classList.add('show');
        if (this.badge) this.badge.style.display = 'none';

        // Hide trigger on mobile to prevent overlap
        if (window.innerWidth <= 576) {
            this.trigger.style.opacity = '0';
            this.trigger.style.pointerEvents = 'none';
        }

        setTimeout(() => {
            if (this.isOpen) (window.innerWidth <= 576 ? this.closeBtn : this.input)?.focus();
        }, 180);

        if (this.messagesContainer && this.messagesContainer.children.length === 0) {
            this.showWelcome();
        }
    }

    close() {
        this.isOpen = false;
        this.window.classList.remove('open');
        this.trigger.classList.remove('active');
        if (this.overlay) this.overlay.classList.remove('show');
        this.window.inert = true;
        this.trigger.setAttribute('aria-expanded', 'false');
        this.trigger.setAttribute('aria-label', 'Open GPA Assistant chatbot');
        document.body.style.overflow = this.previousOverflow || '';
        (this.backgroundInertState || []).forEach(({ el, inert }) => { el.inert = inert; });

        // Restore trigger on mobile
        this.trigger.style.opacity = '';
        this.trigger.style.pointerEvents = '';
        this.trigger.focus();
    }

    updateServiceStatus() {
        if (this.statusText) this.statusText.textContent = this.ragAvailable ? 'Connected to GPA' : 'Connection unavailable';
        if (this.serviceStatus) this.serviceStatus.classList.toggle('unavailable', !this.ragAvailable);
    }

    // ─── Welcome Message ───
    showWelcome() {
        const welcomeHTML = `
            <div class="chatbot-welcome">
                <div class="chatbot-welcome-icon"><img src="assets/gpa-assistant-mark.svg" alt="" width="68" height="68"></div>
                <span class="chatbot-eyebrow">YOUR CAMPUS COMPANION</span>
                <h3>College updates,<br>made clear.</h3>
                <p>Understand notices, check dates and find campus information. Ask in English or Hindi.</p>
            </div>
            <div class="chatbot-starter-label">Start with a question</div>
            <div class="chatbot-quick-actions">
                <button class="quick-action-btn" data-query="Summarize the latest college notices and any important deadlines.">
                    <i class="fas fa-file-lines" aria-hidden="true"></i><span><strong>Latest notices</strong><small>What should I know?</small></span><span class="starter-arrow" aria-hidden="true">↗</span>
                </button>
                <button class="quick-action-btn" data-query="What does the knowledge base say about the exam schedule and registration deadlines?">
                    <i class="fas fa-calendar-days" aria-hidden="true"></i><span><strong>Exams & dates</strong><small>Plan your next step</small></span><span class="starter-arrow" aria-hidden="true">↗</span>
                </button>
                <button class="quick-action-btn" data-query="What is the admission process?">
                    <i class="fas fa-graduation-cap" aria-hidden="true"></i><span><strong>Admissions</strong><small>Find your way to GPA</small></span><span class="starter-arrow" aria-hidden="true">↗</span>
                </button>
                <button class="quick-action-btn" data-query="Tell me about campus facilities and hostel information.">
                    <i class="fas fa-building-columns" aria-hidden="true"></i><span><strong>Campus life</strong><small>Facilities & student help</small></span><span class="starter-arrow" aria-hidden="true">↗</span>
                </button>
            </div>
        `;

        const welcomeDiv = document.createElement('div');
        welcomeDiv.innerHTML = welcomeHTML;
        welcomeDiv.className = 'chatbot-welcome-wrapper';
        this.messagesContainer.appendChild(welcomeDiv);
    }

    // ─── Send Message ───
    async handleSend() {
        if (this.isProcessing) return;

        const text = (this.input.value || '').trim();
        if (!text) return;
        this.lastQuery = text;

        // Clear welcome if present
        const welcome = this.messagesContainer.querySelector('.chatbot-welcome-wrapper');
        if (welcome) {
            welcome.style.opacity = '0';
            welcome.style.transform = 'translateY(-10px)';
            welcome.style.transition = 'all 0.3s ease';
            setTimeout(() => welcome.remove(), 300);
        }

        // Add user message
        this.addMessage(text, 'user');
        this.input.value = '';
        this.autoResizeInput();

        // Process
        this.isProcessing = true;
        this.sendBtn.disabled = true;
        this.messagesContainer.setAttribute('aria-busy', 'true');
        const typingEl = this.showTyping();

        try {
            const response = await this.processQuery(text);
            this.removeTyping(typingEl);
            this.messagesContainer.setAttribute('aria-busy', 'false');
            await this.addBotMessageAnimated(response.answer, response.source);
        } catch (error) {
            this.removeTyping(typingEl);
            let msg = error.message || 'Something went wrong. Please try again.';
            if (!error.backendReached && (msg.includes('quota') || msg.includes('429') || msg.includes('rate'))) {
                msg = 'The AI service is temporarily busy. Please wait a moment and try again.';
            } else if (!error.backendReached && msg.includes('API key')) {
                msg = 'API connection issue. Please try again shortly.';
            }
            this.showError(msg);
        }

        this.isProcessing = false;
        this.sendBtn.disabled = false;
        this.messagesContainer.setAttribute('aria-busy', 'false');
        this.updateServiceStatus();
        this.saveHistory();
    }

    async processQuery(query) {
        const conversation = this.conversationalReply(query);
        if (conversation) return { answer: conversation, source: 'conversation', sources: [] };
        // Only the backend may decide that the KB lacks an answer. A network
        // error, health-check race, or refusal must never bypass uploaded notices.
        CHATBOT_CONFIG.RAG_BACKEND_URL = localStorage.getItem('gpa_rag_url') || CHATBOT_CONFIG.RAG_BACKEND_URL;
        try {
            const result = await this._callRAGBackend(query);
            if (!result || typeof result.answer !== 'string' || !result.answer.trim()) {
                throw new Error('Invalid response from knowledge base');
            }
            this.ragAvailable = true;
            return {
                answer: result.answer,
                source: result.source_type === 'internet' ? 'internet' :
                    result.source_type === 'rag' ? 'rag' : 'none',
                sources: result.sources || []
            };
        } catch (error) {
            this.ragAvailable = error.backendReached === true;
            console.warn('[Chatbot] Knowledge base query failed:', error.message);
            if (error.backendReached) throw error;
            throw new Error('I could not check the college knowledge base. Please try again when the backend is available.');
        }
    }

    conversationalReply(query) {
        const text = query.normalize('NFKC').trim().toLowerCase().replace(/[.!?,;:।🙏👋😊]+$/gu, '').trim();
        if (/^(?:h+i+|hello+|hey+|hlw|hlo|helo|hellow|good (?:morning|afternoon|evening))(?:\s+(?:gpa|assistant|bot))?$/u.test(text)) {
            return 'Hi! Welcome to GPA Assistant. How can I help you with college notices or information today?';
        }
        if (['नमस्ते', 'नमस्कार', 'हाय', 'हेलो'].includes(text)) return 'नमस्ते! GPA Assistant में आपका स्वागत है। कॉलेज की सूचनाओं या जानकारी के बारे में मैं आपकी कैसे मदद कर सकता हूँ?';
        if (['thanks', 'thank you', 'thankyou', 'thx', 'ty', 'धन्यवाद', 'शुक्रिया'].includes(text)) return 'You’re welcome! Let me know if you need help with anything else about GPA.';
        if (['how are you', 'how are you doing', 'how r u'].includes(text)) return 'I’m here and ready to help! What would you like to know about GPA?';
        if (['help', 'help me', 'what can you do', 'who are you'].includes(text)) return 'I’m GPA Assistant. I can help you understand uploaded college notices, find deadlines, and ask about admissions, exams or campus facilities. Ask in English or Hindi.';
        if (['bye', 'goodbye', 'see you', 'good night', 'goodnight'].includes(text)) return 'Goodbye! You can come back whenever you need help with GPA notices or college information.';
        return null;
    }

    // ─── Call RAG Backend ───
    async _callRAGBackend(query) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), CHATBOT_CONFIG.RAG_TIMEOUT);

        try {
            const response = await fetch(`${CHATBOT_CONFIG.RAG_BACKEND_URL}/api/rag/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
                signal: controller.signal
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                const rawMessage = typeof err.error === 'string' ? err.error : '';
                let message = rawMessage || `Knowledge-base request failed (HTTP ${response.status}).`;
                // Also recognize authentication errors from older running servers.
                if (/api.?key.*(?:invalid|not valid)|API_KEY_INVALID|unauthorized|authentication/i.test(rawMessage)) {
                    message = 'The knowledge base is reachable, but its AI provider API key is missing or invalid. Ask the administrator to update the backend key and restart the Python server.';
                }
                const error = new Error(message);
                error.backendReached = true;
                error.code = err.code || 'RAG_REQUEST_ERROR';
                throw error;
            }

            return await response.json();
        } catch (e) {
            if (e.name === 'AbortError') {
                throw new Error('RAG backend timed out');
            }
            throw e;
        } finally {
            clearTimeout(timeout);
        }
    }

    // ─── Multi-LLM Call via Proxy (Gemini models) ───
    async callLLM(prompt) {
        const recentHistory = this.conversationHistory.slice(-CHATBOT_CONFIG.MAX_HISTORY);

        // ── Attempt 1: OpenRouter (Llama 3.3 70B) via Proxy ──
        try {
            console.log(`[Chatbot] Trying OpenRouter model: ${CHATBOT_CONFIG.OPENROUTER_MODEL}...`);
            const text = await this._callOpenRouterViaProxy(CHATBOT_CONFIG.OPENROUTER_MODEL, prompt, recentHistory);
            this._pushHistory(prompt, text);
            return text;
        } catch (err) {
            console.warn('[Chatbot] OpenRouter failed, falling back to Gemini:', err.message);
        }

        // ── Attempt 2: Gemini models fallback ──
        const geminiBody = this._buildGeminiBody(prompt, recentHistory);
        let lastError = null;
        for (const model of CHATBOT_CONFIG.GEMINI_MODELS) {
            try {
                console.log(`[Chatbot] Trying Gemini model: ${model}...`);
                const text = await this._callGeminiViaProxy(model, geminiBody);
                this._pushHistory(prompt, text);
                return text;
            } catch (err) {
                lastError = err;
                const isQuota = err.message?.includes('429') || err.message?.includes('quota') || err.message?.includes('RESOURCE_EXHAUSTED');
                if (isQuota) {
                    console.warn(`[Chatbot] ${model} quota exhausted, trying next...`);
                    continue;
                }
                throw err;
            }
        }

        throw lastError || new Error('All models are currently unavailable. Please try again later.');
    }

    // ─── Push to conversation history ───
    _pushHistory(prompt, text) {
        this.conversationHistory.push({ role: 'user', text: prompt.substring(0, 500) });
        this.conversationHistory.push({ role: 'assistant', text: text.substring(0, 500) });
        if (this.conversationHistory.length > CHATBOT_CONFIG.MAX_HISTORY * 2) {
            this.conversationHistory = this.conversationHistory.slice(-CHATBOT_CONFIG.MAX_HISTORY * 2);
        }
    }

    // ─── Build Gemini request body ───
    _buildGeminiBody(prompt, history) {
        const contents = [];
        for (const msg of history) {
            contents.push({
                role: msg.role === 'user' ? 'user' : 'model',
                parts: [{ text: msg.text }]
            });
        }
        contents.push({ role: 'user', parts: [{ text: prompt }] });

        return {
            contents,
            systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
            generationConfig: {
                temperature: 0.7,
                topP: 0.9,
                topK: 40,
                maxOutputTokens: 1024,
            },
            safetySettings: [
                { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_ONLY_HIGH" },
                { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_ONLY_HIGH" },
                { category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold: "BLOCK_ONLY_HIGH" },
                { category: "HARM_CATEGORY_DANGEROUS_CONTENT", threshold: "BLOCK_ONLY_HIGH" }
            ]
        };
    }

    // ─── Call OpenRouter via Cloudflare Worker Proxy ───
    async _callOpenRouterViaProxy(model, prompt, history) {
        // Convert history to OpenAI format
        const messages = history.map(m => ({
            role: m.role === 'assistant' ? 'assistant' : 'user',
            content: m.text
        }));
        messages.push({ role: 'system', content: SYSTEM_PROMPT });
        messages.push({ role: 'user', content: prompt });

        const response = await fetch(`${CHATBOT_CONFIG.API_PROXY_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model,
                messages,
                temperature: 0.7
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData?.error || `OpenRouter proxy error: ${response.status}`);
        }

        const data = await response.json();
        // The proxy returns Gemini-compatible format even for OpenRouter calls
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
        
        if (!text) throw new Error('Empty response from OpenRouter.');
        return text;
    }

    // ─── Call Gemini via Cloudflare Worker Proxy ───
    async _callGeminiViaProxy(model, body) {
        const maxRetries = CHATBOT_CONFIG.MAX_RETRIES;
        let lastError = null;

        for (let attempt = 0; attempt < maxRetries; attempt++) {
            try {
                const response = await fetch(`${CHATBOT_CONFIG.API_PROXY_URL}/api/gemini`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, ...body })
                });

                if (response.status === 429) {
                    if (attempt < maxRetries - 1) {
                        const delay = CHATBOT_CONFIG.RETRY_BASE_DELAY * Math.pow(2, attempt);
                        console.warn(`[Gemini] Rate limited on ${model}, retrying in ${delay}ms...`);
                        await this._sleep(delay);
                        continue;
                    }
                    throw new Error(`429 quota exceeded for ${model}`);
                }

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData?.error || `Proxy error: ${response.status}`);
                }

                const data = await response.json();

                const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
                if (!text) throw new Error('Empty response from AI.');

                console.log(`[Chatbot] ✅ Response from Gemini ${model}`);
                return text;

            } catch (err) {
                lastError = err;
                if (!err.message?.includes('429')) throw err;
            }
        }

        throw lastError;
    }

    // ─── Async sleep helper ───
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ─── Message Rendering ───
    addMessage(text, role) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chatbot-msg ${role}`;

        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

        if (role === 'user') {
            msgDiv.innerHTML = `
                <div class="msg-bubble">
                    <div class="msg-text-content">${this.escapeHtml(text)}</div>
                    <span class="msg-time">${timeStr}</span>
                </div>
            `;
        } else {
            msgDiv.innerHTML = `
                <div class="msg-avatar" aria-hidden="true"><img src="assets/gpa-assistant-mark.svg" alt=""></div>
                <div class="msg-bubble">
                    <div class="msg-text-content">${this.formatMarkdown(text)}</div>
                    <span class="msg-time">${timeStr}</span>
                </div>
            `;
        }

        this.messagesContainer.appendChild(msgDiv);
        this.scrollToBottom();
    }

    async addBotMessageAnimated(text, source) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chatbot-msg bot';

        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

        msgDiv.innerHTML = `
            <div class="msg-avatar" aria-hidden="true"><img src="assets/gpa-assistant-mark.svg" alt=""></div>
            <div class="msg-bubble">
                <div class="msg-text-content"></div>
                <div class="msg-footer"><span class="msg-time">${timeStr}</span><button class="msg-copy" type="button" aria-label="Copy answer"><i class="far fa-copy" aria-hidden="true"></i><span>Copy</span></button></div>
            </div>
        `;

        this.messagesContainer.appendChild(msgDiv);

        const textContainer = msgDiv.querySelector('.msg-text-content');
        await this.streamText(textContainer, text);
        this.scrollToBottom();
    }

    async copyAnswer(button) {
        const text = button.closest('.msg-bubble').querySelector('.msg-text-content').textContent;
        const label = button.querySelector('span');
        try {
            await navigator.clipboard.writeText(text);
            label.textContent = 'Copied';
            button.setAttribute('aria-label', 'Answer copied');
        } catch (_) {
            label.textContent = 'Select text to copy';
        }
        setTimeout(() => { label.textContent = 'Copy'; button.setAttribute('aria-label', 'Copy answer'); }, 2200);
    }

    async streamText(container, text) {
        // Render the complete answer once: readable formatting and one screen-reader update.
        container.innerHTML = this.formatMarkdown(text);
    }

    // ─── Typing Indicator ───
    showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chatbot-typing';
        typingDiv.id = 'chatbotTyping';
        typingDiv.setAttribute('role', 'status');

        typingDiv.innerHTML = `
            <div class="msg-avatar" aria-hidden="true"><img src="assets/gpa-assistant-mark.svg" alt=""></div>
            <div class="typing-dots">
                <span aria-hidden="true"></span><span aria-hidden="true"></span><span aria-hidden="true"></span><small>Finding an answer…</small>
            </div>
        `;

        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
        return typingDiv;
    }

    removeTyping(el) {
        if (el && el.parentNode) {
            el.style.opacity = '0';
            el.style.transform = 'translateY(-8px)';
            el.style.transition = 'all 0.2s ease';
            setTimeout(() => el.remove(), 200);
        }
    }

    // ─── Error ───
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chatbot-error';
        errorDiv.setAttribute('role', 'alert');
        errorDiv.innerHTML = `<i class="fas fa-exclamation-circle" aria-hidden="true"></i><div><strong>Let’s try that again</strong><p>${this.escapeHtml(message)}</p><button type="button" class="chatbot-retry">Retry question <span aria-hidden="true">↗</span></button></div>`;
        const query = this.lastQuery;
        errorDiv.querySelector('button').addEventListener('click', () => {
            if (this.isProcessing) return;
            errorDiv.remove();
            this.input.value = query || '';
            this.handleSend();
        });
        this.messagesContainer.appendChild(errorDiv);
        this.scrollToBottom();

    }

    // ─── Helpers ───
    scrollToBottom() {
        if (this.messagesContainer) {
            requestAnimationFrame(() => {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            });
        }
    }

    autoResizeInput() {
        if (!this.input) return;
        this.input.style.height = 'auto';
        this.input.style.height = Math.min(this.input.scrollHeight + 2, 100) + 'px';
        this.input.style.overflowY = this.input.scrollHeight > 100 ? 'auto' : 'hidden';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatMarkdown(text) {
        if (!text) return '';

        let html = this.escapeHtml(text);

        // Bold **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italic *text*
        html = html.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

        // Inline code `text`
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');

        // Unordered lists
        html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Ordered lists
        html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

        // Line breaks
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');

        // Wrap in paragraph
        if (!html.startsWith('<')) {
            html = '<p>' + html + '</p>';
        }

        return html;
    }

    clearChat() {
        if (this.isProcessing) return;
        if (this.messagesContainer) {
            this.messagesContainer.innerHTML = '';
        }
        this.conversationHistory = [];
        localStorage.removeItem(CHATBOT_CONFIG.STORAGE_KEY_HISTORY);
        this.showWelcome();
    }

    saveHistory() {
        try {
            const messages = [];
            this.messagesContainer.querySelectorAll('.chatbot-msg').forEach(msg => {
                const role = msg.classList.contains('user') ? 'user' : 'bot';
                const text = msg.querySelector('.msg-text-content')?.textContent?.trim() || '';
                if (text) messages.push({ role, text: text.substring(0, 300) });
            });
            const toStore = messages.slice(-20);
            localStorage.setItem(CHATBOT_CONFIG.STORAGE_KEY_HISTORY, JSON.stringify(toStore));
        } catch (e) {
            // localStorage may be full
        }
    }

    loadHistory() {
        try {
            const stored = JSON.parse(localStorage.getItem(CHATBOT_CONFIG.STORAGE_KEY_HISTORY) || '[]');
            this.conversationHistory = stored.map(m => ({
                role: m.role === 'user' ? 'user' : 'assistant',
                text: m.text
            })).slice(-CHATBOT_CONFIG.MAX_HISTORY);
        } catch (e) {
            this.conversationHistory = [];
        }
    }
}


// ─────────────────────────────────────────────────────────────
// INITIALIZE CHATBOT
// ─────────────────────────────────────────────────────────────
const gpaChatbot = new ChatbotUI();
