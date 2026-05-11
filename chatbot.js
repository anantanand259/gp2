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
    RAG_BACKEND_URL: localStorage.getItem('gpa_rag_url') || 'http://localhost:5000',

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
    RAG_TIMEOUT: 15000,  // 15s timeout for RAG backend calls
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
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(`${CHATBOT_CONFIG.RAG_BACKEND_URL}/api/health`, {
                signal: controller.signal
            });
            clearTimeout(timeout);

            if (response.ok) {
                const data = await response.json();
                this.ragAvailable = true;
                console.log(`[Chatbot] ✅ RAG backend connected — ${data.total_chunks} chunks, ${data.total_documents} docs`);
            }
        } catch (e) {
            this.ragAvailable = false;
            console.log('[Chatbot] ⚠️ RAG backend not available — using LLM-only mode');
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
                if (e.key === 'Enter' && !e.shiftKey) {
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
            });
        }
    }

    // ─── Open / Close ───
    toggle() {
        if (this.isOpen) this.close();
        else this.open();
    }

    open() {
        this.isOpen = true;
        this.window.classList.add('open');
        this.trigger.classList.add('active');
        if (this.overlay) this.overlay.classList.add('show');
        if (this.badge) this.badge.style.display = 'none';

        // Hide trigger on mobile to prevent overlap
        if (window.innerWidth <= 576) {
            this.trigger.style.opacity = '0';
            this.trigger.style.pointerEvents = 'none';
        }

        setTimeout(() => {
            if (this.input) this.input.focus();
        }, 400);

        if (this.messagesContainer && this.messagesContainer.children.length === 0) {
            this.showWelcome();
        }
    }

    close() {
        this.isOpen = false;
        this.window.classList.remove('open');
        this.trigger.classList.remove('active');
        if (this.overlay) this.overlay.classList.remove('show');

        // Restore trigger on mobile
        this.trigger.style.opacity = '';
        this.trigger.style.pointerEvents = '';
    }

    // ─── Welcome Message ───
    showWelcome() {
        const ragBadge = this.ragAvailable
            ? '<span style="color:#4ade80;font-size:0.75rem;">● Knowledge Base Connected</span>'
            : '<span style="color:#facc15;font-size:0.75rem;">● LLM-Only Mode</span>';

        const welcomeHTML = `
            <div class="chatbot-welcome">
                <div class="chatbot-welcome-icon">🎓</div>
                <h4>Welcome to GPA Assistant!</h4>
                <p>I can help you with college information, placements, notices, syllabus, and more. I strictly answer questions based on the GPA Knowledge Base.</p>
                ${ragBadge}
            </div>
            <div class="chatbot-quick-actions" style="justify-content: center;">
                <button class="quick-action-btn" data-query="What departments are available?">
                    <i class="fas fa-building-columns"></i> Departments
                </button>
                <button class="quick-action-btn" data-query="Tell me about placements at GPA">
                    <i class="fas fa-briefcase"></i> Placements
                </button>
                <button class="quick-action-btn" data-query="What is the admission process?">
                    <i class="fas fa-user-plus"></i> Admissions
                </button>
                <button class="quick-action-btn" data-query="Show latest notices">
                    <i class="fas fa-bell"></i> Notices
                </button>
                <button class="quick-action-btn" data-query="Tell me about hostel facilities">
                    <i class="fas fa-bed"></i> Hostel
                </button>
                <button class="quick-action-btn" data-query="Who are the faculty members?">
                    <i class="fas fa-chalkboard-user"></i> Faculty
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
        const typingEl = this.showTyping();

        try {
            const response = await this.processQuery(text);
            this.removeTyping(typingEl);
            await this.addBotMessageAnimated(response.answer, response.source);
        } catch (error) {
            this.removeTyping(typingEl);
            let msg = error.message || 'Something went wrong. Please try again.';
            if (msg.includes('quota') || msg.includes('429') || msg.includes('rate')) {
                msg = 'The AI service is temporarily busy. Please wait a moment and try again.';
            } else if (msg.includes('API key')) {
                msg = 'API connection issue. Please try again shortly.';
            }
            this.showError(msg);
        }

        this.isProcessing = false;
        this.sendBtn.disabled = false;
        this.saveHistory();
    }

    async processQuery(query) {
        // ── STEP 1: Try RAG Backend for ALL queries ──
        if (this.ragAvailable) {
            try {
                console.log('[Chatbot] 🔍 Querying RAG backend...');
                const ragResult = await this._callRAGBackend(query);

                if (ragResult && ragResult.answer) {
                    // Use RAG answer if it found chunks OR if it explicitly says it's from internet
                    if (ragResult.chunk_count > 0 || ragResult.source_type === 'internet') {
                        if (!ragResult.answer.includes('specific information is not available')) {
                            console.log(`[Chatbot] ✅ RAG response — ${ragResult.source_type || 'kb'}`);
                            return {
                                answer: ragResult.answer,
                                source: ragResult.source_type === 'internet' ? 'llm' : 'rag',
                                sources: ragResult.sources || []
                            };
                        }
                    }
                }
                console.log('[Chatbot] ℹ️ RAG returned no relevant results, falling back to local LLM engine');
            } catch (e) {
                console.warn('[Chatbot] ⚠️ RAG backend error, falling back to LLM:', e.message);
            }
        }

        // ── STEP 2: Direct LLM call via proxy (Internet / General Knowledge Fallback) ──
        const prompt = `USER QUESTION: ${query}\n\nAnswer this question using your general knowledge, or if it relates to GPA college, use the quick facts from your system prompt.`;
        
        const answer = await this.callLLM(prompt);
        return { answer, source: 'llm' };
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
            clearTimeout(timeout);

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `RAG error: ${response.status}`);
            }

            return await response.json();
        } catch (e) {
            clearTimeout(timeout);
            if (e.name === 'AbortError') {
                throw new Error('RAG backend timed out');
            }
            throw e;
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
                    ${this.escapeHtml(text)}
                    <span class="msg-time">${timeStr}</span>
                </div>
            `;
        } else {
            msgDiv.innerHTML = `
                <div class="msg-avatar"><i class="fas fa-university"></i></div>
                <div class="msg-bubble">
                    ${this.formatMarkdown(text)}
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

        let sourceTag = '';
        if (source === 'rag') {
            sourceTag = `<div class="msg-source"><i class="fas fa-database"></i> From Knowledge Base</div>`;
        } else if (source === 'llm') {
            sourceTag = `<div class="msg-source" style="color: #6b7280; border-color: #d1d5db;"><i class="fas fa-globe"></i> From Internet / AI</div>`;
        }

        msgDiv.innerHTML = `
            <div class="msg-avatar"><i class="fas fa-university"></i></div>
            <div class="msg-bubble">
                <div class="msg-text-content"></div>
                ${sourceTag}
                <span class="msg-time">${timeStr}</span>
            </div>
        `;

        this.messagesContainer.appendChild(msgDiv);

        const textContainer = msgDiv.querySelector('.msg-text-content');
        await this.streamText(textContainer, text);
        this.scrollToBottom();
    }

    async streamText(container, text) {
        const formatted = this.formatMarkdown(text);
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = formatted;

        const fullText = tempDiv.innerHTML;
        let current = '';
        const speed = CHATBOT_CONFIG.TYPING_SPEED;
        const chunkSize = 3;

        for (let i = 0; i < fullText.length; i += chunkSize) {
            current += fullText.substring(i, i + chunkSize);
            container.innerHTML = current;
            this.scrollToBottom();

            if (fullText[i] !== '<') {
                await new Promise(r => setTimeout(r, speed));
            }
        }

        container.innerHTML = formatted;
    }

    // ─── Typing Indicator ───
    showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chatbot-typing';
        typingDiv.id = 'chatbotTyping';

        typingDiv.innerHTML = `
            <div class="msg-avatar"><i class="fas fa-university"></i></div>
            <div class="typing-dots">
                <span></span><span></span><span></span>
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
        errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${this.escapeHtml(message)}`;
        this.messagesContainer.appendChild(errorDiv);
        this.scrollToBottom();

        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.style.opacity = '0';
                errorDiv.style.transition = 'opacity 0.3s ease';
                setTimeout(() => errorDiv.remove(), 300);
            }
        }, 8000);
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
        this.input.style.height = Math.min(this.input.scrollHeight, 100) + 'px';
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
                const text = msg.querySelector('.msg-bubble')?.textContent?.trim() || '';
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
