/* ═══════════════════════════════════════════════════════════════
   GPA Chatbot – RAG + Multi-LLM Engine
   Client-side RAG · Proxy → OpenRouter (Qwen) + Gemini Fallback
   All API keys are stored server-side in the Cloudflare Worker proxy.
   ═══════════════════════════════════════════════════════════════ */



// ─────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────
const CHATBOT_CONFIG = {
    API_PROXY_URL: 'https://gpa-api-proxy.anantanand259.workers.dev',

    // Stable fallback model (NOT elephant)
    OPENROUTER_MODEL: 'meta-llama/llama-3.3-70b-instruct:free',

    // Updated Gemini models
    GEMINI_MODELS: ['gemini-2.5-flash', 'gemini-2.5-flash-lite'],

    EMBED_MODEL: 'text-embedding-004',

    MAX_RETRIES: 3,
    RETRY_BASE_DELAY: 2000,
    CHUNK_SIZE: 500,
    CHUNK_OVERLAP: 80,
    TOP_K_RESULTS: 3,
    SIMILARITY_THRESHOLD: 0.35,
    MAX_HISTORY: 10,
    TYPING_SPEED: 12,
    
    // Kept to prevent local storage crashes:
    STORAGE_KEY_KB: 'gpa_chatbot_kb',
    STORAGE_KEY_HISTORY: 'gpa_chatbot_history',
    STORAGE_KEY_ADMIN: 'gpa_chatbot_admin',
};

// System prompt for the Gemini model
const SYSTEM_PROMPT = `You are "GPA Assistant" — an intelligent, friendly chatbot for Government Polytechnic Adityapur (GPA), Jamshedpur, Jharkhand, India.

ROLE & BEHAVIOR:
- You help students, parents, and visitors with information about GPA college.
- When CONTEXT is provided, use it to answer accurately. Cite the context.
- When NO context is relevant, answer general/universal questions using your own knowledge.
- You can solve basic math problems (arithmetic, algebra, percentages, etc.).
- Always be polite, concise, and helpful.
- If you don't know something college-specific and no context is given, say so honestly and suggest contacting the college.
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

// Keywords for RAG routing
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
        this.ragEngine = new RAGEngine();

        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.init());
        } else {
            this.init();
        }
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadHistory();
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
            // Auto-resize textarea
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

        // Focus input after animation
        setTimeout(() => {
            if (this.input) this.input.focus();
        }, 450);

        // Show welcome if first time
        if (this.messagesContainer && this.messagesContainer.children.length === 0) {
            this.showWelcome();
        }
    }

    close() {
        this.isOpen = false;
        this.window.classList.remove('open');
        this.trigger.classList.remove('active');
        if (this.overlay) this.overlay.classList.remove('show');
    }

    // ─── Welcome Message ───
    showWelcome() {
        const welcomeHTML = `
            <div class="chatbot-welcome">
                <div class="chatbot-welcome-icon">🎓</div>
                <h4>Welcome to GPA Assistant!</h4>
                <p>I can help you with college information, placements, notices, syllabus, and more. I can also answer general questions & solve math!</p>
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

        // Show typing indicator
        const typingEl = this.showTyping();

        try {
            const response = await this.processQuery(text);
            this.removeTyping(typingEl);
            await this.addBotMessageAnimated(response.answer, response.source);
        } catch (error) {
            this.removeTyping(typingEl);
            // Show user-friendly messages for common errors
            let msg = error.message || 'Something went wrong. Please try again.';
            if (msg.includes('quota') || msg.includes('429') || msg.includes('rate')) {
                msg = 'The AI service is temporarily busy. Please wait a moment and try again.';
            } else if (msg.includes('Forbidden') || msg.includes('403')) {
                msg = 'API connection issue (CORS). Please try again shortly.';
            } else if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
                msg = 'Network error. Please check your internet connection.';
            }
            console.error('[Chatbot] Error:', error);
            this.showError(msg);
        }

        this.isProcessing = false;
        this.sendBtn.disabled = false;
        this.saveHistory();
    }

    // ─── Process Query (RAG + LLM Router) ───
    async processQuery(query) {
        const lowerQuery = query.toLowerCase();

        // Check if it's a college-related query
        const isCollegeQuery = COLLEGE_KEYWORDS.some(kw => lowerQuery.includes(kw));

        let context = '';
        let source = 'llm';

        // Try RAG for college-related queries
        if (isCollegeQuery) {
            try {
                const ragResults = await this.ragEngine.search(query);
                if (ragResults && ragResults.length > 0) {
                    context = ragResults.map((r, i) => `[Source ${i + 1}]: ${r.text}`).join('\n\n');
                    source = 'rag';
                }
            } catch (e) {
                console.warn('RAG search failed, falling back to LLM:', e);
            }
        }

        // Build prompt
        let prompt = '';
        if (context) {
            prompt = `CONTEXT FROM KNOWLEDGE BASE:\n${context}\n\nUSER QUESTION: ${query}\n\nAnswer the question using the provided context. If the context doesn't fully answer the question, supplement with your knowledge but clarify what came from context vs your knowledge.`;
        } else {
            prompt = `USER QUESTION: ${query}\n\nAnswer this question. If it's about Government Polytechnic Adityapur specifically and you don't have the specific data, suggest checking the official website or contacting the college.`;
        }

        // Call LLM: try Qwen (OpenRouter) first, then Gemini fallback
        const answer = await this.callLLM(prompt);
        return { answer, source };
    }

    // ─── FINAL WORKING callLLM() (USE THIS) ───
    async callLLM(prompt) {
        const recentHistory = this.conversationHistory.slice(-CHATBOT_CONFIG.MAX_HISTORY);

        const body = this._buildGeminiBody(prompt, recentHistory);

        for (const model of CHATBOT_CONFIG.GEMINI_MODELS) {
            try {
                const text = await this._callGeminiViaProxy(model, body);
                this._pushHistory(prompt, text);
                return text;
            } catch (err) {
                console.warn(`Gemini ${model} failed`);
            }
        }

        throw new Error("AI is busy. Try again.");
    }

    // ─── Push to conversation history ───
    _pushHistory(prompt, text) {
        this.conversationHistory.push({ role: 'user', text: prompt.substring(0, 500) });
        this.conversationHistory.push({ role: 'assistant', text: text.substring(0, 500) });
        if (this.conversationHistory.length > CHATBOT_CONFIG.MAX_HISTORY * 2) {
            this.conversationHistory = this.conversationHistory.slice(-CHATBOT_CONFIG.MAX_HISTORY * 2);
        }
    }



    // ─── Gemini via Proxy ───
    async _callGeminiViaProxy(model, body) {
        const response = await fetch(`${CHATBOT_CONFIG.API_PROXY_URL}/api/gemini`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...body, model })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || 'Gemini proxy error');
        }

        const data = await response.json();
        const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;

        if (!text) throw new Error('Empty response');

        return text;
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

        const sourceTag = source === 'rag'
            ? `<div class="msg-source"><i class="fas fa-database"></i> From Knowledge Base</div>`
            : '';

        msgDiv.innerHTML = `
            <div class="msg-avatar"><i class="fas fa-university"></i></div>
            <div class="msg-bubble">
                <div class="msg-text-content"></div>
                ${sourceTag}
                <span class="msg-time">${timeStr}</span>
            </div>
        `;

        this.messagesContainer.appendChild(msgDiv);

        // Streaming animation
        const textContainer = msgDiv.querySelector('.msg-text-content');
        await this.streamText(textContainer, text);
        this.scrollToBottom();
    }

    async streamText(container, text) {
        const formatted = this.formatMarkdown(text);

        // Parse HTML and stream it
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = formatted;

        // Stream by adding HTML character by character
        const fullText = tempDiv.innerHTML;
        let current = '';
        const speed = CHATBOT_CONFIG.TYPING_SPEED;

        // Use a faster approach: add chunks
        const chunkSize = 3;
        for (let i = 0; i < fullText.length; i += chunkSize) {
            current += fullText.substring(i, i + chunkSize);
            container.innerHTML = current;
            this.scrollToBottom();

            // Small delay for streaming effect but skip for HTML tags
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

        // Auto-remove after 8s
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
            // Only keep last 20 messages for storage
            const toStore = messages.slice(-20);
            localStorage.setItem(CHATBOT_CONFIG.STORAGE_KEY_HISTORY, JSON.stringify(toStore));
        } catch (e) {
            // localStorage may be full
        }
    }

    loadHistory() {
        // We don't restore visual messages on reload, just start fresh
        // But we keep conversation context for better responses
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
// RAG ENGINE — Client-side Retrieval Augmented Generation
// ─────────────────────────────────────────────────────────────
class RAGEngine {
    constructor() {
        this.knowledgeBase = this.loadKnowledgeBase();
    }

    // ─── Load KB from localStorage ───
    loadKnowledgeBase() {
        try {
            const stored = localStorage.getItem(CHATBOT_CONFIG.STORAGE_KEY_KB);
            if (stored) {
                return JSON.parse(stored);
            }
        } catch (e) {
            console.warn('Failed to load knowledge base:', e);
        }
        return { entries: [], chunks: [] };
    }

    // ─── Save KB to localStorage ───
    saveKnowledgeBase() {
        try {
            localStorage.setItem(CHATBOT_CONFIG.STORAGE_KEY_KB, JSON.stringify(this.knowledgeBase));
        } catch (e) {
            console.error('Failed to save knowledge base:', e);
            throw new Error('Storage full. Please delete some entries.');
        }
    }

    // ─── Add Entry ───
    async addEntry(title, content) {
        const id = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
        const entry = { id, title, content, createdAt: new Date().toISOString() };

        // Chunk the content
        const chunks = this.chunkText(content, CHATBOT_CONFIG.CHUNK_SIZE, CHATBOT_CONFIG.CHUNK_OVERLAP);

        // Generate embeddings for each chunk
        const embeddedChunks = [];
        for (const chunk of chunks) {
            const contextualChunk = `${title}: ${chunk}`;
            try {
                const embedding = await this.getEmbedding(contextualChunk);
                embeddedChunks.push({
                    entryId: id,
                    text: contextualChunk,
                    embedding: embedding
                });
            } catch (e) {
                console.error('Embedding failed for chunk:', e);
                // Store without embedding, will use keyword search
                embeddedChunks.push({
                    entryId: id,
                    text: contextualChunk,
                    embedding: null
                });
            }
        }

        this.knowledgeBase.entries.push(entry);
        this.knowledgeBase.chunks.push(...embeddedChunks);
        this.saveKnowledgeBase();

        return { entry, chunksCount: embeddedChunks.length };
    }

    // ─── Delete Entry ───
    deleteEntry(entryId) {
        this.knowledgeBase.entries = this.knowledgeBase.entries.filter(e => e.id !== entryId);
        this.knowledgeBase.chunks = this.knowledgeBase.chunks.filter(c => c.entryId !== entryId);
        this.saveKnowledgeBase();
    }

    // ─── Update Entry ───
    async updateEntry(entryId, title, content) {
        this.deleteEntry(entryId);
        return await this.addEntry(title, content);
    }

    // ─── Search (Semantic + Keyword fallback) ───
    async search(query) {
        if (!this.knowledgeBase.chunks || this.knowledgeBase.chunks.length === 0) {
            return [];
        }

        let results = [];

        // Try semantic search first
        const hasEmbeddings = this.knowledgeBase.chunks.some(c => c.embedding !== null);

        if (hasEmbeddings) {
            try {
                const queryEmbedding = await this.getEmbedding(query);
                results = this.semanticSearch(queryEmbedding);
            } catch (e) {
                console.warn('Semantic search failed, using keyword fallback:', e);
                results = this.keywordSearch(query);
            }
        } else {
            results = this.keywordSearch(query);
        }

        return results.slice(0, CHATBOT_CONFIG.TOP_K_RESULTS);
    }

    // ─── Semantic Search ───
    semanticSearch(queryEmbedding) {
        const scored = this.knowledgeBase.chunks
            .filter(c => c.embedding !== null)
            .map(chunk => ({
                text: chunk.text,
                score: this.cosineSimilarity(queryEmbedding, chunk.embedding)
            }))
            .filter(r => r.score >= CHATBOT_CONFIG.SIMILARITY_THRESHOLD)
            .sort((a, b) => b.score - a.score);

        return scored;
    }

    // ─── Keyword Search (Fallback) ───
    keywordSearch(query) {
        const queryWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);

        const scored = this.knowledgeBase.chunks.map(chunk => {
            const chunkLower = chunk.text.toLowerCase();
            let score = 0;

            for (const word of queryWords) {
                if (chunkLower.includes(word)) {
                    score += 1;
                    // Bonus for exact word match
                    const regex = new RegExp(`\\b${word}\\b`, 'gi');
                    const matches = chunkLower.match(regex);
                    if (matches) score += matches.length * 0.5;
                }
            }

            // Normalize
            score = score / Math.max(queryWords.length, 1);

            return { text: chunk.text, score };
        })
            .filter(r => r.score > 0.2)
            .sort((a, b) => b.score - a.score);

        return scored;
    }

    // ─── Text Chunking ───
    chunkText(text, chunkSize, overlap) {
        if (!text || text.length <= chunkSize) return [text];

        const chunks = [];
        const sentences = text.split(/(?<=[.!?])\s+/);
        let currentChunk = '';

        for (const sentence of sentences) {
            if (currentChunk.length + sentence.length > chunkSize && currentChunk.length > 0) {
                chunks.push(currentChunk.trim());

                // Overlap: keep last portion
                const words = currentChunk.split(' ');
                const overlapWords = Math.ceil(overlap / 5);
                currentChunk = words.slice(-overlapWords).join(' ') + ' ' + sentence;
            } else {
                currentChunk += (currentChunk ? ' ' : '') + sentence;
            }
        }

        if (currentChunk.trim()) {
            chunks.push(currentChunk.trim());
        }

        return chunks;
    }

    // ─── Embeddings via Proxy ───
    async getEmbedding(text) {
        const response = await fetch(`${CHATBOT_CONFIG.API_PROXY_URL}/api/embed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content: { parts: [{ text: text.substring(0, 2048) }] }
            })
        });

        if (!response.ok) throw new Error(`Embed error: ${response.status}`);
        const data = await response.json();
        return data.embedding?.values || [];
    }

    // ─── Cosine Similarity ───
    cosineSimilarity(a, b) {
        if (!a || !b || a.length !== b.length) return 0;

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator === 0 ? 0 : dotProduct / denominator;
    }

    // ─── Get Stats ───
    getStats() {
        return {
            totalEntries: this.knowledgeBase.entries.length,
            totalChunks: this.knowledgeBase.chunks.length,
            embeddedChunks: this.knowledgeBase.chunks.filter(c => c.embedding !== null).length,
            storageUsed: new Blob([JSON.stringify(this.knowledgeBase)]).size
        };
    }

    // ─── Export / Import ───
    exportKB() {
        return JSON.stringify(this.knowledgeBase, null, 2);
    }

    importKB(jsonString) {
        try {
            const data = JSON.parse(jsonString);
            if (data.entries && data.chunks) {
                this.knowledgeBase = data;
                this.saveKnowledgeBase();
                return true;
            }
            throw new Error('Invalid format');
        } catch (e) {
            throw new Error('Invalid knowledge base file: ' + e.message);
        }
    }

    // ─── Clear All ───
    clearAll() {
        this.knowledgeBase = { entries: [], chunks: [] };
        this.saveKnowledgeBase();
    }
}


// ─────────────────────────────────────────────────────────────
// INITIALIZE CHATBOT
// ─────────────────────────────────────────────────────────────
const gpaChatbot = new ChatbotUI();
