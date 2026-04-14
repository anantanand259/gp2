/* ═══════════════════════════════════════════════════════════════
   GPA API Proxy — Cloudflare Worker
   Securely proxies LLM API calls, hiding API keys server-side.

   Routes:
     POST /api/chat    → OpenRouter (Qwen 2.5 14B) chat completion
     POST /api/gemini  → Gemini generateContent (fallback chat)
     POST /api/embed   → Gemini embedContent (for RAG embeddings)
     GET  /api/health  → Health check

   Secrets (set via `wrangler secret put`):
     OPENROUTER_API_KEY  → For chat (Qwen model)
     GEMINI_API_KEY      → For embeddings
   ═══════════════════════════════════════════════════════════════ */

// ─── Configuration ───
const ALLOWED_ORIGINS = [
    'https://anantanand259.github.io',
    'http://localhost:3000',
    'http://localhost:5500',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5500',
    'http://localhost:8080',
];

const OPENROUTER_CHAT_URL = 'https://openrouter.ai/api/v1/chat/completions';
const OPENROUTER_MODEL = 'openai/gpt-oss-120b:free';

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';
const EMBED_MODEL = 'text-embedding-004';

const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 1500; // ms

// Rate limit: 60 requests per minute per IP
const RATE_LIMIT_WINDOW = 60_000; // 1 minute in ms
const RATE_LIMIT_MAX = 60;

// In-memory rate limit store (resets on Worker restart, which is fine)
const rateLimitMap = new Map();

// ─── Main Handler ───
export default {
    async fetch(request, env, ctx) {
        // Handle CORS preflight
        if (request.method === 'OPTIONS') {
            return handleCORS(request);
        }

        const origin = request.headers.get('Origin') || '';
        const url = new URL(request.url);
        const path = url.pathname;

        // Allow health check and root without origin validation
        if (request.method === 'GET' && (path === '/api/health' || path === '/')) {
            const healthData = {
                status: 'ok',
                timestamp: new Date().toISOString(),
                message: 'GPA API Proxy is running ✅',
                models: {
                    chat: OPENROUTER_MODEL,
                    embed: EMBED_MODEL,
                }
            };
            const response = jsonResponse(healthData, 200);
            // Add CORS headers if origin is present and allowed
            if (origin && isAllowedOrigin(origin)) {
                return corsResponse(response, origin);
            }
            return response;
        }

        // For all other routes, validate origin
        if (!isAllowedOrigin(origin)) {
            return jsonResponse({ error: 'Forbidden: Origin not allowed' }, 403);
        }

        // Rate limiting
        const clientIP = request.headers.get('CF-Connecting-IP') || 'unknown';
        if (isRateLimited(clientIP)) {
            return corsResponse(
                jsonResponse({ error: 'Rate limit exceeded. Please slow down.' }, 429),
                origin
            );
        }

        // Route handling
        try {
            let response;

            if (path === '/api/chat' && request.method === 'POST') {
                response = await handleChat(request, env);
            } else if (path === '/api/gemini' && request.method === 'POST') {
                response = await handleGemini(request, env);
            } else if (path === '/api/embed' && request.method === 'POST') {
                response = await handleEmbed(request, env);
            } else {
                response = jsonResponse({ error: 'Not found' }, 404);
            }

            return corsResponse(response, origin);
        } catch (err) {
            return corsResponse(
                jsonResponse({ error: 'Internal server error', detail: err.message }, 500),
                origin
            );
        }
    }
};


// ═══════════════════════════════════════════════════════════════
// ROUTE HANDLERS
// ═══════════════════════════════════════════════════════════════

// ─── /api/chat → OpenRouter (Qwen 2.5 14B) ───
async function handleChat(request, env) {
    const apiKey = env.OPENROUTER_API_KEY;
    if (!apiKey) {
        return jsonResponse({ error: 'Server misconfigured: OPENROUTER_API_KEY not set' }, 500);
    }

    const body = await request.json();

    // ── Accept two formats from the client ──
    //
    // FORMAT A (OpenAI-style — preferred, sent by new chatbot.js):
    //   { messages: [...], model: "...", temperature: 0.7, ... }
    //
    // FORMAT B (Legacy Gemini-style — for backward compatibility):
    //   { contents: [{ role, parts: [{ text }] }], systemInstruction: { parts: [{ text }] } }

    let messages;

    if (body.messages && Array.isArray(body.messages)) {
        // FORMAT A: already OpenAI-compatible, pass through
        messages = body.messages;
    } else if (body.contents && Array.isArray(body.contents)) {
        // FORMAT B: convert Gemini format → OpenAI format
        messages = [];

        // Add system instruction if present
        if (body.systemInstruction?.parts?.[0]?.text) {
            messages.push({
                role: 'system',
                content: body.systemInstruction.parts[0].text
            });
        }

        // Convert contents
        for (const entry of body.contents) {
            const role = entry.role === 'model' ? 'assistant' : entry.role;
            const content = entry.parts?.map(p => p.text).join('\n') || '';
            messages.push({ role, content });
        }
    } else {
        return jsonResponse({ error: 'Invalid request: "messages" or "contents" array required' }, 400);
    }

    // ── Call OpenRouter with retry ──
    let lastError = null;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(OPENROUTER_CHAT_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`,
                    'HTTP-Referer': 'https://gpa.ac.in',
                    'X-Title': 'GPA Assistant Chatbot',
                },
                body: JSON.stringify({
                    model: body.model || OPENROUTER_MODEL,
                    messages: messages,
                    temperature: body.temperature ?? 0.7,
                    top_p: body.top_p ?? 0.9,
                    max_tokens: body.max_tokens ?? 1024,
                })
            });

            if (response.status === 429) {
                if (attempt < MAX_RETRIES - 1) {
                    const delay = RETRY_BASE_DELAY * Math.pow(2, attempt);
                    await sleep(delay);
                    continue;
                }
                return jsonResponse({ error: 'Rate limit exceeded on AI provider. Try again shortly.' }, 429);
            }

            const data = await response.json();

            if (!response.ok) {
                return jsonResponse({
                    error: data?.error?.message || `OpenRouter error: ${response.status}`,
                    details: data
                }, response.status);
            }

            const output = data?.choices?.[0]?.message?.content;
            if (!output) {
                return jsonResponse({ error: 'Empty response from AI model' }, 502);
            }

            // Return in Gemini-compatible format so the client works universally
            return jsonResponse({
                candidates: [{
                    content: {
                        parts: [{ text: output }]
                    }
                }],
                // Also include the raw OpenAI-style response for flexibility
                _openai: {
                    model: data.model,
                    usage: data.usage,
                }
            });

        } catch (err) {
            lastError = err;
            if (attempt < MAX_RETRIES - 1) {
                await sleep(RETRY_BASE_DELAY * Math.pow(2, attempt));
                continue;
            }
        }
    }

    return jsonResponse({ error: lastError?.message || 'Chat request failed after retries' }, 500);
}


// ─── /api/gemini → Gemini generateContent (Fallback Chat) ───
async function handleGemini(request, env) {
    const apiKey = env.GEMINI_API_KEY;
    if (!apiKey) {
        return jsonResponse({ error: 'Server misconfigured: GEMINI_API_KEY not set' }, 500);
    }

    const body = await request.json();
    const model = body.model || 'gemini-2.0-flash';

    // Remove 'model' from the body before forwarding
    const { model: _, ...geminiBody } = body;

    const geminiUrl = `${GEMINI_BASE}/${model}:generateContent?key=${apiKey}`;

    let lastError = null;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(geminiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(geminiBody)
            });

            if (response.status === 429) {
                if (attempt < MAX_RETRIES - 1) {
                    const delay = RETRY_BASE_DELAY * Math.pow(2, attempt);
                    await sleep(delay);
                    continue;
                }
                return jsonResponse({ error: `429 quota exceeded for ${model}` }, 429);
            }

            const data = await response.json();

            if (!response.ok) {
                return jsonResponse({
                    error: data?.error?.message || `Gemini error: ${response.status}`
                }, response.status);
            }

            return jsonResponse(data, 200);

        } catch (err) {
            lastError = err;
            if (attempt < MAX_RETRIES - 1) {
                await sleep(RETRY_BASE_DELAY * Math.pow(2, attempt));
                continue;
            }
        }
    }

    return jsonResponse({ error: lastError?.message || 'Gemini request failed after retries' }, 500);
}

// ─── /api/embed → Gemini Embedding API ───
async function handleEmbed(request, env) {
    const apiKey = env.GEMINI_API_KEY;
    if (!apiKey) {
        return jsonResponse({ error: 'Server misconfigured: GEMINI_API_KEY not set' }, 500);
    }

    const body = await request.json();

    // Validate structure
    if (!body.content || !body.content.parts) {
        return jsonResponse({ error: 'Invalid request: "content.parts" is required' }, 400);
    }

    // Forward to Gemini embedding API
    const geminiUrl = `${GEMINI_BASE}/${EMBED_MODEL}:embedContent?key=${apiKey}`;

    const geminiResponse = await fetch(geminiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: `models/${EMBED_MODEL}`,
            ...body
        }),
    });

    const data = await geminiResponse.json();

    if (!geminiResponse.ok) {
        return jsonResponse(
            { error: data?.error?.message || `Embedding API error: ${geminiResponse.status}` },
            geminiResponse.status
        );
    }

    return jsonResponse(data, 200);
}


// ═══════════════════════════════════════════════════════════════
// CORS + SECURITY HELPERS
// ═══════════════════════════════════════════════════════════════

function isAllowedOrigin(origin) {
    return ALLOWED_ORIGINS.includes(origin);
}
function handleCORS(request) {
    const origin = request.headers.get('Origin') || '';
    if (!isAllowedOrigin(origin)) {
        return new Response('Forbidden', { status: 403 });
    }

    return new Response(null, {
        status: 204,
        headers: {
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400',
        }
    });
}

function corsResponse(response, origin) {
    const newHeaders = new Headers(response.headers);
    if (origin && isAllowedOrigin(origin)) {
        newHeaders.set('Access-Control-Allow-Origin', origin);
        newHeaders.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        newHeaders.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    }
    return new Response(response.body, {
        status: response.status,
        headers: newHeaders
    });
}

function jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
        status,
        headers: { 'Content-Type': 'application/json' }
    });
}


// ═══════════════════════════════════════════════════════════════
// RATE LIMITING (simple in-memory, per-IP)
// ═══════════════════════════════════════════════════════════════

function isRateLimited(ip) {
    const now = Date.now();
    const record = rateLimitMap.get(ip);

    if (!record || now - record.windowStart > RATE_LIMIT_WINDOW) {
        // New window
        rateLimitMap.set(ip, { windowStart: now, count: 1 });
        return false;
    }

    record.count++;
    if (record.count > RATE_LIMIT_MAX) {
        return true;
    }

    return false;
}


// ═══════════════════════════════════════════════════════════════
// UTILITY
// ═══════════════════════════════════════════════════════════════

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
