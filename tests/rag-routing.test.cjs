const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function chatbot(fetchImpl) {
    const source = fs.readFileSync('chatbot.js', 'utf8').replace('const gpaChatbot = new ChatbotUI();', 'globalThis.TestChatbot = ChatbotUI;');
    const context = vm.createContext({ localStorage: { getItem: () => null }, console, setTimeout, clearTimeout, AbortController, fetch: fetchImpl });
    vm.runInContext(source, context);
    const bot = Object.create(context.TestChatbot.prototype);
    bot.callLLM = () => { throw new Error('General LLM must not be called'); };
    return bot;
}

test('failed health check still queries KB and preserves a refusal', async () => {
    const bot = chatbot();
    bot.ragAvailable = false;
    bot._callRAGBackend = async () => ({ answer: 'This specific information is not available', source_type: 'rag', chunk_count: 1 });
    assert.equal((await bot.processQuery('Question')).source, 'rag');
});

test('hi and hlw reply immediately without a backend request', async () => {
    const bot = chatbot();
    bot._callRAGBackend = () => { throw new Error('Greeting must not query the backend'); };
    for (const text of ['hi', 'HI!', 'hlw', 'hello', 'hlo', 'नमस्ते 🙏', 'thank you']) {
        const result = await bot.processQuery(text);
        assert.equal(result.source, 'conversation');
        assert.ok(result.answer.length > 0);
    }
});

test('a greeting followed by a factual question still queries the KB', async () => {
    const bot = chatbot();
    let queried = false;
    bot._callRAGBackend = async () => { queried = true; return { answer:'From the notice', source_type:'rag' }; };
    await bot.processQuery('Hi, when is my exam?');
    assert.equal(queried, true);
});

test('backend errors never switch to general knowledge', async () => {
    const bot = chatbot();
    bot._callRAGBackend = async () => { throw new Error('Timeout'); };
    await assert.rejects(bot.processQuery('Deadline?'), /could not check/);
});

test('old backend API key error is shown as a configuration failure', async () => {
    const bot = chatbot(async () => Response.json({ error: '400 INVALID_ARGUMENT: API key not valid. API_KEY_INVALID' }, { status: 500 }));
    await assert.rejects(bot.processQuery('Deadline?'), /API key is missing or invalid/);
    assert.equal(bot.ragAvailable, true);
});

test('structured provider error is preserved', async () => {
    const bot = chatbot(async () => Response.json({ code: 'PROVIDER_QUOTA_ERROR', error: 'Provider quota exceeded; please retry later.' }, { status: 503 }));
    await assert.rejects(bot.processQuery('Deadline?'), /Provider quota exceeded/);
});

test('web and no-answer decisions are preserved', async () => {
    const bot = chatbot();
    for (const [source_type, expected] of [['internet', 'internet'], ['none', 'none']]) {
        bot._callRAGBackend = async () => ({ answer: 'Backend decision', source_type, chunk_count: 0 });
        assert.equal((await bot.processQuery('Question')).source, expected);
    }
});

test('proxy health verifies Python backend and reports missing configuration', async () => {
    const source = fs.readFileSync('src/index.js', 'utf8');
    const worker = (await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'))).default;
    const request = new Request('https://proxy.test/api/health', { headers: { Origin: 'https://anantanand259.github.io' } });
    const missing = await worker.fetch(request, {}, {});
    assert.equal(missing.status, 500);
    const originalFetch = global.fetch;
    try {
        global.fetch = async upstream => {
            assert.equal(upstream.url, 'https://backend.test/api/health');
            return Response.json({ retriever_ready: true, total_chunks: 2 });
        };
        const result = await worker.fetch(request, { RAG_BACKEND_URL: 'https://backend.test' }, {});
        assert.equal((await result.json()).retriever_ready, true);
    } finally {
        global.fetch = originalFetch;
    }
});

test('admin inline scripts compile', () => {
    const html = fs.readFileSync('admin.html', 'utf8');
    for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) new vm.Script(match[1]);
});

test('proxy canonicalizes a trailing DNS dot in the Render URL', async () => {
    const source = fs.readFileSync('src/index.js', 'utf8');
    const worker = (await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'))).default;
    const originalFetch = global.fetch;
    try {
        global.fetch = async upstream => {
            assert.equal(upstream.url, 'https://gpa-rag-backend-o9v7.onrender.com/api/health');
            return Response.json({ retriever_ready: true });
        };
        const result = await worker.fetch(new Request('https://proxy.test/api/health'),
            { RAG_BACKEND_URL: ' https://gpa-rag-backend-o9v7.onrender.com./ ' }, {});
        assert.equal(result.status, 200);
        assert.equal((await result.json()).retriever_ready, true);
    } finally { global.fetch = originalFetch; }
});

test('Render temporary non-JSON failure gives a retryable hosting message', async () => {
    const source = fs.readFileSync('src/index.js', 'utf8');
    const worker = (await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'))).default;
    const originalFetch = global.fetch;
    try {
        global.fetch = async () => new Response('', { status: 503 });
        const result = await worker.fetch(new Request('https://proxy.test/api/health'),
            { RAG_BACKEND_URL: 'https://gpa-rag-backend-o9v7.onrender.com' }, {});
        assert.equal(result.status, 503);
        const body = await result.json();
        assert.equal(body.code, 'BACKEND_TEMPORARILY_UNAVAILABLE');
        assert.equal(body.retry_after_seconds, 5);
        assert.match(body.error, /Render/);
    } finally { global.fetch = originalFetch; }
});
