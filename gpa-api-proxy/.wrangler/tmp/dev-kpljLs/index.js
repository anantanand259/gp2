var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// src/index.js
var ALLOWED_ORIGINS = [
  "https://anantanand259.github.io",
  "http://localhost:3000",
  "http://localhost:5500",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:5500",
  "http://localhost:8080"
];
var GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
var EMBED_MODEL = "text-embedding-004";
var RATE_LIMIT_WINDOW = 6e4;
var RATE_LIMIT_MAX = 60;
var rateLimitMap = /* @__PURE__ */ new Map();
var src_default = {
  async fetch(request, env, ctx) {
    if (request.method === "OPTIONS") {
      return handleCORS(request);
    }
    const origin = request.headers.get("Origin") || "";
    if (!isAllowedOrigin(origin)) {
      return jsonResponse({ error: "Forbidden: Origin not allowed" }, 403);
    }
    const clientIP = request.headers.get("CF-Connecting-IP") || "unknown";
    if (isRateLimited(clientIP)) {
      return corsResponse(
        jsonResponse({ error: "Rate limit exceeded. Please slow down." }, 429),
        origin
      );
    }
    const openrouterKey = env.OPENROUTER_API_KEY;
    if (!openrouterKey) {
      return corsResponse(
        jsonResponse({ error: "OpenRouter API key not set" }, 500),
        origin
      );
    }
    const url = new URL(request.url);
    const path = url.pathname;
    try {
      let response;
      if (path === "/api/chat" && request.method === "POST") {
        response = await handleChat(request, apiKey);
      } else if (path === "/api/embed" && request.method === "POST") {
        response = await handleEmbed(request, apiKey);
      } else if (path === "/api/health" && request.method === "GET") {
        response = jsonResponse({
          status: "ok",
          timestamp: (/* @__PURE__ */ new Date()).toISOString(),
          message: "GPA API Proxy is running"
        }, 200);
      } else {
        response = jsonResponse({ error: "Not found" }, 404);
      }
      return corsResponse(response, origin);
    } catch (err) {
      return corsResponse(
        jsonResponse({ error: "Internal server error", detail: err.message }, 500),
        origin
      );
    }
  }
};
async function handleChat(request, env) {
  const body = await request.json();
  const userMessage = body.contents?.[0]?.parts?.[0]?.text || "";
  const apiKey2 = env.OPENROUTER_API_KEY;
  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey2}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "openai/gpt-oss-20b:free",
        messages: [
          {
            role: "user",
            content: userMessage
          }
        ]
      })
    });
    const data = await response.json();
    if (!response.ok) {
      return jsonResponse({
        error: data?.error?.message || "Model failed",
        details: data
      }, 500);
    }
    const output = data?.choices?.[0]?.message?.content || "No response";
    return jsonResponse({
      candidates: [
        {
          content: {
            parts: [{ text: output }]
          }
        }
      ]
    });
  } catch (err) {
    return jsonResponse({ error: err.message }, 500);
  }
}
__name(handleChat, "handleChat");
async function handleEmbed(request, apiKey2) {
  const body = await request.json();
  if (!body.content || !body.content.parts) {
    return jsonResponse({ error: 'Invalid request: "content.parts" is required' }, 400);
  }
  const geminiUrl = `${GEMINI_BASE}/${EMBED_MODEL}:embedContent?key=${apiKey2}`;
  const geminiResponse = await fetch(geminiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: `models/${EMBED_MODEL}`,
      ...body
    })
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
__name(handleEmbed, "handleEmbed");
function isAllowedOrigin(origin) {
  return ALLOWED_ORIGINS.includes(origin);
}
__name(isAllowedOrigin, "isAllowedOrigin");
function handleCORS(request) {
  const origin = request.headers.get("Origin") || "";
  if (!isAllowedOrigin(origin)) {
    return new Response("Forbidden", { status: 403 });
  }
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Max-Age": "86400"
    }
  });
}
__name(handleCORS, "handleCORS");
function corsResponse(response, origin) {
  const newHeaders = new Headers(response.headers);
  if (origin && isAllowedOrigin(origin)) {
    newHeaders.set("Access-Control-Allow-Origin", origin);
    newHeaders.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    newHeaders.set("Access-Control-Allow-Headers", "Content-Type");
  }
  return new Response(response.body, {
    status: response.status,
    headers: newHeaders
  });
}
__name(corsResponse, "corsResponse");
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}
__name(jsonResponse, "jsonResponse");
function isRateLimited(ip) {
  const now = Date.now();
  const record = rateLimitMap.get(ip);
  if (!record || now - record.windowStart > RATE_LIMIT_WINDOW) {
    rateLimitMap.set(ip, { windowStart: now, count: 1 });
    return false;
  }
  record.count++;
  if (record.count > RATE_LIMIT_MAX) {
    return true;
  }
  return false;
}
__name(isRateLimited, "isRateLimited");

// C:/Users/DELL/AppData/Roaming/npm/node_modules/wrangler/templates/middleware/middleware-ensure-req-body-drained.ts
var drainBody = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } finally {
    try {
      if (request.body !== null && !request.bodyUsed) {
        const reader = request.body.getReader();
        while (!(await reader.read()).done) {
        }
      }
    } catch (e) {
      console.error("Failed to drain the unused request body.", e);
    }
  }
}, "drainBody");
var middleware_ensure_req_body_drained_default = drainBody;

// .wrangler/tmp/bundle-4NvPwz/middleware-insertion-facade.js
var __INTERNAL_WRANGLER_MIDDLEWARE__ = [
  middleware_ensure_req_body_drained_default
];
var middleware_insertion_facade_default = src_default;

// C:/Users/DELL/AppData/Roaming/npm/node_modules/wrangler/templates/middleware/common.ts
var __facade_middleware__ = [];
function __facade_register__(...args) {
  __facade_middleware__.push(...args.flat());
}
__name(__facade_register__, "__facade_register__");
function __facade_invokeChain__(request, env, ctx, dispatch, middlewareChain) {
  const [head, ...tail] = middlewareChain;
  const middlewareCtx = {
    dispatch,
    next(newRequest, newEnv) {
      return __facade_invokeChain__(newRequest, newEnv, ctx, dispatch, tail);
    }
  };
  return head(request, env, ctx, middlewareCtx);
}
__name(__facade_invokeChain__, "__facade_invokeChain__");
function __facade_invoke__(request, env, ctx, dispatch, finalMiddleware) {
  return __facade_invokeChain__(request, env, ctx, dispatch, [
    ...__facade_middleware__,
    finalMiddleware
  ]);
}
__name(__facade_invoke__, "__facade_invoke__");

// .wrangler/tmp/bundle-4NvPwz/middleware-loader.entry.ts
var __Facade_ScheduledController__ = class ___Facade_ScheduledController__ {
  constructor(scheduledTime, cron, noRetry) {
    this.scheduledTime = scheduledTime;
    this.cron = cron;
    this.#noRetry = noRetry;
  }
  static {
    __name(this, "__Facade_ScheduledController__");
  }
  #noRetry;
  noRetry() {
    if (!(this instanceof ___Facade_ScheduledController__)) {
      throw new TypeError("Illegal invocation");
    }
    this.#noRetry();
  }
};
function wrapExportedHandler(worker) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return worker;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  const fetchDispatcher = /* @__PURE__ */ __name(function(request, env, ctx) {
    if (worker.fetch === void 0) {
      throw new Error("Handler does not export a fetch() function.");
    }
    return worker.fetch(request, env, ctx);
  }, "fetchDispatcher");
  return {
    ...worker,
    fetch(request, env, ctx) {
      const dispatcher = /* @__PURE__ */ __name(function(type, init) {
        if (type === "scheduled" && worker.scheduled !== void 0) {
          const controller = new __Facade_ScheduledController__(
            Date.now(),
            init.cron ?? "",
            () => {
            }
          );
          return worker.scheduled(controller, env, ctx);
        }
      }, "dispatcher");
      return __facade_invoke__(request, env, ctx, dispatcher, fetchDispatcher);
    }
  };
}
__name(wrapExportedHandler, "wrapExportedHandler");
function wrapWorkerEntrypoint(klass) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return klass;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  return class extends klass {
    #fetchDispatcher = /* @__PURE__ */ __name((request, env, ctx) => {
      this.env = env;
      this.ctx = ctx;
      if (super.fetch === void 0) {
        throw new Error("Entrypoint class does not define a fetch() function.");
      }
      return super.fetch(request);
    }, "#fetchDispatcher");
    #dispatcher = /* @__PURE__ */ __name((type, init) => {
      if (type === "scheduled" && super.scheduled !== void 0) {
        const controller = new __Facade_ScheduledController__(
          Date.now(),
          init.cron ?? "",
          () => {
          }
        );
        return super.scheduled(controller);
      }
    }, "#dispatcher");
    fetch(request) {
      return __facade_invoke__(
        request,
        this.env,
        this.ctx,
        this.#dispatcher,
        this.#fetchDispatcher
      );
    }
  };
}
__name(wrapWorkerEntrypoint, "wrapWorkerEntrypoint");
var WRAPPED_ENTRY;
if (typeof middleware_insertion_facade_default === "object") {
  WRAPPED_ENTRY = wrapExportedHandler(middleware_insertion_facade_default);
} else if (typeof middleware_insertion_facade_default === "function") {
  WRAPPED_ENTRY = wrapWorkerEntrypoint(middleware_insertion_facade_default);
}
var middleware_loader_entry_default = WRAPPED_ENTRY;
export {
  __INTERNAL_WRANGLER_MIDDLEWARE__,
  middleware_loader_entry_default as default
};
//# sourceMappingURL=index.js.map
