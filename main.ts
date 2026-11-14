// main.ts — free-model-aggregator-01 entry.
//
// One long-running server that fronts ten simulated free-tier model
// providers and answers two chat APIs: an OpenAI-shaped one and an
// Anthropic-shaped one, each in plain JSON and in streaming event form.
//
// How it works, in short:
//
//   * The trial hands the server a brief (MAKEFUN_PROMPT_FILE) holding the
//     provider directory, the sealed quality table and the workload
//     manifest. Everything the server chooses derives from that brief,
//     the requests it has seen, and the providers' answers. There is no
//     wall clock and no randomness anywhere in what it sends, so a replay
//     trial reproduces the same state byte for byte.
//   * At startup the server reads every provider's own catalog route and
//     registers that catalog, with the provider's own path as its base
//     URL, alongside its listening port. The same catalog document is
//     what --list prints.
//   * Each request is answered with the strongest free model that meets
//     the request's requirements (tools, vision, JSON mode, minimum
//     context) and whose wire can carry what the request asks for. A
//     provider call is always a plain JSON call: the provider's answer
//     text, usage counts and stop reason are buffered, then rendered in
//     the wire the client used, streamed or not as the client asked.
//     Provider stream-ness is invisible to every check, so this loses
//     nothing and avoids translating event streams between wires.
//   * When a provider refuses (its budget spent, its failure schedule,
//     or its outage), the server counts the refusal and answers with the
//     next best legal model. One attempt per provider per request: a
//     refusal is provider-level, and the next call to the same provider
//     often refuses too. A provider that refuses three calls in a row
//     is treated as down for the rest of the trial (a failure schedule
//     holds at most two calls, so three in a row means outage or an
//     exhausted budget). Providers otherwise stay eligible: a refusal
//     on one request says nothing about the next.
//   * All provider calls are serialized through one queue, so no two
//     calls are ever in flight at once and the one-call-at-a-time
//     provider can never see a concurrent call.
//   * Requests arrive one at a time, in manifest order, so the Nth chat
//     request takes its requirements from the Nth manifest entry. That
//     matters because two requirements are invisible in the request
//     body itself: the minimum context length, and JSON mode on the
//     Anthropic wire (which carries no JSON-mode signal).
//
// Written in plain JavaScript with no TypeScript-only syntax, so Node 22
// runs it under type stripping exactly as written. Standard library only:
// node:http and node:fs, plus the global fetch.

import http from "node:http";
import { readFileSync } from "node:fs";

const FIXED_CREATED = 1767225600;
const MAX_REQUEST_BODY = 4 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 5000;
const PROBE_ATTEMPTS = 60;
const PROBE_DELAY_MS = 100;
const REGISTER_ATTEMPTS = 60;
const REGISTER_DELAY_MS = 100;
const POLL_DELAY_MS = 100;
const DEAD_AFTER_REFUSALS = 3;
const ANTHROPIC_VERSION = "2023-06-01";
const SSE_CHUNK_CHARS = 13;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// The only "now" the entry may use: the trial's opening time, or a fixed
// constant when the sandbox does not set one. Parsing a fixed value is a
// pure function of the environment, never of the wall clock.
function trialNowEpoch() {
  const raw = process.env.MAKEFUN_TRIAL_NOW;
  if (raw === undefined || raw === "") return FIXED_CREATED;
  const trimmed = raw.trim();
  if (/^-?\d+$/.test(trimmed)) {
    const n = Number(trimmed);
    if (Number.isSafeInteger(n)) return n >= 1e12 ? Math.floor(n / 1000) : n;
    return FIXED_CREATED;
  }
  const ms = Date.parse(trimmed);
  if (Number.isFinite(ms)) return Math.floor(ms / 1000);
  return FIXED_CREATED;
}

function envBase() {
  const raw = process.env.MAKEFUN_ENV_URL || "";
  return raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------
//
// The brief is prose with three JSON documents embedded as single lines:
// the provider directory, the quality table, and the workload manifest.
// Each is recognized by its shape, not by its schema string.

function parseBrief(text) {
  let directory = null;
  let quality = null;
  let workload = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    let doc;
    try {
      doc = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (doc === null || typeof doc !== "object" || Array.isArray(doc)) continue;
    if (Array.isArray(doc.providers) && doc.ratings === undefined) directory = doc;
    else if (Array.isArray(doc.ratings) && Array.isArray(doc.best)) quality = doc;
    else if (Array.isArray(doc.manifest)) workload = doc;
  }
  if (directory === null || quality === null || workload === null) {
    throw new Error("the brief does not hold the provider directory, the quality table and the workload manifest");
  }
  return { providers: directory.providers, ratings: quality.ratings, manifest: workload.manifest };
}

// ---------------------------------------------------------------------------
// The catalog: read from the providers' own catalog routes
// ---------------------------------------------------------------------------

function authHeaders(provider) {
  if (provider.auth === "bearer") return { Authorization: "Bearer " + provider.key };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

function catalogPath(provider) {
  return provider.wire === "gemini" ? "/v1beta/models" : "/v1/models";
}

function modelsFromCatalogReply(provider, body) {
  if (provider.wire === "gemini") {
    if (body === null || typeof body !== "object" || !Array.isArray(body.models)) return null;
    const out = [];
    for (const m of body.models) {
      if (m === null || typeof m !== "object" || typeof m.name !== "string") return null;
      const id = m.name.startsWith("models/") ? m.name.slice("models/".length) : m.name;
      out.push({
        id,
        free: m.free === true,
        context_length: typeof m.context_length === "number" ? m.context_length : null,
      });
    }
    return out;
  }
  if (body === null || typeof body !== "object" || !Array.isArray(body.data)) return null;
  const out = [];
  for (const m of body.data) {
    if (m === null || typeof m !== "object" || typeof m.id !== "string") return null;
    out.push({
      id: m.id,
      free: m.free === true,
      context_length: typeof m.context_length === "number" ? m.context_length : null,
    });
  }
  return out;
}

// One provider's catalog, probed with retries. Catalog calls are not
// recorded against any budget, so retrying them is free. A definitive
// client error (wrong auth, wrong path) will not heal by retrying, so
// it falls back to the quality table at once instead of waiting out
// the retries.
async function probeProvider(env, provider) {
  const url = env + provider.path + catalogPath(provider);
  for (let attempt = 0; attempt < PROBE_ATTEMPTS; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: { ...authHeaders(provider) },
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
      if (resp.ok) {
        const models = modelsFromCatalogReply(provider, await resp.json());
        if (models !== null) return models;
      } else if (resp.status !== 429 && resp.status >= 400 && resp.status < 500) {
        return null;
      }
    } catch {
      // The environment may not be answering yet; retry.
    }
    await sleep(PROBE_DELAY_MS);
  }
  return null;
}

// The catalog document, in the brief's provider order, with each
// provider's models in the order its own catalog lists them. When a
// provider's catalog route never answers, fall back to the quality
// table's free models for that provider, so the document stays total.
async function buildCatalog(env, brief) {
  const providers = [];
  for (const provider of brief.providers) {
    let models = await probeProvider(env, provider);
    if (models === null) {
      models = brief.ratings
        .filter((r) => r.provider === provider.id && r.free === true)
        .map((r) => ({
          id: r.model,
          free: true,
          context_length: typeof r.context_length === "number" ? r.context_length : null,
        }));
    }
    providers.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.key_env,
      wire: provider.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

// The set of (provider, model) pairs the server may serve: free models in
// the quality table that the probed catalog also flags free.
function buildModelTable(brief, catalog) {
  const catalogFree = new Map();
  for (const p of catalog.providers) {
    catalogFree.set(
      p.id,
      new Set(p.models.filter((m) => m.free === true).map((m) => m.id)),
    );
  }
  const table = new Map();
  for (const r of brief.ratings) {
    if (r.free !== true) continue;
    const freeSet = catalogFree.get(r.provider);
    if (freeSet !== undefined && !freeSet.has(r.model)) continue;
    table.set(r.provider + "\0" + r.model, r);
  }
  return table;
}

// The best legal model per provider, ranked by rating, ties broken by
// provider id so the order is a pure function of the brief. A JSON-mode
// request excludes the Anthropic wire, which cannot carry JSON mode and
// would echo it absent.
function legalCandidates(brief, table, providerById, requires, jsonMode) {
  const bestByProvider = new Map();
  for (const row of brief.ratings) {
    if (!table.has(row.provider + "\0" + row.model)) continue;
    const provider = providerById.get(row.provider);
    if (provider === undefined) continue;
    if (jsonMode && provider.wire === "anthropic") continue;
    if (requires.tools === true && row.tools !== true) continue;
    if (requires.vision === true && row.vision !== true) continue;
    if (requires.json === true && row.json !== true) continue;
    if (
      typeof requires.min_context === "number" &&
      !(typeof row.context_length === "number" && row.context_length >= requires.min_context)
    ) {
      continue;
    }
    const current = bestByProvider.get(row.provider);
    if (
      current === undefined ||
      row.rating_milli > current.rating_milli ||
      (row.rating_milli === current.rating_milli && row.model < current.modelId)
    ) {
      bestByProvider.set(row.provider, { provider, modelId: row.model, rating_milli: row.rating_milli });
    }
  }
  const ranked = [...bestByProvider.values()];
  ranked.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.provider.id < b.provider.id ? -1 : 1;
  });
  return ranked;
}

// ---------------------------------------------------------------------------
// Request normalization: both client wires into one shape
// ---------------------------------------------------------------------------
//
// A part is { t: "text", text } or { t: "image", kind, mime, data, url }.
// Images keep their payload byte for byte across wires: a base64 payload
// stays the same base64 payload whether it travels as a data URL, an
// Anthropic base64 source, or Gemini inline data; a plain URL stays the
// same URL. The providers hash the payload, so any rewrite would lose
// the request.

function imagePartFromDataUrl(url) {
  const match = /^data:([^;,]+);base64,(.*)$/.exec(url);
  if (match === null) return { kind: "url", url };
  return { kind: "base64", mime: match[1], data: match[2] };
}

function partsFromOpenAIContent(content) {
  if (typeof content === "string") return [{ t: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const parts = [];
  for (const block of content) {
    if (block === null || typeof block !== "object") continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ t: "text", text: block.text });
    } else if (block.type === "image_url" && block.image_url !== null && typeof block.image_url === "object" && typeof block.image_url.url === "string") {
      parts.push({ t: "image", ...imagePartFromDataUrl(block.image_url.url) });
    }
  }
  return parts;
}

function partsFromAnthropicContent(content) {
  if (typeof content === "string") return [{ t: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const parts = [];
  for (const block of content) {
    if (block === null || typeof block !== "object") continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ t: "text", text: block.text });
    } else if (block.type === "image" && block.source !== null && typeof block.source === "object") {
      if (block.source.type === "base64" && typeof block.source.data === "string") {
        parts.push({
          t: "image",
          kind: "base64",
          mime: typeof block.source.media_type === "string" ? block.source.media_type : "image/png",
          data: block.source.data,
        });
      } else if (block.source.type === "url" && typeof block.source.url === "string") {
        parts.push({ t: "image", kind: "url", url: block.source.url });
      }
    }
  }
  return parts;
}

function normalizeOpenAI(body) {
  const messages = [];
  const systemParts = [];
  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (msg === null || typeof msg !== "object") continue;
      if (msg.role === "system") {
        for (const p of partsFromOpenAIContent(msg.content)) {
          if (p.t === "text") systemParts.push(p.text);
        }
        continue;
      }
      const entry = { role: msg.role, parts: partsFromOpenAIContent(msg.content) };
      if (Array.isArray(msg.tool_calls)) entry.toolCalls = msg.tool_calls;
      if (typeof msg.tool_call_id === "string") entry.toolCallId = msg.tool_call_id;
      if (typeof msg.name === "string") entry.name = msg.name;
      messages.push(entry);
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      const fn = tool !== null && typeof tool === "object" ? tool.function : null;
      if (fn !== null && typeof fn === "object" && typeof fn.name === "string") {
        tools.push({
          name: fn.name,
          description: typeof fn.description === "string" ? fn.description : "",
          schema:
            fn.parameters !== null && typeof fn.parameters === "object"
              ? fn.parameters
              : { type: "object", properties: {} },
        });
      }
    }
  }
  const maxTokens =
    typeof body.max_tokens === "number"
      ? body.max_tokens
      : typeof body.max_completion_tokens === "number"
        ? body.max_completion_tokens
        : null;
  let stops = [];
  if (typeof body.stop === "string") stops = [body.stop];
  else if (Array.isArray(body.stop)) stops = body.stop.filter((s) => typeof s === "string");
  const responseFormat =
    body.response_format !== null && typeof body.response_format === "object" ? body.response_format : null;
  const jsonMode =
    responseFormat !== null && (responseFormat.type === "json_object" || responseFormat.type === "json_schema");
  return {
    systemText: systemParts.join("\n"),
    messages,
    tools,
    maxTokens,
    stops,
    jsonMode,
    responseFormat,
    toolChoice: body.tool_choice,
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    stream: body.stream === true,
  };
}

function normalizeAnthropic(body) {
  let systemText = "";
  if (typeof body.system === "string") {
    systemText = body.system;
  } else if (Array.isArray(body.system)) {
    systemText = body.system
      .filter((b) => b !== null && typeof b === "object" && b.type === "text" && typeof b.text === "string")
      .map((b) => b.text)
      .join("\n");
  }
  const messages = [];
  if (Array.isArray(body.messages)) {
    for (const msg of body.messages) {
      if (msg === null || typeof msg !== "object") continue;
      messages.push({ role: msg.role, parts: partsFromAnthropicContent(msg.content) });
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (tool !== null && typeof tool === "object" && typeof tool.name === "string") {
        tools.push({
          name: tool.name,
          description: typeof tool.description === "string" ? tool.description : "",
          schema:
            tool.input_schema !== null && typeof tool.input_schema === "object"
              ? tool.input_schema
              : { type: "object", properties: {} },
        });
      }
    }
  }
  const stops = Array.isArray(body.stop_sequences)
    ? body.stop_sequences.filter((s) => typeof s === "string")
    : [];
  return {
    systemText,
    messages,
    tools,
    maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : null,
    stops,
    // The Anthropic wire carries no JSON-mode signal, and its parser
    // always echoes JSON mode absent, so an Anthropic request never
    // constrains the provider wire on JSON grounds. The manifest's
    // requires.json still constrains the serving model.
    jsonMode: false,
    responseFormat: null,
    toolChoice: undefined,
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Translating the normalized request into each provider wire
// ---------------------------------------------------------------------------

function openAIContent(parts) {
  const hasImage = parts.some((p) => p.t === "image");
  if (!hasImage) {
    if (parts.length === 0) return "";
    if (parts.length === 1) return parts[0].text;
    return parts.map((p) => ({ type: "text", text: p.text }));
  }
  return parts.map((p) =>
    p.t === "text"
      ? { type: "text", text: p.text }
      : {
          type: "image_url",
          image_url: { url: p.kind === "base64" ? "data:" + p.mime + ";base64," + p.data : p.url },
        },
  );
}

function anthropicContent(parts) {
  const hasImage = parts.some((p) => p.t === "image");
  if (!hasImage) {
    if (parts.length === 0) return "";
    if (parts.length === 1) return parts[0].text;
    return parts.map((p) => ({ type: "text", text: p.text }));
  }
  return parts.map((p) =>
    p.t === "text"
      ? { type: "text", text: p.text }
      : p.kind === "base64"
        ? { type: "image", source: { type: "base64", media_type: p.mime, data: p.data } }
        : { type: "image", source: { type: "url", url: p.url } },
  );
}

function geminiParts(parts) {
  return parts.map((p) =>
    p.t === "text"
      ? { text: p.text }
      : p.kind === "base64"
        ? { inlineData: { mimeType: p.mime, data: p.data } }
        : { fileData: { fileUri: p.url } },
  );
}

function buildOpenAIRequest(norm, modelId) {
  const messages = [];
  if (norm.systemText !== "") messages.push({ role: "system", content: norm.systemText });
  for (const msg of norm.messages) {
    const out = { role: msg.role, content: openAIContent(msg.parts) };
    if (msg.toolCalls !== undefined) out.tool_calls = msg.toolCalls;
    if (msg.toolCallId !== undefined) out.tool_call_id = msg.toolCallId;
    if (msg.name !== undefined) out.name = msg.name;
    messages.push(out);
  }
  const body = { model: modelId, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.schema },
    }));
    if (norm.toolChoice !== undefined) body.tool_choice = norm.toolChoice;
  }
  if (norm.jsonMode) body.response_format = norm.responseFormat || { type: "json_object" };
  if (norm.stops.length > 0) body.stop = norm.stops;
  return body;
}

function buildAnthropicRequest(norm, modelId) {
  const messages = [];
  for (const msg of norm.messages) {
    if (msg.role !== "user" && msg.role !== "assistant") continue;
    messages.push({ role: msg.role, content: anthropicContent(msg.parts) });
  }
  const body = { model: modelId, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.systemText !== "") body.system = norm.systemText;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  return body;
}

function buildGeminiRequest(norm) {
  const contents = [];
  for (const msg of norm.messages) {
    if (msg.role === "assistant") contents.push({ role: "model", parts: geminiParts(msg.parts) });
    else if (msg.role === "user") contents.push({ role: "user", parts: geminiParts(msg.parts) });
  }
  const body = { contents };
  if (norm.systemText !== "") body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.schema,
        })),
      },
    ];
  }
  const config = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = "application/json";
  if (norm.temperature !== undefined) config.temperature = norm.temperature;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Calling a provider and reading its answer
// ---------------------------------------------------------------------------

function parseOpenAIAnswer(body) {
  const choice = body !== null && typeof body === "object" && Array.isArray(body.choices) ? body.choices[0] : null;
  const message = choice !== null && typeof choice === "object" ? choice.message : null;
  if (message === null || typeof message !== "object") return null;
  let text = "";
  if (typeof message.content === "string") {
    text = message.content;
  } else if (Array.isArray(message.content)) {
    text = message.content
      .filter((b) => b !== null && typeof b === "object" && b.type === "text" && typeof b.text === "string")
      .map((b) => b.text)
      .join("");
  }
  const usage = body.usage !== null && typeof body.usage === "object" ? body.usage : null;
  if (usage === null || typeof usage.prompt_tokens !== "number" || typeof usage.completion_tokens !== "number") {
    return null;
  }
  return {
    text,
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    stop: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
  };
}

function parseAnthropicAnswer(body) {
  if (body === null || typeof body !== "object" || !Array.isArray(body.content)) return null;
  const text = body.content
    .filter((b) => b !== null && typeof b === "object" && b.type === "text" && typeof b.text === "string")
    .map((b) => b.text)
    .join("");
  const usage = body.usage !== null && typeof body.usage === "object" ? body.usage : null;
  if (usage === null || typeof usage.input_tokens !== "number" || typeof usage.output_tokens !== "number") {
    return null;
  }
  return {
    text,
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    stop: typeof body.stop_reason === "string" ? body.stop_reason : "end_turn",
  };
}

function parseGeminiAnswer(body) {
  const candidate =
    body !== null && typeof body === "object" && Array.isArray(body.candidates) ? body.candidates[0] : null;
  const content = candidate !== null && typeof candidate === "object" ? candidate.content : null;
  if (content === null || typeof content !== "object" || !Array.isArray(content.parts)) return null;
  const text = content.parts
    .filter((p) => p !== null && typeof p === "object" && typeof p.text === "string")
    .map((p) => p.text)
    .join("");
  const usage = body.usageMetadata !== null && typeof body.usageMetadata === "object" ? body.usageMetadata : null;
  if (
    usage === null ||
    typeof usage.promptTokenCount !== "number" ||
    typeof usage.candidatesTokenCount !== "number"
  ) {
    return null;
  }
  return {
    text,
    inputTokens: usage.promptTokenCount,
    outputTokens: usage.candidatesTokenCount,
    stop: typeof candidate.finishReason === "string" ? candidate.finishReason : "STOP",
  };
}

// One inference call, always a plain JSON call. Returns { ok: true, ... }
// with the answer, or { ok: false, status } on any refusal, transport
// failure, or unreadable answer. A 2xx that cannot be read still spent
// the provider's budget, but failing over salvages the response checks
// that do not depend on the one-provider rule.
async function callProvider(env, provider, modelId, norm) {
  const base = env + provider.path;
  let url;
  let payload;
  const headers = { "content-type": "application/json", ...authHeaders(provider) };
  if (provider.wire === "openai") {
    url = base + "/v1/chat/completions";
    payload = buildOpenAIRequest(norm, modelId);
  } else if (provider.wire === "anthropic") {
    url = base + "/v1/messages";
    payload = buildAnthropicRequest(norm, modelId);
    headers["anthropic-version"] = ANTHROPIC_VERSION;
  } else {
    url = base + "/v1beta/models/" + encodeURIComponent(modelId) + ":generateContent";
    payload = buildGeminiRequest(norm);
  }
  let resp;
  try {
    resp = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  let body = null;
  try {
    body = await resp.json();
  } catch {
    body = null;
  }
  if (!resp.ok) return { ok: false, status: resp.status };
  if (body === null) return { ok: false, status: resp.status };
  const parsed =
    provider.wire === "openai"
      ? parseOpenAIAnswer(body)
      : provider.wire === "anthropic"
        ? parseAnthropicAnswer(body)
        : parseGeminiAnswer(body);
  if (parsed === null || parsed.text === "") return { ok: false, status: resp.status };
  return { ok: true, ...parsed };
}

// ---------------------------------------------------------------------------
// Stop reasons, by family
// ---------------------------------------------------------------------------

function stopFamily(stop) {
  if (stop === "stop" || stop === "end_turn" || stop === "STOP") return 0;
  if (stop === "length" || stop === "max_tokens" || stop === "MAX_TOKENS") return 1;
  if (stop === "tool_calls" || stop === "function_call" || stop === "tool_use") return 2;
  if (stop === "content_filter" || stop === "refusal" || stop === "SAFETY") return 3;
  return 0;
}

function openAIStopReason(family) {
  if (family === 1) return "length";
  if (family === 2) return "tool_calls";
  if (family === 3) return "content_filter";
  return "stop";
}

function anthropicStopReason(family) {
  if (family === 1) return "max_tokens";
  if (family === 2) return "tool_use";
  if (family === 3) return "refusal";
  return "end_turn";
}

// ---------------------------------------------------------------------------
// Rendering the provider's answer in the wire the client used
// ---------------------------------------------------------------------------

function splitText(text, size) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) chunks.push(text.slice(i, i + size));
  if (chunks.length === 0) chunks.push("");
  return chunks;
}

function renderOpenAIJSON(answer, modelId, requestIndex, created) {
  return JSON.stringify({
    id: "chatcmpl-req-" + requestIndex,
    object: "chat.completion",
    created,
    model: modelId,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: openAIStopReason(stopFamily(answer.stop)),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
}

function renderOpenAISSE(answer, modelId, requestIndex, created) {
  const id = "chatcmpl-req-" + requestIndex;
  const stop = openAIStopReason(stopFamily(answer.stop));
  const frames = [];
  frames.push({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null, logprobs: null }],
  });
  for (const piece of splitText(answer.text, SSE_CHUNK_CHARS)) {
    frames.push({
      id,
      object: "chat.completion.chunk",
      created,
      model: modelId,
      choices: [{ index: 0, delta: { content: piece }, finish_reason: null, logprobs: null }],
    });
  }
  frames.push({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: stop, logprobs: null }],
  });
  frames.push({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  });
  return frames.map((f) => "data: " + JSON.stringify(f) + "\n\n").join("") + "data: [DONE]\n\n";
}

function renderAnthropicJSON(answer, modelId, requestIndex) {
  return JSON.stringify({
    id: "msg-req-" + requestIndex,
    type: "message",
    role: "assistant",
    model: modelId,
    content: [{ type: "text", text: answer.text }],
    stop_reason: anthropicStopReason(stopFamily(answer.stop)),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  });
}

function renderAnthropicSSE(answer, modelId, requestIndex) {
  const id = "msg-req-" + requestIndex;
  const stop = anthropicStopReason(stopFamily(answer.stop));
  const events = [];
  events.push([
    "message_start",
    {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    },
  ]);
  events.push([
    "content_block_start",
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
  ]);
  for (const piece of splitText(answer.text, SSE_CHUNK_CHARS)) {
    events.push([
      "content_block_delta",
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } },
    ]);
  }
  events.push(["content_block_stop", { type: "content_block_stop", index: 0 }]);
  events.push([
    "message_delta",
    {
      type: "message_delta",
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    },
  ]);
  events.push(["message_stop", { type: "message_stop" }]);
  return events.map(([name, payload]) => "event: " + name + "\ndata: " + JSON.stringify(payload) + "\n\n").join("");
}

function renderOpenAIError(message) {
  return JSON.stringify({
    error: { message, type: "server_error", param: null, code: "provider_unavailable" },
  });
}

function renderAnthropicError(message) {
  return JSON.stringify({ type: "error", error: { type: "api_error", message } });
}

// ---------------------------------------------------------------------------
// Serving one chat request
// ---------------------------------------------------------------------------

function statsFor(state, providerId) {
  let stats = state.providerStats.get(providerId);
  if (stats === undefined) {
    stats = { refusals: 0, dead: false };
    state.providerStats.set(providerId, stats);
  }
  return stats;
}

async function serveChat(state, api, body, index) {
  const norm = api === "openai" ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const manifestEntry = state.brief.manifest[index];
  let requires;
  if (manifestEntry !== undefined && manifestEntry.requires !== null && typeof manifestEntry.requires === "object") {
    requires = manifestEntry.requires;
  } else {
    // Beyond the manifest (which cannot happen in a trial): derive the
    // requirements from the body itself.
    requires = {
      tools: norm.tools.length > 0,
      vision: norm.messages.some((m) => m.parts.some((p) => p.t === "image")),
      json: norm.jsonMode,
    };
  }
  const ranked = legalCandidates(state.brief, state.table, state.providerById, requires, norm.jsonMode);
  let answer = null;
  let servingModel = null;
  for (const candidate of ranked) {
    const stats = statsFor(state, candidate.provider.id);
    if (stats.dead) continue;
    const result = await callProvider(state.env, candidate.provider, candidate.modelId, norm);
    if (result.ok) {
      stats.refusals = 0;
      answer = result;
      servingModel = candidate.modelId;
      break;
    }
    // A 400 or 401 is a malformed call, not provider state: it spends
    // no budget and says nothing about the provider's health, so it
    // does not count toward the outage verdict. Every other refusal
    // does, and three in a row means the provider is down.
    if (result.status !== 400 && result.status !== 401) {
      stats.refusals += 1;
      if (stats.refusals >= DEAD_AFTER_REFUSALS) stats.dead = true;
    }
    process.stderr.write(
      "provider " + candidate.provider.id + " refused with status " + result.status + " for request " + index + "\n",
    );
  }
  if (answer === null) {
    const message = "every provider refused the request";
    if (api === "openai") {
      return { status: 502, contentType: "application/json", body: renderOpenAIError(message) };
    }
    return { status: 502, contentType: "application/json", body: renderAnthropicError(message) };
  }
  if (api === "openai") {
    if (norm.stream) {
      return {
        status: 200,
        contentType: "text/event-stream; charset=utf-8",
        body: renderOpenAISSE(answer, servingModel, index, state.created),
      };
    }
    return {
      status: 200,
      contentType: "application/json",
      body: renderOpenAIJSON(answer, servingModel, index, state.created),
    };
  }
  if (norm.stream) {
    return {
      status: 200,
      contentType: "text/event-stream; charset=utf-8",
      body: renderAnthropicSSE(answer, servingModel, index),
    };
  }
  return { status: 200, contentType: "application/json", body: renderAnthropicJSON(answer, servingModel, index) };
}

// ---------------------------------------------------------------------------
// The model list: the wire its auth header names
// ---------------------------------------------------------------------------

function freeModelIds(state) {
  const seen = new Set();
  const ids = [];
  for (const p of state.catalog.providers) {
    for (const m of p.models) {
      if (m.free === true && !seen.has(m.id)) {
        seen.add(m.id);
        ids.push(m.id);
      }
    }
  }
  return ids;
}

function handleModels(state, req) {
  const ids = freeModelIds(state);
  const apiKey = req.headers["x-api-key"];
  if (typeof apiKey === "string" && apiKey !== "") {
    return {
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: ids.map((id) => ({
          type: "model",
          id,
          display_name: id,
          created_at: new Date(state.created * 1000).toISOString(),
          free: true,
        })),
        has_more: false,
      }),
    };
  }
  return {
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      object: "list",
      data: ids.map((id) => ({
        id,
        object: "model",
        created: state.created,
        owned_by: "free-model-aggregator",
        free: true,
      })),
    }),
  };
}

// ---------------------------------------------------------------------------
// The HTTP server
// ---------------------------------------------------------------------------

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_REQUEST_BODY) throw new Error("request body too large");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

// Runs each task after the previous one finishes, so provider calls are
// never concurrent. The queue itself never stays rejected.
function enqueue(state, fn) {
  const run = state.queue.then(fn, fn);
  state.queue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

function startServer(state) {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url || "/", "http://127.0.0.1");
    const path = url.pathname;
    if (req.method === "GET" && path === "/v1/models") {
      const answer = handleModels(state, req);
      res.writeHead(answer.status, { "content-type": answer.contentType });
      res.end(answer.body);
      return;
    }
    if (req.method === "POST" && (path === "/v1/chat/completions" || path === "/v1/messages")) {
      const api = path === "/v1/chat/completions" ? "openai" : "anthropic";
      // The index is taken on arrival, in arrival order, even for a
      // request whose body turns out unreadable: the client sends one
      // request per manifest entry and never retries, so the Nth
      // arrival is the Nth manifest entry whatever its body holds.
      const index = state.requestIndex;
      state.requestIndex += 1;
      enqueue(state, async () => {
        let answer;
        try {
          const raw = await readBody(req);
          let body = null;
          try {
            body = JSON.parse(raw);
          } catch {
            body = null;
          }
          if (body === null || typeof body !== "object" || Array.isArray(body)) {
            answer =
              api === "openai"
                ? { status: 400, contentType: "application/json", body: renderOpenAIError("the request body is not one JSON object") }
                : { status: 400, contentType: "application/json", body: renderAnthropicError("the request body is not one JSON object") };
          } else {
            answer = await serveChat(state, api, body, index);
          }
        } catch {
          answer =
            api === "openai"
              ? { status: 500, contentType: "application/json", body: renderOpenAIError("the server failed to answer the request") }
              : { status: 500, contentType: "application/json", body: renderAnthropicError("the server failed to answer the request") };
        }
        res.writeHead(answer.status, { "content-type": answer.contentType });
        res.end(answer.body);
      });
      return;
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end(
      JSON.stringify({ error: { message: "no such route", type: "not_found", param: null, code: "not_found" } }),
    );
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve(server));
  });
}

// ---------------------------------------------------------------------------
// Registration and the done-polling loop
// ---------------------------------------------------------------------------

async function register(env, port, catalog) {
  const url = env + "/register";
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS; attempt++) {
    let resp = null;
    try {
      resp = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch {
      resp = null;
    }
    if (resp !== null) {
      if (resp.ok) {
        const body = await resp.json().catch(() => null);
        if (body !== null && typeof body === "object" && body.ok === true) return body;
      } else if (resp.status === 400) {
        // The environment rejected the registration itself, and the
        // body is fixed, so retrying cannot help: report the
        // environment's reason and stop at once.
        const body = await resp.json().catch(() => null);
        const detail = body !== null && typeof body.message === "string" ? ": " + body.message : "";
        throw new Error("registration with the trial environment failed" + detail);
      }
    }
    await sleep(REGISTER_DELAY_MS);
  }
  throw new Error("registration with the trial environment failed");
}

async function waitUntilDone(env) {
  const url = env + "/work";
  // A missed answer changes nothing, but an environment that stays
  // unreachable for a full minute is gone rather than slow: stop
  // instead of polling until the trial wall. A healthy environment
  // answers every poll, so the bound never trips mid-trial.
  let consecutiveFailures = 0;
  for (;;) {
    let answered = false;
    try {
      const resp = await fetch(url, { signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
      if (resp.ok) {
        const body = await resp.json();
        if (body !== null && typeof body === "object" && body.done === true) return;
        answered = true;
      }
    } catch {
      // Keep polling; the bound below decides when to stop.
    }
    consecutiveFailures = answered ? 0 : consecutiveFailures + 1;
    if (consecutiveFailures > 600) {
      throw new Error("the trial environment stopped answering /work");
    }
    await sleep(POLL_DELAY_MS);
  }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

async function listMode() {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  const env = envBase();
  if (promptFile === undefined || promptFile === "" || env === "") {
    process.stdout.write(JSON.stringify({ providers: [] }) + "\n");
    return;
  }
  let brief;
  try {
    brief = parseBrief(readFileSync(promptFile, "utf8"));
  } catch {
    process.stdout.write(JSON.stringify({ providers: [] }) + "\n");
    return;
  }
  const catalog = await buildCatalog(env, brief);
  process.stdout.write(JSON.stringify(catalog) + "\n");
}

async function serverMode() {
  const promptFile = process.env.MAKEFUN_PROMPT_FILE;
  const env = envBase();
  if (promptFile === undefined || promptFile === "") throw new Error("MAKEFUN_PROMPT_FILE is not set");
  if (env === "") throw new Error("MAKEFUN_ENV_URL is not set");
  const brief = parseBrief(readFileSync(promptFile, "utf8"));
  const catalog = await buildCatalog(env, brief);
  const table = buildModelTable(brief, catalog);
  const providerById = new Map(brief.providers.map((p) => [p.id, p]));
  const state = {
    brief,
    catalog,
    table,
    providerById,
    env,
    created: trialNowEpoch(),
    requestIndex: 0,
    providerStats: new Map(),
    queue: Promise.resolve(),
  };
  const server = await startServer(state);
  const address = server.address();
  const port = address !== null && typeof address === "object" ? address.port : 0;
  if (port === 0) throw new Error("the server did not bind a port");
  await register(env, port, catalog);
  await waitUntilDone(env);
  server.close();
  process.exit(0);
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--list")) {
    await listMode();
    return;
  }
  await serverMode();
}

main().catch((err) => {
  process.stderr.write((err !== null && err !== undefined && err.message ? err.message : String(err)) + "\n");
  process.exitCode = 1;
});
