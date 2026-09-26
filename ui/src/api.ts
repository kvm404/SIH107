import type { ChatResponse, Lang } from "./types";
import type { FeedbackResult } from "./types";

export interface ServerThread {
  id: string;
  token: string;
}

const CHAT_TIMEOUT_MS = 120_000;

async function req(path: string, init?: RequestInit, timeoutMs = 15000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(path, { ...init, signal: ctl.signal });
    if (r.status === 410) throw new Error("Thread expired — starting a new topic");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}

export async function checkHealth(): Promise<{ ok: boolean; speech: boolean }> {
  try {
    const j = await req("/api/health");
    return { ok: j?.ok === true, speech: j?.speech === true };
  } catch {
    return { ok: false, speech: false };
  }
}

export async function transcribeAudio(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    bin += String.fromCharCode(...bytes.subarray(i, i + step));
  }
  const audio_b64 = btoa(bin);
  const j = (await req(
    "/api/transcribe",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_b64, mime: blob.type || "audio/webm" }),
    },
    45_000,
  )) as { text?: string };
  return typeof j?.text === "string" ? j.text.trim() : "";
}

export async function sendChat(
  query: string,
  lang: Lang,
  thread?: ServerThread | null,
  force = false,
  newTopic = false,
): Promise<{ resp: ChatResponse; ms: number; thread: ServerThread | null }> {
  const t0 = performance.now();
  const body: Record<string, unknown> = { query, force, new_topic: newTopic };
  if (lang !== "auto") body.lang = lang;
  // Design 3: new_topic ignores thread server-side; don't send a stale id.
  if (thread && !newTopic) body.thread_id = thread.id;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (thread && !newTopic) headers["X-Owner-Token"] = thread.token;
  const raw = (await req("/api/chat", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  }, CHAT_TIMEOUT_MS)) as Partial<ChatResponse>;
  const resp = normalizeChatResponse(raw, query);
  const next: ServerThread | null = resp.thread_id
    ? { id: resp.thread_id, token: resp.owner_token ?? thread?.token ?? "" }
    : null;
  return { resp, ms: Math.round(performance.now() - t0), thread: next };
}

/**
 * Validate/normalize any /chat payload into a full ChatResponse (issue #4
 * P1-13): proxy errors, version drift, or truncated bodies must degrade to
 * an inline error answer instead of crashing on `resp!.field` access.
 */
export function normalizeChatResponse(raw: unknown, query = ""): ChatResponse {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const str = (v: unknown, fb = ""): string => (typeof v === "string" ? v : fb);
  const arr = <T>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);
  const lang = r.lang === "hi" ? "hi" : "en";
  if (typeof r.text !== "string" || typeof r.refused !== "boolean") {
    return {
      text: "The chatbot is unavailable because the server returned an invalid response. Please try again later.",
      refused: false,
      kind: "model_unavailable",
      lang,
      citations: [],
      pii: {},
      needs_info: false,
      questions: [],
      known: [],
      assumptions: [],
      context: { history: query ? [query] : [], rounds: 0 },
      rag_used_llm: false,
      model_available: false,
    };
  }
  return {
    text: str(r.text),
    refused: r.refused as boolean,
    kind: str(r.kind, "answered"),
    lang,
    citations: arr<string>(r.citations).filter((c) => typeof c === "string"),
    pii: (r.pii && typeof r.pii === "object" ? r.pii : {}) as Record<string, boolean>,
    needs_info: r.needs_info === true,
    questions: arr(r.questions),
    known: arr(r.known),
    assumptions: arr<string>(r.assumptions).filter((a) => typeof a === "string"),
    context: (r.context && typeof r.context === "object"
      ? r.context
      : { history: [], rounds: 0 }) as ChatResponse["context"],
    structured_citations: arr(r.structured_citations),
    thread_id: typeof r.thread_id === "string" ? r.thread_id : undefined,
    owner_token: typeof r.owner_token === "string" ? r.owner_token : undefined,
    sources: arr(r.sources ?? r.rag_evidence),
    rag_evidence: arr(r.rag_evidence ?? r.sources),
    related_sources: arr(r.related_sources),
    search_query: typeof r.search_query === "string" ? r.search_query : undefined,
    retryable: r.retryable === true,
    rag_mode: typeof r.rag_mode === "string" ? r.rag_mode : undefined,
    rag_used_llm: r.rag_used_llm === true,
    model_available: r.model_available === true,
    intent: typeof r.intent === "string" ? r.intent : undefined,
    intent_confidence: typeof r.intent_confidence === "string" ? r.intent_confidence : undefined,
    context_summary: typeof r.context_summary === "string" ? r.context_summary : undefined,
    guidance_adaptive: r.guidance_adaptive === true,
  };
}

/**
 * Design 3 common-case facade: one happy-path entry point with strong defaults.
 * `sendChat` above stays as the low-level compat primitive; new code should
 * prefer `bisChat.send()` which centralises the fresh/force policy.
 */
export interface BisChatSendOpts {
  lang?: Lang;
  thread?: ServerThread | null;
  force?: boolean;
  /** Drop follow-up context server-side (POSTs new_topic, sends no stale id). */
  fresh?: boolean;
}

export const bisChat = {
  async send(
    query: string,
    opts: BisChatSendOpts = {},
  ): Promise<{ resp: ChatResponse; ms: number; thread: ServerThread | null }> {
    const { lang = "auto", thread = null, force = false, fresh = false } = opts;
    return sendChat(query, lang, fresh ? null : thread, force, fresh);
  },
  /** Keep context across every model turn, not only clarification turns. */
  shouldKeepThread(resp: ChatResponse): boolean {
    return true;
  },
  newTopic(): null {
    return null;
  },
};

/**
 * POST /feedback when the backend ships it
 * (body {thread_id, rating: -1..1, note}, X-Owner-Token for owned threads).
 * Honest states (issue #4 P1-14): `{ok:true}` live, `{ok:true, fixture:true}`
 * when the backend has no feedback route yet, `{ok:false, error}` when the
 * call itself failed — callers must surface that instead of thanking.
 */
export async function sendFeedback(
  threadId: string,
  rating: 1 | -1,
  ownerToken?: string,
  note?: string,
): Promise<FeedbackResult> {
  try {
    const j = await req("/api/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(ownerToken ? { "X-Owner-Token": ownerToken } : {}),
      },
      // Backend schema: {thread_id, rating: -1..1, note: str ≤1000}; UI only ever sends ±1.
      body: JSON.stringify({ thread_id: threadId, rating, note: note ?? "" }),
    });
    if (j && j.ok === true) return { ok: true };
    return { ok: false, error: "feedback was not accepted" };
  } catch (e) {
    const msg = e instanceof Error ? e.message : "request failed";
    if (/HTTP 404/.test(msg)) return { ok: true, fixture: true };
    return { ok: false, error: msg };
  }
}

/** POST /title with the opening exchange; null on any failure (caller keeps heuristic). */
export async function fetchTitle(
  userText: string,
  assistantText: string,
  lang: Lang,
): Promise<string | null> {
  try {
    const j = await req(
      "/api/title",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_text: userText.slice(0, 1000),
          assistant_text: assistantText.slice(0, 1500),
          lang: lang === "auto" ? "en" : lang,
        }),
      },
      30000,
    );
    const t = typeof j?.title === "string" ? j.title.trim() : "";
    return t || null;
  } catch {
    return null;
  }
}

/** DELETE /threads/{id} with owner token; never throws (best-effort cleanup). */
export async function deleteThread(thread: ServerThread): Promise<void> {
  try {
    await req(
      `/api/threads/${encodeURIComponent(thread.id)}`,
      { method: "DELETE", headers: { "X-Owner-Token": thread.token } },
      8000,
    );
  } catch {
    /* abandoned threads expire server-side via thread TTL */
  }
}
