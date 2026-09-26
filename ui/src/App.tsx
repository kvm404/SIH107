import { useCallback, useEffect, useRef, useState } from "react";
import { bisChat, checkHealth, deleteThread, fetchTitle, sendFeedback, transcribeAudio } from "./api";
import {
  AnswerBody,
  CopyButton,
  FeedbackButtons,
  RawJson,
  RichText,
  SkeletonAnswer,
  StarterPrompts,
  cleanAnswerText,
} from "./components";
import type { AnswerTone, StarterPrompt } from "./components";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronDownIcon,
  DevIcon,
  GlobeIcon,
  ManakEmblemIcon,
  MicIcon,
  MicOffIcon,
  SpinnerIcon,
  MoonIcon,
  NewChatIcon,
  RetryIcon,
  SidebarToggleIcon,
  SunIcon,
  XIcon,
} from "./icons";
import type { ChatResponse, Lang, Msg } from "./types";
import type { ServerThread } from "./api";
import "./tokens.css";
import "./styles.css";

const APP_NAME = "Manak Mitra";
const APP_TAGLINE = "BIS Standards Assistant";
const DEV_FLAG_KEY = "manak-mitra-dev-mode";
const HISTORY_KEY = "manak-mitra-history";
const HISTORY_LIMIT = 20;

/** One example per kind of user the assistant serves. */
const STARTER_PROMPTS: StarterPrompt[] = [
  { who: "Manufacturer", text: "Which Indian Standard applies to steel water bottles, and is BIS certification compulsory?" },
  { who: "Electronics", text: "I make LED bulbs. Do I need CRS registration?" },
  { who: "Testing", text: "Suggest BIS recognised labs to test two-wheeler helmets" },
  { who: "Consumer", text: "How do I verify the HUID on my gold jewellery?" },
  { who: "Complaint", text: "How do I complain about a fake ISI mark?" },
  { who: "हिंदी", text: "प्रेशर कुकर के लिए कौन सा मानक अनिवार्य है?" },
];

/** Devanagari text gets lang="hi" so screen readers switch voice. */
const DEVANAGARI_RE = /[\u0900-\u097F]/;

/** Temporary states where the same question can simply be sent again. */
const RETRYABLE_KINDS = new Set(["model_unavailable", "model_busy"]);

/** Replies that are not answers get a notice frame instead of plain prose. */
function toneOf(kind: string | undefined): AnswerTone | undefined {
  if (!kind) return undefined;
  if (RETRYABLE_KINDS.has(kind)) return "busy";
  return kind === "grounding_refusal" ? "refusal" : undefined;
}

const LANG_OPTIONS: { id: Lang; label: string }[] = [
  { id: "auto", label: "Auto" },
  { id: "en", label: "English" },
  { id: "hi", label: "हिंदी" },
];

interface StoredMsg {
  role: "user" | "assistant";
  text: string;
  resp?: ChatResponse;
  error?: string;
}

interface HistoryEntry {
  q: string;
  /** short generated label; "" when nothing title-worthy yet */
  title: string;
  at: number;
  /** topic generation that created the entry; retitles only apply within it */
  topic: number;
  /** full thread so opening history does not regenerate or typewrite */
  msgs?: StoredMsg[];
}

let nextId = 1;

function snapshotMsgs(list: Msg[]): StoredMsg[] {
  return list
    .filter((m) => !m.retrying)
    .map(({ role, text, resp, error }) => ({ role, text, resp, error }));
}

function hydrateMsgs(list: StoredMsg[]): Msg[] {
  return list.map((m) => ({
    id: nextId++,
    role: m.role,
    text: m.text,
    resp: m.resp,
    error: m.error,
    streamIn: false,
  }));
}

/** Map a send failure to UI text. Expired threads are resumable on a fresh thread. */
function friendlyError(msg: string): { text: string; expired: boolean } {
  const expired = msg.startsWith("Thread expired");
  const text = /HTTP 429/.test(msg)
    ? "Rate limited. Please wait a minute and retry."
    : expired
      ? "This conversation has expired. Please start a new topic to continue."
      : "The chatbot is offline or the server could not be reached. Please check your connection and try again.";
  return { text, expired };
}

/** Turn a raw first query into a short sidebar title ("" = not title-worthy). */
function makeTitle(q: string): string {
  let s = (q ?? "").replace(/\s+/g, " ").trim().replace(/[!?.…,]+$/g, "").trim();
  if (!s) return "";
  const greeted = /^(hi|hello|hey|namaste)\b/i.test(s);
  s = s.replace(/^(hi|hello|hey|namaste)\b[,. ]+/i, "").trim();
  // "hello there, what is …" — drop the discourse filler, but never touch a
  // real existential ("There is a problem with…").
  if (greeted) s = s.replace(/^(there|here)\b[,. ]+/i, "").trim();
  if (!s) return "";
  if (/^(hi|hii+|hello|hey|namaste|namaskar|good\s?(morning|afternoon|evening))[!?.\s]*$/i.test(s)) return "";
  if (/^(how are you|how r u|what'?s up|who are you|how is it going|are you (there|online))[!?.\s]*$/i.test(s)) return "";
  if (/^[^aeiou\s]{9,}$/i.test(s)) return "";
  s = s
    .replace(/^(please\s+)?(can you|could you|would you|will you|kindly|please)\s+/i, "")
    .trim();
  // "explain in more detail, N words" is a follow-up style request, not a topic.
  if (/more\s+details?/i.test(s)) {
    const wc = s.match(/(\d+)\s*words?/i);
    return wc ? `More detail, ${wc[1]} words` : "More detail";
  }
  s = s
    .replace(/^(tell me|explain to me|explain|describe|answer|give me)(\s+in(\s+more)?\s+detail)?\s+/i, "")
    .trim();
  // Strip question scaffolding, but never the "IS" in "IS 10500".
  // Strip question scaffolding ("what does", "how do I"), but never the
  // "IS" in "IS 10500".
  for (let i = 0; i < 3 && !/^is\s*\d/i.test(s); i++) {
    const next = s
      .replace(/^(what(?:'s| is| are)?|which|how|why|when|where|is|are|do|does|can|should)\b\s+/i, "")
      .replace(/^(i|we)\b\s+/i, "")
      .trim();
    if (next === s) break;
    s = next;
  }
  if (!s || /^(that|this|it|there|here|yes|no|okay|ok|thanks|thank you)$/i.test(s)) return "";
  s = s.charAt(0).toUpperCase() + s.slice(1);
  if (s.length <= 44) return s;
  const cut = s.slice(0, 44);
  const sp = cut.lastIndexOf(" ");
  return (sp > 20 ? cut.slice(0, sp) : cut).trimEnd() + "…";
}

function loadHistory(): HistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (e): e is HistoryEntry =>
          typeof e === "object" &&
          e !== null &&
          typeof (e as HistoryEntry).q === "string" &&
          typeof (e as HistoryEntry).at === "number",
      )
      .map((e) => {
        // Entries written before topic tags existed predate LLM titles too,
        // so re-run the current heuristic over them exactly once per load.
        const legacy = typeof e.topic !== "number" || e.topic === -1;
        const msgs = Array.isArray((e as HistoryEntry).msgs)
          ? (e as HistoryEntry).msgs
          : undefined;
        return {
          q: e.q,
          title: legacy || typeof e.title !== "string" ? makeTitle(e.q) : e.title,
          at: e.at,
          topic: legacy ? -1 : (e.topic as number),
          msgs,
        };
      })
      .slice(0, HISTORY_LIMIT);
  } catch {
    return [];
  }
}

const THEME_KEY = "manak-mitra-theme";

/** Saved theme, else the system preference (index.html applies it before paint). */
function loadDarkMode(): boolean {
  try {
    const saved = window.localStorage.getItem(THEME_KEY);
    if (saved === "dark" || saved === "light") return saved === "dark";
  } catch {
    /* storage blocked: fall through to the system preference */
  }
  return typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function loadDevFlag(): boolean {
  try {
    return window.localStorage.getItem(DEV_FLAG_KEY) === "1";
  } catch {
    return false;
  }
}

/** Minimal Web Speech API typing (no `any`). */
interface SpeechRecognizer {
  lang: string;
  interimResults: boolean;
  onresult: ((ev: SpeechResultEvent) => void) | null;
  onerror: ((ev: SpeechErrorEvent) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

interface SpeechResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}

interface SpeechErrorEvent {
  error: string;
}

type SpeechRecognizerCtor = new () => SpeechRecognizer;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognizerCtor;
    webkitSpeechRecognition?: SpeechRecognizerCtor;
  }
}

export default function App() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [lang, setLang] = useState<Lang>("auto");
  const [busy, setBusy] = useState(false);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [thread, setThread] = useState<ServerThread | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(loadDarkMode);
  const [toast, setToast] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>(() =>
    typeof window === "undefined" ? [] : loadHistory(),
  );
  const [historyOpen, setHistoryOpen] = useState(true);
  const [devMode, setDevMode] = useState<boolean>(() =>
    typeof window === "undefined" ? false : loadDevFlag(),
  );
  // Epoch-based so reloaded history entry tags (small ints) can never
  // collide with this session's topic generations.
  const [topicKey, setTopicKey] = useState(() => Date.now());
  const [listening, setListening] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [voiceReady, setVoiceReady] = useState(false);
  const [langOpen, setLangOpen] = useState(false);
  const [showJump, setShowJump] = useState(false);
  const langMenuRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const heroBoxRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const toastTimer = useRef<number | null>(null);
  const recognizerRef = useRef<SpeechRecognizer | null>(null);
  const mediaRecRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  // Bumped when the user leaves a chat; replies to an older generation are
  // dropped instead of landing in the chat that replaced it.
  const genRef = useRef(0);
  const busyRef = useRef(false);
  const markBusy = useCallback((v: boolean) => {
    busyRef.current = v;
    setBusy(v);
  }, []);

  const speechSupported =
    typeof window !== "undefined" &&
    typeof MediaRecorder === "function" &&
    !!navigator.mediaDevices?.getUserMedia;

  useEffect(() => {
    document.title = `${APP_NAME} · ${APP_TAGLINE} | Bureau of Indian Standards`;
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    // Swap every colour at once; per-element transitions would fade unevenly.
    root.classList.add("theme-switching");
    root.classList.toggle("dark-theme", darkMode);
    const t = window.setTimeout(() => root.classList.remove("theme-switching"), 50);
    return () => {
      window.clearTimeout(t);
      root.classList.remove("dark-theme");
    };
  }, [darkMode]);

  const toggleTheme = useCallback(() => {
    setDarkMode((v) => {
      const next = !v;
      try {
        window.localStorage.setItem(THEME_KEY, next ? "dark" : "light");
      } catch {
        /* private mode: session only */
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSidebarOpen(false);
        setLangOpen(false);
        if (listening) {
          recognizerRef.current?.stop();
          mediaRecRef.current?.stop();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [listening]);

  useEffect(() => {
    if (!langOpen) return;
    const onDown = (e: MouseEvent) => {
      if (langMenuRef.current && !langMenuRef.current.contains(e.target as Node)) {
        setLangOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [langOpen]);

  useEffect(() => () => {
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    recognizerRef.current?.stop();
    mediaRecRef.current?.stop();
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  const showToast = useCallback((t: string) => {
    setToast(t);
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(""), 4000);
  }, []);

  const toggleDevMode = useCallback(() => {
    setDevMode((v) => {
      const next = !v;
      try {
        window.localStorage.setItem(DEV_FLAG_KEY, next ? "1" : "0");
      } catch {
        /* private mode — session only */
      }
      return next;
    });
  }, []);

  const persistHistory = useCallback((next: HistoryEntry[]) => {
    const capped = next.slice(0, HISTORY_LIMIT);
    try {
      window.localStorage.setItem(HISTORY_KEY, JSON.stringify(capped));
    } catch {
      /* private mode — session only */
    }
    return capped;
  }, []);

  const rememberTopic = useCallback((q: string, fresh?: boolean, topic?: number) => {
    const title = makeTitle(q);
    setHistory((h) => {
      let next: HistoryEntry[];
      const idx = h.findIndex((e) => e.q === q);
      if (idx >= 0) {
        // Same question asked again: bump to top, fill in a title if it lacked one.
        const cur = h[idx];
        const upd = { ...cur, at: Date.now(), title: cur.title || title, topic: topic ?? cur.topic };
        next = [upd, ...h.slice(0, idx), ...h.slice(idx + 1)];
      } else if (!fresh && h.length > 0 && !h[0].title && title && h[0].topic === topic) {
        // Topic opened with a greeting, real question arrived: retitle the head entry.
        next = [{ q, title, at: Date.now(), topic: topic ?? -1 }, ...h.slice(1)];
      } else if (!fresh && topic !== undefined && topic !== -1 && h.some((e) => e.topic === topic)) {
        // Follow-up in an open chat: one sidebar entry per chat, keyed by its
        // opening question, so just bump it to the top.
        const at = h.findIndex((e) => e.topic === topic);
        next = [{ ...h[at], at: Date.now() }, ...h.slice(0, at), ...h.slice(at + 1)];
      } else {
        next = [{ q, title, at: Date.now(), topic: topic ?? -1 }];
        next.push(...h);
      }
      return persistHistory(next);
    });
  }, [persistHistory]);

  useEffect(() => {
    const el = mainRef.current;
    if (!el) return;
    const onScroll = () => {
      setShowJump(el.scrollHeight - el.scrollTop - el.clientHeight > 300);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (input === "" && heroBoxRef.current) heroBoxRef.current.style.height = "";
  }, [input]);

  const ping = useCallback(async () => {
    const h = await checkHealth();
    setHealthy(h.ok);
    setVoiceReady(h.speech);
  }, []);

  const jumpToBottom = useCallback(() => {
    const reduceMotion =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    bottomRef.current?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  }, []);

  useEffect(() => {
    ping();
    const t = setInterval(ping, 30000);
    return () => clearInterval(t);
  }, [ping]);

  useEffect(() => {
    if (msgs.length > 0) {
      const reduceMotion =
        typeof window !== "undefined" &&
        typeof window.matchMedia === "function" &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      bottomRef.current?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
    }
  }, [msgs, busy]);

  const saveTranscript = useCallback(
    (list: Msg[]) => {
      const first = list.find((m) => m.role === "user");
      if (!first) return;
      const snap = snapshotMsgs(list);
      setHistory((h) => {
        const idx = h.findIndex((e) => e.q === first.text);
        if (idx < 0) return h;
        const cur = h[idx];
        const prev = cur.msgs;
        if (
          prev &&
          prev.length === snap.length &&
          prev[prev.length - 1]?.text === snap[snap.length - 1]?.text
        ) {
          return h;
        }
        return persistHistory(
          h.map((e, i) => (i === idx ? { ...cur, msgs: snap } : e)),
        );
      });
    },
    [persistHistory],
  );

  const send = useCallback(
    async (query: string, opts?: { force?: boolean; fresh?: boolean; streamIn?: boolean }) => {
      const q = query.trim();
      if (!q || busyRef.current) return;
      markBusy(true);
      const gen = genRef.current;
      const streamIn = opts?.streamIn !== false;
      const userMsg: Msg = { id: nextId++, role: "user", text: q };
      setMsgs((m) => [...m, userMsg]);
      setInput("");
      // One attempt, optionally re-run once on a fresh server thread when the
      // old one expired. The user bubble is appended once, above.
      const attempt = async (fresh: boolean): Promise<void> => {
        const useThread = fresh ? bisChat.newTopic() : thread;
        try {
          const { resp, thread: next } = await bisChat.send(q, {
            lang,
            thread: useThread,
            force: opts?.force ?? false,
            fresh,
          });
          if (genRef.current !== gen) return;
          setMsgs((m) => [
            ...m,
            {
              id: nextId++,
              role: "assistant",
              text: resp.text,
              resp,
              feedback: null,
              retryQ: q,
              streamIn,
            },
          ]);
          rememberTopic(q, fresh, topicKey);
          setThread(bisChat.shouldKeepThread(resp) ? next : null);
        } catch (e) {
          if (genRef.current !== gen) return;
          const msg = e instanceof Error ? e.message : "request failed";
          const { text: friendly, expired } = friendlyError(msg);
          if (expired && !fresh) {
            setThread(null);
            return attempt(true);
          }
          if (expired) setThread(null);
          setMsgs((m) => [...m, { id: nextId++, role: "assistant", text: "", error: friendly, retryQ: q }]);
          setHealthy(false);
        }
      };
      try {
        await attempt(opts?.fresh ?? false);
      } finally {
        if (genRef.current === gen) markBusy(false);
      }
    },
    [lang, thread, topicKey, rememberTopic, markBusy],
  );

  useEffect(() => {
    if (busy || msgs.length === 0) return;
    saveTranscript(msgs);
  }, [msgs, busy, history, saveTranscript]);

  // Re-run a failed turn inside its own message slot: no duplicate question,
  // shimmer while running, new answer (or error) swaps in place.
  const retryMessage = useCallback(
    async (id: number) => {
      const target = msgs.find((m) => m.id === id);
      const q = target?.retryQ?.trim();
      if (!q || busyRef.current || target?.role !== "assistant") return;
      markBusy(true);
      const gen = genRef.current;
      setMsgs((m) => m.map((x) => (x.id === id ? { ...x, retrying: true } : x)));
      const t0 = Date.now();
      let outcome: Partial<Msg> | null = null;
      try {
        const { resp, thread: next } = await bisChat.send(q, { lang, thread });
        setThread(bisChat.shouldKeepThread(resp) ? next : null);
        outcome = { text: resp.text, resp, error: undefined, feedback: null, retryQ: q, streamIn: true };
        rememberTopic(q, false, topicKey);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "request failed";
        const { text: friendly, expired } = friendlyError(msg);
        if (expired) setThread(null);
        outcome = { text: "", error: friendly, retryQ: q };
        setHealthy(false);
      }
      // Let the shimmer paint at least briefly so the retry reads as intentional.
      const wait = 600 - (Date.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      if (genRef.current !== gen) return;
      const final = outcome;
      setMsgs((m) => m.map((x) => (x.id === id ? { ...x, ...final, retrying: false } : x)));
      markBusy(false);
    },
    [msgs, lang, thread, topicKey, rememberTopic, markBusy],
  );

  const newTopic = useCallback(() => {
    // A new chat is a clean break: drop the server thread (erased, not just
    // hidden), clear messages, draft, and pending state.
    if (thread) void deleteThread(thread);
    genRef.current += 1;
    markBusy(false);
    setThread(null);
    setMsgs([]);
    setInput("");
    setTopicKey((k) => k + 1);
    setSidebarOpen(false);
    inputRef.current?.focus();
  }, [thread, markBusy]);

  const openHistoryTopic = useCallback(
    (q: string) => {
      const entry = history.find((e) => e.q === q);
      if (thread) void deleteThread(thread);
      genRef.current += 1;
      markBusy(false);
      setThread(null);
      setInput("");
      const reopened = topicKey + 1;
      setTopicKey(reopened);
      setSidebarOpen(false);
      if (entry?.msgs && entry.msgs.length > 0) {
        // Follow-ups in the reopened chat belong to this same entry.
        setHistory((h) =>
          persistHistory(h.map((e) => (e.q === q ? { ...e, topic: reopened } : e))),
        );
        setMsgs(hydrateMsgs(entry.msgs));
        return;
      }
      setMsgs([]);
      void send(q, { fresh: true, streamIn: false });
    },
    [send, thread, history, topicKey, persistHistory, markBusy],
  );

  // Upgrade the instant heuristic title with an LLM one once the first
  // grounded reply lands. Fires once per question; any failure keeps the heuristic.
  const titleReqRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const last = [...msgs]
      .reverse()
      .find((m) => m.role === "assistant" && m.resp && !m.error);
    if (!last?.resp || RETRYABLE_KINDS.has(last.resp.kind) || !last.text) return;
    const uq = [...msgs.slice(0, msgs.lastIndexOf(last))]
      .reverse()
      .find((m) => m.role === "user");
    if (!uq || titleReqRef.current.has(uq.text)) return;
    const entry = history.find((e) => e.q === uq.text);
    if (!entry || (entry.title && entry.title !== makeTitle(entry.q))) return;
    titleReqRef.current.add(uq.text);
    const answerLang: Lang = last.resp.lang === "hi" || lang === "hi" ? "hi" : "en";
    void fetchTitle(uq.text, last.text, answerLang).then((t) => {
      if (!t) return;
      setHistory((h) =>
        persistHistory(h.map((e) => (e.q === uq.text ? { ...e, title: t } : e))),
      );
    });
  }, [msgs, history, lang, persistHistory]);

  const rate = useCallback(
    async (id: number, rating: 1 | -1) => {
      const target = msgs.find((m) => m.id === id);
      if (!target?.resp?.thread_id && !thread) {
        setMsgs((m) => m.map((x) => (x.id === id ? { ...x, feedback: rating } : x)));
        return;
      }
      const tid = target?.resp?.thread_id ?? thread?.id ?? "local";
      const ownerToken =
        thread?.id === tid ? thread.token : target?.resp?.owner_token || thread?.token;
      setMsgs((m) => m.map((x) => (x.id === id ? { ...x, feedback: rating } : x)));
      const res = await sendFeedback(tid, rating, ownerToken);
      if (!res.ok) showToast(`Feedback failed: ${res.error ?? "request failed"}.`);
      else if (res.fixture) showToast("Feedback recorded locally.");
      else showToast("Thank you. Feedback submitted for quality evaluation.");
    },
    [msgs, thread, showToast],
  );

  const toggleListening = useCallback(() => {
    if (transcribing) return;
    if (!voiceReady) return;
    if (listening) {
      mediaRecRef.current?.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder !== "function") {
      return;
    }
    void (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : MediaRecorder.isTypeSupported("audio/webm")
            ? "audio/webm"
            : "";
        const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
        const chunks: Blob[] = [];
        rec.ondataavailable = (ev) => {
          if (ev.data.size) chunks.push(ev.data);
        };
        rec.onerror = () => {
          setListening(false);
          setTranscribing(false);
          stream.getTracks().forEach((t) => t.stop());
        };
        rec.onstop = () => {
          setListening(false);
          stream.getTracks().forEach((t) => t.stop());
          mediaRecRef.current = null;
          mediaStreamRef.current = null;
          const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
          if (blob.size < 400) {
            setTranscribing(false);
            showToast("Didn't catch that. Tap the mic and try again.");
            return;
          }
          setTranscribing(true);
          void transcribeAudio(blob)
            .then((text) => {
              if (text) setInput((prev) => (prev ? `${prev} ${text}` : text));
              else showToast("Could not transcribe. Type instead.");
            })
            .catch(() => {
              setVoiceReady(false);
            })
            .finally(() => setTranscribing(false));
        };
        mediaRecRef.current = rec;
        rec.start();
        setListening(true);
      } catch (err) {
        const name = err instanceof DOMException ? err.name : "";
        if (name === "NotAllowedError") {
          showToast("Microphone is blocked. Allow access in the address bar, then retry.");
        } else if (name === "NotFoundError") {
          showToast("No microphone found.");
        }
      }
    })();
  }, [listening, transcribing, voiceReady, showToast]);

  const currentFirst = msgs.length > 0 ? msgs[0].text : null;
  const currentTitle = currentFirst
    ? history.find((e) => e.q === currentFirst)?.title || makeTitle(currentFirst) || "New conversation"
    : null;
  const pastTopics = history.filter((e) => e.q !== currentFirst).slice(0, 8);


  const renderComposer = (centered: boolean) => (
    <div
      className={
        centered
          ? "floating-composer-container composer-centered"
          : "floating-composer-container"
      }
    >
      <form
        className="floating-capsule"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <label className="sr-only" htmlFor={centered ? "composer-input-hero" : "composer-input"}>
          Ask about Indian Standards
        </label>
        {centered ? (
          <textarea
            id="composer-input-hero"
            ref={heroBoxRef}
            className="capsule-input-field composer-multiline"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              const el = e.target;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 170)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            placeholder="Ask anything about Indian Standards, ISI mark, CRS..."
            autoComplete="off"
            rows={4}
          />
        ) : (
          <input
            id="composer-input"
            ref={inputRef}
            type="text"
            className="capsule-input-field"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask anything about Indian Standards, ISI mark, CRS..."
            autoComplete="off"
          />
        )}

        {speechSupported && (
          <button
            type="button"
            className={[
              "capsule-icon-btn",
              listening ? "listening" : "",
              transcribing ? "transcribing" : "",
              !voiceReady ? "muted" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            title={
              !voiceReady
                ? "Voice is unavailable"
                : transcribing
                  ? "Transcribing"
                  : listening
                    ? "Stop listening"
                    : "Voice input"
            }
            onClick={toggleListening}
            disabled={!voiceReady || transcribing}
            aria-label={
              !voiceReady
                ? "Voice is unavailable"
                : transcribing
                  ? "Transcribing"
                  : listening
                    ? "Stop voice input"
                    : "Start voice input"
            }
            aria-pressed={listening}
          >
            {transcribing ? (
              <SpinnerIcon className="mic-spin" size={18} />
            ) : !voiceReady ? (
              <MicOffIcon size={18} />
            ) : (
              <MicIcon size={18} />
            )}
          </button>
        )}

        <button
          type="submit"
          className={`capsule-send-circle${input.trim() && !busy ? " active" : ""}`}
          disabled={busy || !input.trim()}
          aria-label="Send query"
        >
          <ArrowUpIcon size={16} />
        </button>
      </form>
      <p className="composer-disclaimer">
        {APP_NAME} answers from official BIS sources. Verify critical compliance decisions
        with BIS.
      </p>
    </div>
  );

  void healthy;

  return (
    <div className={`app-container${darkMode ? " dark-theme" : ""}`}>
      <a className="skip-link" href="#chat-log">
        Skip to conversation
      </a>
      {sidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={`clean-sidebar${sidebarOpen ? " open" : ""}${
          sidebarCollapsed ? " collapsed" : ""
        }`}
        aria-label="Primary"
      >
        <div className="sidebar-top-section">
          <div className="sidebar-brand-row">
            <button
              type="button"
              className="sidebar-rail-toggle"
              onClick={() => {
                if (window.innerWidth <= 840) {
                  setSidebarOpen(false);
                } else {
                  setSidebarCollapsed((v) => !v);
                }
              }}
              aria-label={sidebarCollapsed ? "Open sidebar" : "Close sidebar"}
              title={sidebarCollapsed ? "Open sidebar" : "Close sidebar"}
            >
              <span className="sidebar-rail-logo" aria-hidden="true">
                <ManakEmblemIcon size={26} />
              </span>
              <span className="sidebar-rail-panel" aria-hidden="true">
                <SidebarToggleIcon size={20} />
              </span>
            </button>
            <span className="brand-name-text">{APP_NAME}</span>
            {sidebarOpen && (
              <button
                type="button"
                className="mobile-close-btn"
                onClick={() => setSidebarOpen(false)}
                aria-label="Close navigation"
              >
                <XIcon size={18} />
              </button>
            )}
          </div>

          <button
            type="button"
            className="btn-new-chat"
            onClick={newTopic}
            title="New chat"
            aria-label="New chat"
          >
            <NewChatIcon size={sidebarCollapsed ? 18 : 15} />
            <span>New chat</span>
          </button>

          <div className="history-section">
            <button
              type="button"
              className="history-toggle"
              onClick={() => setHistoryOpen((v) => !v)}
              aria-expanded={historyOpen}
              aria-controls="history-list"
            >
              <span className="history-label">History</span>
              <ChevronDownIcon
                size={14}
                className={`history-chevron${historyOpen ? " open" : ""}`}
              />
            </button>
            {historyOpen && (
              <div className="history-list" id="history-list">
                {currentFirst && (
                  <button
                    type="button"
                    className="history-item active"
                    aria-current="true"
                    title={currentFirst}
                  >
                    <span className="history-item-text">{currentTitle}</span>
                  </button>
                )}
                {pastTopics.map((e) => (
                  <button
                    key={`${e.at}-${e.q}`}
                    type="button"
                    className="history-item"
                    onClick={() => openHistoryTopic(e.q)}
                    title={e.title || e.q}
                  >
                    <span className="history-item-text">{e.title || "New conversation"}</span>
                  </button>
                ))}
                {!currentFirst && pastTopics.length === 0 && (
                  <p className="history-empty">No conversations yet.</p>
                )}
              </div>
            )}
          </div>
        </div>
        <div className="sidebar-foot">
          <p className="sidebar-foot-text">Grounded in official BIS sources.</p>
        </div>
      </aside>

      <div className="clean-main-canvas">
        <header className="clean-topbar">
          <div className="topbar-left-zone">
            <button
              type="button"
              className="icon-btn icon-btn-lg topbar-sidebar-toggle"
              onClick={() => setSidebarOpen((v) => !v)}
              aria-label="Open navigation"
              aria-expanded={sidebarOpen}
              title="Open sidebar"
            >
              <SidebarToggleIcon size={20} />
            </button>
            <span className="topbar-title">
              {APP_NAME} <span className="topbar-title-sub">· {APP_TAGLINE}</span>
            </span>
          </div>

          <div className="topbar-right-zone">
            <div className="lang-menu-wrap" ref={langMenuRef}>
              <button
                type="button"
                className="lang-menu-btn"
                onClick={() => setLangOpen((v) => !v)}
                aria-haspopup="menu"
                aria-expanded={langOpen}
                aria-label="Response language"
                title="Response language"
              >
                <GlobeIcon size={15} />
                <span>{LANG_OPTIONS.find((o) => o.id === lang)?.label ?? "Auto"}</span>
                <ChevronDownIcon
                  size={13}
                  className={`lang-chevron${langOpen ? " open" : ""}`}
                />
              </button>
              {langOpen && (
                <div className="lang-menu" role="menu" aria-label="Response language">
                  {LANG_OPTIONS.map((o) => (
                    <button
                      key={o.id}
                      type="button"
                      role="menuitemradio"
                      aria-checked={lang === o.id}
                      className={`lang-menu-item${lang === o.id ? " active" : ""}`}
                      onClick={() => {
                        setLang(o.id);
                        setLangOpen(false);
                      }}
                    >
                      <span>{o.label}</span>
                      {lang === o.id && <CheckIcon size={13} />}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              type="button"
              className={`icon-btn icon-btn-lg${devMode ? " active" : ""}`}
              onClick={toggleDevMode}
              aria-label="Toggle developer details"
              aria-pressed={devMode}
              title={devMode ? "Hide raw API responses" : "Show raw API responses"}
            >
              <DevIcon size={18} />
            </button>
            <button
              type="button"
              className={`icon-btn icon-btn-lg${darkMode ? " toggled" : ""}`}
              onClick={toggleTheme}
              aria-label="Toggle theme"
              aria-pressed={darkMode}
              title="Toggle light / dark mode"
            >
              {darkMode ? <SunIcon size={20} /> : <MoonIcon size={20} />}
            </button>
          </div>
        </header>

        <main
          ref={mainRef}
          className="clean-body-content"
          id="chat-log"
          role="log"
          aria-live="polite"
          aria-label="Conversation"
        >
          <div className="chat-layout-wrap" key={topicKey}>
            {msgs.length === 0 ? (
              <div className="hero-center-container view-enter">
                <div className="hero-emblem-wrap">
                  <ManakEmblemIcon size={46} />
                </div>
                <h1 className="hero-headline">
                  Namaste, I&apos;m <span className="hero-bold-name">{APP_NAME}</span>
                </h1>
                <p className="hero-sub">
                  Ask about Indian Standards, ISI and CRS certification, hallmarking,
                  lab testing and BIS services, in English or हिंदी. Every answer cites
                  its official source.
                </p>

                {renderComposer(true)}
                <StarterPrompts
                  prompts={STARTER_PROMPTS}
                  disabled={busy}
                  onPick={(text) => void send(text)}
                />
              </div>
            ) : (
              <div className="chat-messages-container view-enter">
                {msgs.map((m) =>
                  m.role === "user" ? (
                    <div key={m.id} className={`user-message-row${m.streamIn ? " msg-enter" : ""}`}>
                      <div className="user-stack">
                        <div className="user-bubble" lang={DEVANAGARI_RE.test(m.text) ? "hi" : undefined}>
                          <RichText text={m.text} />
                        </div>
                        <div className="user-actions-row">
                          <CopyButton text={m.text} />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div key={m.id} className={`assistant-message-row${m.streamIn ? " msg-enter" : ""}`}>
                      <div className="assistant-avatar">
                        <ManakEmblemIcon size={24} />
                      </div>
                      <div className="assistant-content" lang={m.resp?.lang === "hi" ? "hi" : undefined}>
                        {m.retrying ? (
                          <div className="clean-answer-container">
                            <div className="sk-lines" aria-hidden="true">
                              <span className="sk-line sk-w90" />
                              <span className="sk-line sk-w70" />
                            </div>
                            <span className="sr-only">Retrying your question</span>
                          </div>
                        ) : m.error ? (
                          <div className="clean-error-card" role="alert">
                            <p className="error-text">{m.error}</p>
                              <button
                              type="button"
                              className="pill-btn pill-btn-solid"
                              disabled={busy}
                              onClick={() => void retryMessage(m.id)}
                            >
                              <RetryIcon size={14} />
                              <span>Retry</span>
                            </button>
                          </div>
                        ) : (
                          <div className="clean-answer-container">
                            <AnswerBody
                              text={cleanAnswerText(m.text)}
                              animate={!!m.streamIn && !RETRYABLE_KINDS.has(m.resp?.kind ?? "")}
                              sources={m.resp?.sources}
                              related={m.resp?.related_sources}
                              tone={toneOf(m.resp?.kind)}
                            />
                            {m.resp && <RawJson data={m.resp} enabled={devMode} />}

                            <div className="message-footer-row">
                              <div className="footer-left">
                                <CopyButton text={cleanAnswerText(m.text)} />
                                {RETRYABLE_KINDS.has(m.resp?.kind ?? "") && m.retryQ && (
                                  <button
                                    type="button"
                                    className="pill-btn pill-btn-sm"
                                    disabled={busy}
                                    onClick={() => void retryMessage(m.id)}
                                    title="Resend this question"
                                  >
                                    <RetryIcon size={13} />
                                    <span>Retry</span>
                                  </button>
                                )}
                              </div>
                              {!RETRYABLE_KINDS.has(m.resp?.kind ?? "") && (
                                <FeedbackButtons
                                  value={m.feedback}
                                  disabled={busy}
                                  onRate={(r) => rate(m.id, r)}
                                />
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ),
                )}

                {busy && !msgs.some((m) => m.retrying) && <SkeletonAnswer />}
                <div ref={bottomRef} />
              </div>
            )}

            {msgs.length > 0 && renderComposer(false)}
          </div>
        </main>
        {showJump && msgs.length > 0 && (
          <button
            type="button"
            className="jump-bottom"
            onClick={jumpToBottom}
            aria-label="Jump to latest messages"
            title="Jump to latest"
          >
            <ArrowDownIcon size={16} />
          </button>
        )}
      </div>

      {toast && (
        <div className="toast-notification" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
