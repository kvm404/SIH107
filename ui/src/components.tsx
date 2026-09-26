import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import type { RagSource } from "./types";
import {
  CheckIcon,
  ChevronDownIcon,
  CopyIcon,
  ExternalLinkIcon,
  XIcon,
  ThumbsDownIcon,
  ThumbsUpIcon,
} from "./icons";

/** Opens source n (1-based) of the answer being rendered; null outside answers. */
const CitationContext = createContext<((n: number) => void) | null>(null);

/** Superscript citation marker; clicking opens that source in the panel. */
function Cite({ n }: { n: number }) {
  const onCite = useContext(CitationContext);
  if (!onCite) return <sup className="cite cite-static">{n}</sup>;
  return (
    <sup>
      <button
        type="button"
        className="cite"
        onClick={() => onCite(n)}
        aria-label={`Source ${n}`}
        title={`Show source ${n}`}
      >
        {n}
      </button>
    </sup>
  );
}

const CITE_RE = /\[(\d{1,2})\]/g;

function citeify(text: string, keyPrefix: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let last = 0;
  let k = 0;
  let m: RegExpExecArray | null;
  CITE_RE.lastIndex = 0;
  while ((m = CITE_RE.exec(text)) !== null) {
    if (m.index > last) out.push(<React.Fragment key={`${keyPrefix}-${k++}`}>{text.slice(last, m.index)}</React.Fragment>);
    out.push(<Cite key={`${keyPrefix}-${k++}`} n={Number(m[1])} />);
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(<React.Fragment key={`${keyPrefix}-${k++}`}>{text.slice(last)}</React.Fragment>);
  return out;
}

/** Inline markdown: **bold**, [n] citations, auto-linked https:// URLs. No HTML injection. */
const URL_RE = /(https?:\/\/[^\s)<\]]+)/g;

export function renderInline(body: string, keyPrefix: string): React.ReactNode[] {
  const boldParts = body.split("**");
  const out: React.ReactNode[] = [];
  boldParts.forEach((chunk, bi) => {
    if (bi % 2 === 1) {
      out.push(
        <strong key={`${keyPrefix}-b${bi}`}>
          {linkifyChunk(chunk, `${keyPrefix}-b${bi}`)}
        </strong>,
      );
      return;
    }
    out.push(...linkifyChunk(chunk, `${keyPrefix}-t${bi}`));
  });
  return out;
}

function linkifyChunk(chunk: string, keyPrefix: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  URL_RE.lastIndex = 0;
  let k = 0;
  while ((m = URL_RE.exec(chunk)) !== null) {
    let url = m[1];
    const trail = url.match(/[.,;!?)\]]+$/);
    let suffix = "";
    if (trail) {
      suffix = trail[0];
      url = url.slice(0, -suffix.length);
    }
    if (m.index > last) {
      out.push(...citeify(chunk.slice(last, m.index), `${keyPrefix}-c${k++}`));
    }
    out.push(
      <a
        key={`${keyPrefix}-${k++}`}
        href={url}
        target="_blank"
        rel="noreferrer"
        className="rlink"
      >
        <span>{url}</span>
        <ExternalLinkIcon className="link-ext-icon" />
      </a>,
    );
    if (suffix) out.push(<React.Fragment key={`${keyPrefix}-${k++}`}>{suffix}</React.Fragment>);
    last = m.index + m[1].length;
  }
  if (last < chunk.length) out.push(...citeify(chunk.slice(last), `${keyPrefix}-c${k++}`));
  return out;
}

const NUM_RE = /^(\d+)[.)]\s+(.*)$/;
const BULLET_RE = /^[-*•–—]\s+(.*)$/;
const HEADING_RE = /^#{1,3}\s+(.*)$/;
const BOLD_LINE_RE = /^\*\*(.+?)\*\*\s*$/;
const DIVIDER_RE = /^(---|\*\*\*|___)\s*$/;
const QUOTE_RE = /^>\s?(.*)$/;

/** True when the user prefers reduced motion (typewriter + animations off). */
export function useReducedMotion(): boolean {
  const [reduce, setReduce] = useState(
    () =>
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduce(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduce;
}

/** Structured reply renderer: headings, numbered items, nested bullets, callouts, dividers, links. */
export function RichText({ text }: { text: string | null | undefined }) {
  const segments = splitCodeSegments(text ?? "");
  return (
    <div className="prose-container">
      {segments.map((seg, si) =>
        seg.code ? (
          <pre key={`c${si}`} className="rcode">
            <code>{seg.body.join("\n")}</code>
          </pre>
        ) : (
          <React.Fragment key={`p${si}`}>{renderProseLines(seg.body, `s${si}`)}</React.Fragment>
        ),
      )}
    </div>
  );
}

function splitCodeSegments(text: string): { code: boolean; body: string[] }[] {
  const segments: { code: boolean; body: string[] }[] = [];
  let cur: { code: boolean; body: string[] } = { code: false, body: [] };
  text.split("\n").forEach((ln) => {
    if (ln.trimStart().startsWith("```")) {
      segments.push(cur);
      cur = { code: !cur.code, body: [] };
      return;
    }
    cur.body.push(ln);
  });
  segments.push(cur);
  return segments.filter((s, i) => s.body.length > 0 || i === segments.length - 1);
}

function renderProseLines(lines: string[], keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let prevGap = true;

  lines.forEach((ln, i) => {
    const trimmed = ln.trim();
    if (trimmed === "") {
      if (!prevGap) {
        nodes.push(<div key={i} className="gap" aria-hidden="true" />);
        prevGap = true;
      }
      return;
    }
    prevGap = false;
    if (DIVIDER_RE.test(trimmed)) {
      nodes.push(<hr key={i} className="rdiv" />);
      prevGap = true;
      return;
    }
    const leading = ln.length - ln.trimStart().length;
    const lvl = Math.min(2, Math.floor(leading / 2));
    let m: RegExpMatchArray | null;

    if ((m = trimmed.match(HEADING_RE))) {
      nodes.push(<div key={i} className="line h">{renderInline(m[1], `h${i}`)}</div>);
      return;
    }
    if ((m = trimmed.match(BOLD_LINE_RE))) {
      nodes.push(<div key={i} className="line h">{renderInline(m[1], `h${i}`)}</div>);
      return;
    }
    if ((m = trimmed.match(QUOTE_RE))) {
      nodes.push(<div key={i} className="line quote">{renderInline(m[1], `q${i}`)}</div>);
      return;
    }
    if ((m = trimmed.match(NUM_RE))) {
      nodes.push(
        <div key={i} className={`line num lvl-${lvl}`}>
          <span className="n" aria-hidden="true">{m[1]}</span>
          <span className="num-body">{renderInline(m[2], `n${i}`)}</span>
        </div>,
      );
      return;
    }
    if ((m = trimmed.match(BULLET_RE))) {
      const isWarn = /^(warning|note)\s*:/i.test(m[1]);
      nodes.push(
        <div key={i} className={`line bullet lvl-${lvl}${isWarn ? " warn" : ""}`}>
          <span className="dot" aria-hidden="true">•</span>
          <span className="bullet-body">{renderInline(m[1], `b${i}`)}</span>
        </div>,
      );
      return;
    }
    if (/^(note|warning)\s*:/i.test(trimmed)) {
      nodes.push(<div key={i} className="line note">{renderInline(trimmed, `c${i}`)}</div>);
      return;
    }
    if (trimmed.length <= 90 && trimmed.endsWith(":")) {
      nodes.push(<div key={i} className="line label">{renderInline(trimmed, `l${i}`)}</div>);
      return;
    }
    nodes.push(<div key={i} className={`line lvl-${lvl}`}>{renderInline(ln.trim(), `p${i}`)}</div>);
  });

  return nodes;
}

/** Copy-to-clipboard button with transient confirmation. */
export function CopyButton({ text, label = "Copy response" }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);
  useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current);
  }, []);
  const handleCopy = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      setCopied(true);
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };
  return (
    <button
      type="button"
      className={`icon-btn${copied ? " copied" : ""}`}
      onClick={handleCopy}
      aria-label={label}
      title={copied ? "Copied" : label}
    >
      {copied ? <CheckIcon size={13} /> : <CopyIcon size={13} />}
    </button>
  );
}

/** Typewriter reveal for freshly arrived assistant text (full text when reduced motion). */
export function TypewriterText({ text }: { text: string }) {
  const reduce = useReducedMotion();
  const [n, setN] = useState(() => (reduce ? text.length : 0));
  useEffect(() => {
    if (reduce) {
      setN(text.length);
      return;
    }
    setN(0);
    if (!text) return;
    const step = Math.max(2, Math.ceil(text.length / 140));
    const t = window.setInterval(() => {
      setN((v) => {
        if (v >= text.length) {
          window.clearInterval(t);
          return v;
        }
        return Math.min(text.length, v + step);
      });
    }, 16);
    return () => window.clearInterval(t);
  }, [text, reduce]);
  return (
    <span className="typewriter">
      <RichText text={text.slice(0, n)} />
    </span>
  );
}

/** Tidy answer text. Numbered [n] citations stay; stray model markers go. */
export function cleanAnswerText(text: string): string {
  let s = text.replace(/\s*\[Sources?\s*[\d,\sand&]+\]/gi, "");
  s = s.replace(/\s*\[IS[^\]]*\]/g, "");
  s = s.replace(/\s*,\s*,/g, ",");
  s = s.replace(/\s*,\s*\./g, ".");
  s = s.replace(/\(\s*\)/g, "");
  s = s.replace(/[ \t]{2,}/g, " ");
  s = s.replace(/ +\n/g, "\n");
  return s.trim();
}

const LABEL_NAMES: Record<string, string> = {
  "BIS GUIDANCE PAGE": "BIS guidance",
  "COMPULSORY PRODUCT LIST": "Compulsory product list",
  "LAB DIRECTORY": "Lab directory",
  "STANDARD DOCUMENT EXCERPT": "Standard excerpt",
};

function sourceKind(s: RagSource): string {
  if (s.evidence_type === "catalogue_record" || s.metadata_only) return "Standards catalogue";
  const label = (s.label || "").toUpperCase();
  for (const [k, v] of Object.entries(LABEL_NAMES)) if (label.startsWith(k)) return v;
  if (s.doc_type === "product_manual") return "Product manual";
  if (s.doc_type === "gazette") return "Gazette notification";
  return "BIS document";
}

function sourceName(s: RagSource): string {
  return (s.standard_number || s.heading || s.title || "BIS document").trim();
}

function shortName(s: RagSource): string {
  const t = sourceName(s).replace(/\s+/g, " ");
  return t.length > 34 ? `${t.slice(0, 33)}…` : t;
}

function hostOf(url: string): string {
  try {
    return new URL(url).host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

const EXCERPT_PREVIEW = 320;

function SourceExcerpt({ text }: { text: string }) {
  const [full, setFull] = useState(false);
  const long = text.length > EXCERPT_PREVIEW;
  const shown = !long || full ? text : `${text.slice(0, EXCERPT_PREVIEW).trimEnd()}…`;
  return (
    <>
      <p className="src-page-excerpt">{shown}</p>
      {long && (
        <button type="button" className="src-expand" onClick={() => setFull((v) => !v)}>
          {full ? "Show less" : "Show more"}
        </button>
      )}
    </>
  );
}

/**
 * Numbered source chips under an answer plus a side panel with each source's
 * excerpt and official link. `focus` is the 1-based source to open (null = closed).
 */
export function SourceStrip({
  sources,
  focus,
  onFocus,
  related = false,
}: {
  sources?: RagSource[] | null;
  focus: number | null;
  onFocus: (n: number | null) => void;
  /** unnumbered "related documents" (answer could not be verified) */
  related?: boolean;
}) {
  const items = (sources ?? []).slice(0, 8);
  useEffect(() => {
    if (focus === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onFocus(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focus, onFocus]);
  const openRef = useRef<HTMLDetailsElement | null>(null);
  useEffect(() => {
    openRef.current?.scrollIntoView({ block: "nearest" });
  }, [focus]);

  if (items.length === 0) return null;
  const numberOf = (s: RagSource, i: number) => s.ref ?? i + 1;

  return (
    <>
      <div className="src-row" role="list" aria-label={related ? "Related documents" : "Sources"}>
        <span className="src-row-label">{related ? "Related" : "Sources"}</span>
        {items.map((s, i) => (
          <button
            key={`${sourceName(s)}-${i}`}
            type="button"
            role="listitem"
            className="src-chip"
            onClick={() => onFocus(numberOf(s, i))}
            title={`${sourceKind(s)}: ${sourceName(s)}`}
          >
            {!related && <span className="src-chip-n">{numberOf(s, i)}</span>}
            <span className="src-chip-text">{shortName(s)}</span>
          </button>
        ))}
      </div>
      {focus !== null && (
        <div className="src-page" role="dialog" aria-modal="true" aria-labelledby="src-page-title">
          <button type="button" className="src-page-scrim" onClick={() => onFocus(null)} aria-label="Close sources" />
          <div className="src-page-panel">
            <div className="src-page-top">
              <div id="src-page-title" className="src-page-std">
                {related ? "Related documents" : items.length === 1 ? "Source" : "Sources"}
              </div>
              <button type="button" className="icon-btn" onClick={() => onFocus(null)} aria-label="Close">
                <XIcon size={16} />
              </button>
            </div>
            <div className="src-acc-list">
              {items.map((s, i) => {
                const n = numberOf(s, i);
                const isOpen = n === focus;
                return (
                  <details
                    key={`${sourceName(s)}-${i}`}
                    className={`src-acc${isOpen ? " focused" : ""}`}
                    open={isOpen}
                    ref={isOpen ? openRef : undefined}
                  >
                    <summary className="src-acc-sum">
                      {!related && <span className="src-acc-n">{n}</span>}
                      <div className="src-acc-copy">
                        <span className="src-page-badge">{sourceKind(s)}</span>
                        <div className="src-card-std">{sourceName(s)}</div>
                        {s.title && s.title !== sourceName(s) && (
                          <div className="src-card-title">{s.title}</div>
                        )}
                      </div>
                      <ChevronDownIcon className="src-acc-chevron" size={14} />
                    </summary>
                    <div className="src-acc-body">
                      {s.chunk_text ? (
                        <SourceExcerpt text={s.chunk_text} />
                      ) : (
                        <p className="src-page-excerpt src-page-empty">
                          Catalogue entry: title and designation only. Open the official page for the full standard.
                        </p>
                      )}
                      {s.url && (
                        <a className="src-open" href={s.url} target="_blank" rel="noreferrer">
                          <span>Open on {hostOf(s.url) || "official site"}</span>
                          <ExternalLinkIcon className="link-ext-icon" />
                        </a>
                      )}
                    </div>
                  </details>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/** Answer prose (typewriter or static) with clickable [n] citations and its sources. */
export function AnswerBody({
  text,
  animate,
  sources,
  related,
}: {
  text: string;
  animate: boolean;
  sources?: RagSource[] | null;
  related?: RagSource[] | null;
}) {
  const [focus, setFocus] = useState<number | null>(null);
  const [relatedFocus, setRelatedFocus] = useState<number | null>(null);
  const cited = sources ?? [];
  const onCite = (n: number) => {
    if (n >= 1 && n <= cited.length) setFocus(n);
  };
  return (
    <CitationContext.Provider value={cited.length ? onCite : null}>
      <div className="answer-prose">
        {animate ? <TypewriterText text={text} /> : <RichText text={text} />}
      </div>
      <SourceStrip sources={cited} focus={focus} onFocus={setFocus} />
      <SourceStrip sources={related} focus={relatedFocus} onFocus={setRelatedFocus} related />
    </CitationContext.Provider>
  );
}

export interface StarterPrompt {
  who: string;
  text: string;
}

/** Example questions on the empty screen, one per kind of user. */
export function StarterPrompts({
  prompts,
  onPick,
  disabled,
}: {
  prompts: StarterPrompt[];
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="starter-grid" role="list" aria-label="Example questions">
      {prompts.map((p) => (
        <button
          key={p.text}
          type="button"
          role="listitem"
          className="starter-card"
          disabled={disabled}
          onClick={() => onPick(p.text)}
        >
          <span className="starter-who">{p.who}</span>
          <span className="starter-text">{p.text}</span>
        </button>
      ))}
    </div>
  );
}

/** Raw response JSON inspector — rendered only when developer mode is on. */
export function RawJson({ data, enabled }: { data: unknown; enabled: boolean }) {
  if (!enabled) return null;
  const [copied, setCopied] = useState(false);
  const timer = React.useRef<number | null>(null);
  React.useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current);
  }, []);
  const handleCopy = async () => {
    try {
      const text = JSON.stringify(data, null, 2);
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      setCopied(true);
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <details className="rawjson">
      <summary>
        <span className="rawjson-summary-title">Inspect Raw API Response</span>
        <button
          type="button"
          className="pill-btn pill-btn-sm"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleCopy();
          }}
          aria-label="Copy raw JSON payload"
        >
          {copied ? (
            <>
              <CheckIcon size={12} />
              <span>Copied</span>
            </>
          ) : (
            <>
              <CopyIcon size={12} />
              <span>Copy JSON</span>
            </>
          )}
        </button>
      </summary>
      <div className="expand-wrap">
        <pre className="rawjson-code">{JSON.stringify(data, null, 2)}</pre>
      </div>
    </details>
  );
}

export function FeedbackButtons({
  value,
  disabled,
  onRate,
}: {
  value: 1 | -1 | null | undefined;
  disabled?: boolean;
  onRate: (rating: 1 | -1) => void;
}) {
  const done = value === 1 || value === -1;
  const [gone, setGone] = useState(false);
  useEffect(() => {
    if (!done) {
      setGone(false);
      return;
    }
    const t = window.setTimeout(() => setGone(true), 750);
    return () => window.clearTimeout(t);
  }, [done, value]);
  if (gone) return null;
  return (
    <div className={`fb${done ? " fb-done" : ""}`} role="group" aria-label="Rate this answer">
      <button
        type="button"
        className={`icon-btn fb-btn${value === 1 ? " active-pos fb-win" : done ? " fb-gone" : ""}`}
        aria-label="Helpful answer"
        aria-pressed={value === 1}
        disabled={disabled || done}
        onClick={() => onRate(1)}
        title="Helpful compliance guidance"
        tabIndex={done && value !== 1 ? -1 : undefined}
      >
        <ThumbsUpIcon size={16} />
      </button>
      <button
        type="button"
        className={`icon-btn fb-btn${value === -1 ? " active-neg fb-win" : done ? " fb-gone" : ""}`}
        aria-label="Unhelpful answer"
        aria-pressed={value === -1}
        disabled={disabled || done}
        onClick={() => onRate(-1)}
        title="Unhelpful or inaccurate guidance"
        tabIndex={done && value !== -1 ? -1 : undefined}
      >
        <ThumbsDownIcon size={16} />
      </button>
    </div>
  );
}

/** Shimmer placeholder while the model answer streams in. */
export function SkeletonAnswer() {
  return (
    <div className="assistant-message-row" role="status">
      <div className="assistant-avatar sk-avatar" aria-hidden="true">
        <span />
      </div>
      <div className="assistant-content sk-lines" aria-hidden="true">
        <span className="sk-line sk-w90" />
        <span className="sk-line sk-w70" />
        <span className="sk-line sk-w80" />
      </div>
      <span className="sr-only">Waiting for the assistant&apos;s response</span>
    </div>
  );
}
