export type Lang = "auto" | "en" | "hi";

export interface ThreadCtx {
  history: string[];
  rounds: number;
  force?: boolean;
}

export interface QuestionOpt {
  label: string;
  send: string;
}

export interface Question {
  slot: string;
  text: string;
  options: QuestionOpt[];
}

export interface StructuredCitation {
  is_number: string;
  year: string;
  status: string;
  last_checked: string;
  source_url: string;
  display: string;
}

export interface ChatResponse {
  text: string;
  refused: boolean;
  kind: string;
  lang: "en" | "hi";
  citations: string[];
  pii: Record<string, boolean>;
  needs_info: boolean;
  questions: Question[];
  known: { slot: string; value: string }[];
  assumptions: string[];
  context: ThreadCtx;
  /** Machine-readable citation rows (chat.py facade); absent on legacy responses. */
  structured_citations?: StructuredCitation[];
  thread_id?: string;
  owner_token?: string;
  /** Sources the answer cites, in citation order: `[n]` in text = sources[n-1]. */
  sources?: RagSource[];
  rag_evidence?: RagSource[];
  /** Unnumbered related documents shown when an answer could not be verified. */
  related_sources?: RagSource[];
  /** English search query used when the question was rewritten (Hindi, follow-ups). */
  search_query?: string;
  /** true for temporary states (model_busy) where resending should work. */
  retryable?: boolean;
  rag_mode?: string;
  rag_used_llm?: boolean;
  model_available?: boolean;
  /** NLU intent for this turn (nlu.classify). */
  intent?: string;
  intent_confidence?: string;
  /** Reserved for compatibility; recent user turns are sent to the model. */
  context_summary?: string;
  guidance_adaptive?: boolean;
}

export interface RagSource {
  standard_number: string;
  title: string;
  url: string;
  doc_type: string;
  heading?: string;
  chunk_text?: string;
  chunk_index?: number;
  source_file?: string;
  score: number;
  /** citation number used in the answer text */
  ref?: number;
  /** evidence type label, e.g. "COMPULSORY PRODUCT LIST" */
  label?: string;
  evidence_type?: "document_chunk" | "catalogue_record";
  metadata_only?: boolean;
  department?: string;
  date?: string;
}

export interface FeedbackPayload {
  thread_id: string;
  rating: 1 | -1;
  note?: string;
}

export interface FeedbackResult {
  ok: boolean;
  /** true when the live backend lacks POST /feedback and the UI used the fixture fallback */
  fixture?: boolean;
  /** set when the call itself failed — show it instead of thanking */
  error?: string;
}

export interface Msg {
  id: number;
  role: "user" | "assistant";
  text: string;
  resp?: ChatResponse;
  error?: string;
  ms?: number;
  system?: boolean;
  /** per-answer feedback state */
  feedback?: 1 | -1 | null;
  /** exact query that produced this message — powers per-message retry */
  retryQ?: string;
  /** true while a retry is re-running inside this message slot */
  retrying?: boolean;
  /** typewriter only for text that just arrived from the model */
  streamIn?: boolean;
}
