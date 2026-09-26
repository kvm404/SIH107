/**
 * UI-MS contract test (plan §9): validates ui/tests/fixtures/*.json against the
 * shapes ui/src consumes (types.ts + api.ts). Recorded fixtures, no live backend.
 *
 * Run: npm run test:contract
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isHttpUrl, redactPii } from "../src/redact.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "fixtures");
const files = readdirSync(root).filter((f) => f.endsWith(".json"));
const load = (f) => JSON.parse(readFileSync(join(root, f), "utf8"));

function assertChatResponse(resp, label) {
  assert.equal(typeof resp.text, "string", `${label}: text must be string`);
  assert.equal(typeof resp.refused, "boolean", `${label}: refused must be boolean`);
  assert.equal(typeof resp.kind, "string", `${label}: kind must be string`);
  assert.ok(["en", "hi"].includes(resp.lang), `${label}: lang must be en|hi, got ${resp.lang}`);
  assert.ok(Array.isArray(resp.citations), `${label}: citations must be array`);
  for (const c of resp.citations) assert.equal(typeof c, "string", `${label}: citation must be string`);
  assert.equal(typeof resp.pii, "object", `${label}: pii must be object`);
  assert.equal(typeof resp.needs_info, "boolean", `${label}: needs_info must be boolean`);
  assert.ok(Array.isArray(resp.questions), `${label}: questions must be array`);
  for (const q of resp.questions) {
    assert.equal(typeof q.slot, "string", `${label}: question.slot must be string`);
    assert.equal(typeof q.text, "string", `${label}: question.text must be string`);
    assert.ok(Array.isArray(q.options), `${label}: question.options must be array`);
    for (const o of q.options) {
      assert.equal(typeof o.label, "string", `${label}: option.label must be string`);
      assert.equal(typeof o.send, "string", `${label}: option.send must be string`);
    }
  }
  assert.ok(Array.isArray(resp.known), `${label}: known must be array`);
  for (const k of resp.known) {
    assert.equal(typeof k.slot, "string", `${label}: known.slot must be string`);
    assert.equal(typeof k.value, "string", `${label}: known.value must be string`);
  }
  assert.ok(Array.isArray(resp.assumptions), `${label}: assumptions must be array`);
  for (const a of resp.assumptions) assert.equal(typeof a, "string", `${label}: assumption must be string`);
  // Thread continuation fields the UI relies on (server.py mints thread_id + owner_token).
  if (resp.thread_id !== undefined) assert.equal(typeof resp.thread_id, "string", `${label}: thread_id must be string`);
  if (resp.owner_token !== undefined) assert.equal(typeof resp.owner_token, "string", `${label}: owner_token must be string`);
  // Answers are LLM-written; clarification questions and slot chips are gone.
  assert.equal(resp.needs_info, false, `${label}: needs_info is no longer produced`);
  assert.equal(resp.questions.length, 0, `${label}: questions are no longer produced`);
  // Citation contract: every [n] in the text points at sources[n-1], and
  // sources are exactly the cited ones, numbered 1..k in order.
  const refs = [...resp.text.matchAll(/\[(\d{1,2})\]/g)].map((m) => Number(m[1]));
  const sources = resp.sources ?? [];
  sources.forEach((s, i) => assert.equal(s.ref, i + 1, `${label}: sources must be numbered 1..k`));
  for (const n of refs) {
    assert.ok(n >= 1 && n <= sources.length, `${label}: [${n}] has no matching source`);
  }
  for (const s of sources) assert.ok(refs.includes(s.ref), `${label}: source ${s.ref} is never cited`);
  assert.doesNotMatch(resp.text, /\[Source\s*\d/i, `${label}: raw [Source N] markers must be renumbered`);
  if (resp.related_sources !== undefined) {
    assert.ok(Array.isArray(resp.related_sources), `${label}: related_sources must be array`);
    for (const s of resp.related_sources) assert.equal(s.ref, undefined, `${label}: related sources are unnumbered`);
  }
}

describe("ui contract fixtures", () => {
  it("fixture set is complete", () => {
    for (const expected of [
      "chat-answered.json",
      "chat-hindi.json",
      "chat-lab.json",
      "chat-refused.json",
      "chat-busy.json",
      "feedback.json",
    ]) {
      assert.ok(files.includes(expected), `missing fixture ${expected}`);
    }
  });

  it("chat fixtures match the ChatResponse shape ui/src consumes", () => {
    for (const f of files.filter((x) => x.startsWith("chat-"))) {
      const j = load(f);
      assertChatResponse(j.response, f);
    }
  });

  it("answers cite official sources", () => {
    for (const f of ["chat-answered.json", "chat-hindi.json", "chat-lab.json"]) {
      const r = load(f).response;
      assert.equal(r.kind, "llm_answer", f);
      assert.ok(r.sources.length > 0, `${f}: answers must cite sources`);
      for (const s of r.sources) assert.ok(isHttpUrl(s.url), `${f}: source url must be http(s)`);
    }
    const lab = load("chat-lab.json").response;
    assert.ok(lab.sources.some((s) => s.doc_type === "lab_directory"), "lab answer cites LIMS labs");
  });

  it("hindi fixture replies in Devanagari after an English search rewrite", () => {
    const r = load("chat-hindi.json").response;
    assert.equal(r.lang, "hi");
    assert.match(r.text, /[\u0900-\u097F]/);
    assert.match(r.search_query, /^[\x00-\x7F]+$/, "search query is English");
  });

  it("refusal and busy states carry no cited sources", () => {
    const refused = load("chat-refused.json").response;
    assert.equal(refused.kind, "grounding_refusal");
    assert.equal(refused.sources.length, 0);
    assert.ok(refused.related_sources.length > 0, "refusal offers related documents");
    const busy = load("chat-busy.json").response;
    assert.equal(busy.kind, "model_busy");
    assert.equal(busy.retryable, true);
  });

  it("RAG fields are validated when present", () => {
    for (const f of files.filter((x) => x.startsWith("chat-"))) {
      const resp = load(f).response;
      if (resp.sources !== undefined) {
        assert.ok(Array.isArray(resp.sources), `${f}: sources must be array`);
        for (const s of resp.sources) {
          for (const k of ["standard_number", "title", "url", "doc_type"]) {
            assert.equal(typeof s[k], "string", `${f}: source.${k} must be string`);
          }
          assert.equal(typeof s.score, "number", `${f}: source.score must be number`);
        }
      }
      if (resp.rag_mode !== undefined) assert.equal(typeof resp.rag_mode, "string");
      if (resp.rag_used_llm !== undefined) assert.equal(typeof resp.rag_used_llm, "boolean");
      if (resp.intent !== undefined) assert.equal(typeof resp.intent, "string");
      if (resp.intent_confidence !== undefined) {
        assert.ok(["high", "medium", "low"].includes(resp.intent_confidence));
      }
      if (resp.guidance_adaptive !== undefined) assert.equal(typeof resp.guidance_adaptive, "boolean");
    }
  });

  it("feedback fixture matches POST /feedback contract", () => {
    const j = load("feedback.json");
    assert.equal(j.response.ok, true);
    assert.equal(j.response.status, "pending");
    assert.deepEqual(j.contract.rating_enum, [1, -1]);
    assert.equal(j.contract.note_max_length, 1000);
    for (const ex of j.request_examples) {
      assert.equal(typeof ex.thread_id, "string");
      assert.ok([1, -1].includes(ex.rating), "rating must be 1|-1");
      if (ex.note !== undefined) assert.ok(ex.note.length <= j.contract.note_max_length);
    }
  });

  it("local transcript redact matches server PII rules", () => {
    const raw = "call me 9876543210 mail a@b.com aadhaar 1234 5678 9012";
    const out = redactPii(raw);
    assert.match(out, /REDACTED/);
    assert.doesNotMatch(out, /9876543210/);
    assert.doesNotMatch(out, /a@b\.com/);
    assert.ok(isHttpUrl("https://www.bis.gov.in/x"));
    assert.equal(isHttpUrl("javascript:alert(1)"), false);
  });
});
