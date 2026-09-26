"""Required context slots per IS: what must be known before naming a standard.

Each slot: key, q_en, q_hi, options[{label_en, label_hi, words[]}], and/or
pattern (regex on raw text, for numeric answers like capacity/grade).
"""
from __future__ import annotations
import re



def _tokens(s: str) -> set[str]:
    """Lower-case word tokens plus singular forms (bulbs -> bulb)."""
    out: set[str] = set()
    for t in re.findall(r"[a-z0-9\u0900-\u097F]+", s.lower()):
        out.add(t)
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            out.add(t[:-1])
    return out


_DB_SLOTS: dict[str, list[dict]] | None = None  # Phase 1: DB-backed override


def use_db(path) -> None:
    """Switch slot source to the SQLite KB (KB_BACKEND=sqlite)."""
    global _DB_SLOTS
    from . import kb_store
    conn = kb_store.connect(path)
    try:
        _DB_SLOTS = kb_store.load_slots(conn)
    finally:
        conn.close()


def active_slots() -> dict[str, list[dict]]:
    """Public accessor for the slot table (DB override when set, else SLOTS)."""
    return _DB_SLOTS if _DB_SLOTS is not None else SLOTS


def _active() -> dict[str, list[dict]]:
    return active_slots()

SLOTS: dict[str, list[dict]] = {
    "IS 10500": [
        {"key": "water_source",
         "q_en": "Is this piped/tap water quality, or packaged water for sale?",
         "q_hi": "Kya yah nal/supply ke paani ki gunvatta hai ya bechne wale packaged paani ki?",
         "options": [
             {"label_en": "Piped/tap water", "label_hi": "Nal ka paani",
              "words": ["piped", "pipe", "tap", "municipal", "supply", "nal", "नल"]},
             {"label_en": "Packaged water for sale", "label_hi": "Bechne wala packaged paani",
              "words": ["packaged", "packaging", "bottled", "jar", "plant", "sale", "sell", "bisleri"]}]},
        {"key": "purpose",
         "q_en": "Is it for household supply or a commercial plant/licence?",
         "q_hi": "Gharelu aapurti ke liye hai ya commercial plant/licence ke liye?",
         "options": [
             {"label_en": "Household supply", "label_hi": "Gharelu aapurti",
              "words": ["household", "home", "domestic", "ghar", "घर"]},
             {"label_en": "Commercial plant", "label_hi": "Commercial plant",
              "words": ["commercial", "plant", "factory", "licence", "license", "udyog"]}]},
    ],
    "IS 14543": [
        {"key": "pack_size",
         "q_en": "What packing sizes (e.g. 1 L bottle, 20 L jar)?",
         "q_hi": "Packing size kya hai (jaise 1 L bottle, 20 L jar)?",
         "pattern": r"\d+\s*(ml|l\b|litre|liter|ltr|jar|bottle)",
         "options": []},
        {"key": "plant_stage",
         "q_en": "Is this a new plant licence or an existing renewal?",
         "q_hi": "Kya yah naye plant ka licence hai ya renewal?",
         "options": [
             {"label_en": "New plant", "label_hi": "Naya plant",
              "words": ["new", "start", "setup", "naya", "plant lagana"]},
             {"label_en": "Renewal", "label_hi": "Renewal",
              "words": ["renewal", "renew", "existing", "navinikaran"]}]},
    ],
    "IS 13428": [
        {"key": "source",
         "q_en": "Is the source claimed as natural spring/mineral water?",
         "q_hi": "Kya srot prakritik spring/mineral water hai?",
         "options": [
             {"label_en": "Natural spring/mineral", "label_hi": "Prakritik spring/mineral",
              "words": ["spring", "natural", "mineral", "prakritik", "borewell"]},
             {"label_en": "Treated water", "label_hi": "Treated paani",
              "words": ["treated", "ro", "purified", "shodhit"]}]},
    ],
    "IS 17803": [
        {"key": "type",
         "q_en": "Stainless steel single-wall or vacuum insulated double-wall?",
         "q_hi": "Single-wall steel ya vacuum insulated double-wall?",
         "options": [
             {"label_en": "Vacuum insulated (double-wall)", "label_hi": "Vacuum insulated",
              "words": ["vacuum", "insulated", "double", "thermo", "flask"]},
             {"label_en": "Single-wall", "label_hi": "Single-wall",
              "words": ["single"]}]},
        {"key": "capacity",
         "q_en": "What capacity in ml/L, and is it for household food contact?",
         "q_hi": "Kitni capacity (ml/L), aur kya gharelu khadya sampark ke liye?",
         "pattern": r"\d+\s*(ml|l\b|litre|liter|ltr)",
         "options": []},
    ],
    "IS 694": [
        {"key": "voltage",
         "q_en": "What voltage grade (e.g. up to 1100 V)?",
         "q_hi": "Voltage grade kya hai (jaise 1100 V tak)?",
         "pattern": r"\d+\s*(v\b|volt|kv)",
         "options": []},
        {"key": "kind",
         "q_en": "Single-core or multi-core, sheathed or unsheathed? House wiring or industrial?",
         "q_hi": "Single-core ya multi-core? Ghar wiring ya industrial?",
         "options": [
             {"label_en": "House wiring", "label_hi": "Ghar wiring",
              "words": ["house", "home", "wiring", "ghar", "घर"]},
             {"label_en": "Industrial", "label_hi": "Industrial",
              "words": ["industrial", "factory", "udyog"]}]},
    ],
    "IS 1293": [
        {"key": "rating",
         "q_en": "What rating — 6 A, 16 A or other?",
         "q_hi": "Rating kya hai — 6 A, 16 A ya anya?",
         "pattern": r"\d+\s*(a\b|amp)",
         "options": []},
        {"key": "kind",
         "q_en": "Plug, socket-outlet, or adaptor?",
         "q_hi": "Plug, socket ya adaptor?",
         "options": [
             {"label_en": "Plug", "label_hi": "Plug", "words": ["plug"]},
             {"label_en": "Socket", "label_hi": "Socket", "words": ["socket"]},
             {"label_en": "Adaptor", "label_hi": "Adaptor", "words": ["adaptor", "adapter"]}]},
    ],
    "IS 302-1": [
        {"key": "appliance",
         "q_en": "Which appliance exactly (mixer, iron, geyser, cooler…)?",
         "q_hi": "Kaun sa upkaran (mixer, iron, geyser…)?",
         "options": [
             {"label_en": "Mixer/grinder", "label_hi": "Mixer", "words": ["mixer", "grinder"]},
             {"label_en": "Iron/geyser/other", "label_hi": "Iron/geyser/anya",
              "words": ["iron", "geyser", "microwave", "cooler", "heater", "istri"]}]},
        {"key": "use",
         "q_en": "Household or commercial use?",
         "q_hi": "Gharelu ya commercial upyog?",
         "options": [
             {"label_en": "Household", "label_hi": "Gharelu",
              "words": ["household", "home", "domestic", "ghar"]},
             {"label_en": "Commercial", "label_hi": "Commercial",
              "words": ["commercial", "hotel", "shop", "factory"]}]},
    ],
    "IS 16102-1": [
        {"key": "lamp_kind",
         "q_en": "Self-ballasted retrofit bulb or a luminaire/fixture?",
         "q_hi": "Retrofit bulb ya luminaire/fixture?",
         "options": [
             {"label_en": "Retrofit bulb", "label_hi": "Retrofit bulb",
              "words": ["bulb", "retrofit", "self-ballasted", "self", "ballasted"]},
             {"label_en": "Luminaire/fixture", "label_hi": "Luminaire",
              "words": ["luminaire", "fixture", "tubelight fitting", "panel"]}]},
        {"key": "wattage",
         "q_en": "Wattage and cap type (e.g. 9 W, B22/E27)?",
         "q_hi": "Wattage aur cap type (jaise 9 W, B22/E27)?",
         "pattern": r"\d+\s*w\b",
         "options": []},
    ],
    "IS 9873-1": [
        {"key": "age",
         "q_en": "What age group is the toy for?",
         "q_hi": "Khilauna kis age group ke liye hai?",
         "pattern": r"\d+\s*(year|yr|month|saal|varsh)|infant|toddler",
         "options": []},
        {"key": "kind",
         "q_en": "Mechanical, soft, electric, or painted/coated toy?",
         "q_hi": "Mechanical, soft, electric ya paint wala khilauna?",
         "options": [
             {"label_en": "Soft/mechanical", "label_hi": "Soft/mechanical",
              "words": ["soft", "mechanical", "plastic", "wooden"]},
             {"label_en": "Electric/painted", "label_hi": "Electric/paint wala",
              "words": ["electric", "electronic", "battery", "paint", "coating"]}]},
    ],
    "IS 269": [
        {"key": "grade",
         "q_en": "Which grade/type — OPC 33/43/53, PPC or PSC?",
         "q_hi": "Kaun sa grade/type — OPC 33/43/53, PPC ya PSC?",
         "pattern": r"\b(33|43|53)\b|opc|ppc|psc",
         "options": []},
    ],
    "IS 1786": [
        {"key": "grade",
         "q_en": "Which grade — Fe415, Fe500, Fe550 or other? Diameter range?",
         "q_hi": "Kaun sa grade — Fe415, Fe500, Fe550? Diameter?",
         "pattern": r"fe\s?\d+|\b(415|500|550|600)\b|\d+\s*mm",
         "options": []},
    ],
    "IS 4985": [
        {"key": "application",
         "q_en": "Potable (drinking) water or non-potable use? Pressure or non-pressure?",
         "q_hi": "Peene ke paani ke liye ya anya upyog? Pressure ya non-pressure?",
         "options": [
             {"label_en": "Potable water", "label_hi": "Peene ka paani",
              "words": ["potable", "drinking", "peene", "pani", "paani", "पानी"]},
             {"label_en": "Non-potable / agriculture", "label_hi": "Non-potable/kheti",
              "words": ["agriculture", "irrigation", "drainage", "kheti", "sinchai"]}]},
        {"key": "diameter",
         "q_en": "What diameter (mm)?",
         "q_hi": "Diameter kitna (mm)?",
         "pattern": r"\d+\s*mm",
         "options": []},
    ],
    "IS 4151": [
        {"key": "vehicle",
         "q_en": "Two-wheeler rider helmet? What size range?",
         "q_hi": "Two-wheeler rider helmet? Size?",
         "pattern": r"\b(xs|s\b|m\b|l\b|xl|xxl)\b|\d+\s*cm",
         "options": [
             {"label_en": "Two-wheeler", "label_hi": "Two-wheeler",
              "words": ["two-wheeler", "bike", "scooter", "rider", "helmet"]}]},
    ],
    "IS 2347": [
        {"key": "capacity",
         "q_en": "What capacity (litres) — and aluminium or steel?",
         "q_hi": "Kitne litre — aluminium ya steel?",
         "pattern": r"\d+\s*(l\b|litre|liter|ltr)",
         "options": [
             {"label_en": "Aluminium", "label_hi": "Aluminium", "words": ["aluminium", "aluminum"]},
             {"label_en": "Steel", "label_hi": "Steel", "words": ["steel", "stainless"]}]},
    ],
    "IS 2553-1": [
        {"key": "use",
         "q_en": "Automobile, building, or general use? Toughened or laminated?",
         "q_hi": "Automobile, building ya samanya upyog? Toughened ya laminated?",
         "options": [
             {"label_en": "Automobile", "label_hi": "Automobile",
              "words": ["auto", "car", "windshield", "vehicle", "gadi"]},
             {"label_en": "Building/general", "label_hi": "Building",
              "words": ["building", "window", "door", "flat"]}]},
    ],
}


def _word_hit(word: str, toks: set[str], low: str) -> bool:
    """Token-boundary option matching (issue #4 P0-3).

    Single tokens match the token set only — raw ``w in low`` substring let
    ``iron`` fill ``ro`` (and ``from``/``error``-style collisions). Phrases
    (spaces/hyphens) match with boundary guards so ``tubelight fitting``
    still fills without ``fit``-style prefixes leaking in.
    """
    w = (word or "").lower()
    if not w:
        return False
    if " " in w or "-" in w:
        return re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", low) is not None
    return w in toks


def fills_for(is_number: str, text: str) -> dict[str, dict]:
    """Return {slot_key: {label_en, label_hi, matched}} for filled slots."""
    toks = _tokens(text)
    low = text.lower()
    out: dict[str, dict] = {}
    for slot in _active().get(is_number, []):
        pat = slot.get("pattern")
        if pat:
            m = re.search(pat, low)
            if m:
                out[slot["key"]] = {"label_en": f"specified ({m.group(0).strip()})",
                                    "label_hi": f"diya gaya ({m.group(0).strip()})",
                                    "matched": m.group(0).strip()}
                continue
        for opt in slot.get("options", []):
            words = [w.lower() for w in opt.get("words", [])]
            if any(_word_hit(w, toks, low) for w in words):
                out[slot["key"]] = {"label_en": opt["label_en"], "label_hi": opt["label_hi"],
                                    "matched": opt["label_en"]}
                break
    return out


def unfilled_slots(is_number: str, text: str) -> list[dict]:
    filled = fills_for(is_number, text)
    return [s for s in _active().get(is_number, []) if s["key"] not in filled]
