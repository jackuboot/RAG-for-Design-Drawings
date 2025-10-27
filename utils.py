import re, yaml

def load_config(path: str="config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

SHEET_ID_RE = re.compile(r"\b([A-Z]{1,3}\.?[0-9]{1,2}(?:\.[0-9]{1,2})?)\b")

SECTION_KEYWORDS = {
    "window schedule": ["WINDOW SCHEDULE", "WINDOW SCHED", "WINDOWS SCHEDULE"],
    "door schedule": ["DOOR SCHEDULE", "DOOR SCHED"],
    "title block": ["TITLE BLOCK", "PROJECT", "DRAWING LIST", "REVISION"],
    "general notes": ["GENERAL NOTES", "NOTES", "LEGEND", "ABBREVIATIONS"],
    "elevations": ["ELEVATION", "FRONT ELEVATION", "REAR ELEVATION", "LEFT ELEVATION", "RIGHT ELEVATION"],
    "floor plan": ["FLOOR PLAN", "FIRST FLOOR", "SECOND FLOOR", "PLAN"],
    "roof plan": ["ROOF PLAN"],
    "sections": ["SECTION", "BUILDING SECTION"],
    "schedules": ["SCHEDULE"]  # broad catch all
}

def guess_section(text: str) -> str:
    up = (text[:1500] if text else "").upper()
    for section, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if k in up:
                return section
    return "body"

def find_sheet_id(text: str) -> str:
    matches = SHEET_ID_RE.findall(text)
    if not matches:
        return ""
    matches = sorted(matches, key=lambda s: (len(s), s))
    return matches[0]

def bbox_to_str(b):
    if not b:
        return ""
    x0,y0,x1,y1 = b
    return f"{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}"
