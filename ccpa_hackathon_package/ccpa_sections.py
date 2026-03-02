"""
ccpa_sections.py — Whitelist of valid CCPA section identifiers.

Only these sections may appear in the 'articles' field of the API response.
This prevents the LLM from hallucinating non-existent CCPA section numbers.
"""

# Canonical valid CCPA Civil Code sections
VALID_SECTIONS: set[str] = {
    "Section 1798.100",
    "Section 1798.105",
    "Section 1798.110",
    "Section 1798.115",
    "Section 1798.120",
    "Section 1798.125",
    "Section 1798.130",
    "Section 1798.135",
    "Section 1798.140",
    "Section 1798.145",
    "Section 1798.150",
    "Section 1798.155",
    "Section 1798.160",
    "Section 1798.175",
    "Section 1798.180",
    "Section 1798.185",
    "Section 1798.190",
    "Section 1798.192",
    "Section 1798.194",
    "Section 1798.196",
    "Section 1798.198",
    "Section 1798.199",
}

# Map of section number to short description (used in prompt construction)
SECTION_DESCRIPTIONS: dict[str, str] = {
    "Section 1798.100": "Right to know what personal information is collected",
    "Section 1798.105": "Right to deletion of personal information",
    "Section 1798.110": "Right to know about data collection and use",
    "Section 1798.115": "Right to know about data disclosure and sale",
    "Section 1798.120": "Right to opt-out of sale of personal information",
    "Section 1798.125": "Non-discrimination for exercising consumer rights",
    "Section 1798.130": "Notice and response requirements for businesses",
    "Section 1798.135": "Do Not Sell link requirement",
    "Section 1798.140": "Definitions",
    "Section 1798.145": "Exemptions",
    "Section 1798.150": "Personal information security breaches",
    "Section 1798.155": "Attorney General enforcement",
    "Section 1798.160": "Consumer Privacy Fund",
    "Section 1798.175": "Conflicting provisions",
    "Section 1798.180": "Preemption",
    "Section 1798.185": "Regulations",
    "Section 1798.190": "Anti-waiver provision",
    "Section 1798.192": "No expansion of legal duties",
    "Section 1798.194": "Liberal construction",
    "Section 1798.196": "Severability",
    "Section 1798.198": "Operative date provisions",
    "Section 1798.199": "Additional provisions",
}


def normalize_section(raw: str) -> str | None:
    """
    Attempt to normalize a raw section string to a canonical CCPA section.
    Returns the canonical string if valid, or None if not recognized.

    Handles formats like:
      - "Section 1798.100"
      - "1798.100"
      - "Sec 1798.100"
      - "Sec. 1798.100"
      - "Article 1798.100"
      - "CCPA Section 1798.100"
    """
    import re

    # Extract the section number (e.g., 1798.100)
    match = re.search(r"1798\.\d{2,3}[a-z]?", raw)
    if not match:
        return None

    section_num = match.group(0)
    canonical = f"Section {section_num}"

    if canonical in VALID_SECTIONS:
        return canonical
    return None


def filter_valid_sections(sections: list[str]) -> list[str]:
    """
    Filter and normalize a list of section strings.
    Returns only valid, deduplicated, canonical section strings.
    """
    seen: set[str] = set()
    result: list[str] = []

    for raw in sections:
        normalized = normalize_section(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result
