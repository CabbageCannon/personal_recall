def normalize_text(text: str) -> str:
    return " ".join(text.split())

def evidence_matches_source(
    evidence: str,
    source_content: str,
) -> bool:
    evidence = normalize_text(evidence)
    source_content = normalize_text(source_content)

    # 1. Strict full-line match
    if evidence in source_content:
        return True

    # 2. Boundary-tolerant match:
    # remove the timestamp prefix, but keep "speaker: message"
    if "] " in evidence:
        _, message_part = evidence.split("] ", 1)

        if message_part in source_content:
            return True

    return False
  
def strict_evidence_matches_source(
    evidence: str,
    source_content: str,
) -> bool:
    evidence = normalize_text(evidence)
    source_content = normalize_text(source_content)

    return evidence in source_content  
