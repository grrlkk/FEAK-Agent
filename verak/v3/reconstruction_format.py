"""Fixed first-sentence extraction, never quality-based selection or retries."""
import re


def first_sentence(text):
    text=text.strip()
    if not text:
        raise ValueError('Expected a nonempty reconstruction')
    # Formatting may contain line wrapping. Preserve words, normalize only that
    # whitespace; use the first completed sentence rather than the best sentence.
    normalized=re.sub(r'\s*\n\s*',' ',text)
    ending=re.search(r'[.!?。！？][\"\'”’）)\]]*(?=\s|$)',normalized)
    candidate=normalized[:ending.end()] if ending else normalized
    if '[MISSING_SENTENCE]' in candidate:
        raise ValueError('Gap marker is not a reconstruction')
    return candidate
