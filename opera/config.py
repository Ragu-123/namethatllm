import re

PROVIDER_MAP = {
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot",
    "qwen3.8-max": "Alibaba",
}

CLASSES = sorted(list(PROVIDER_MAP.keys()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

PROVIDERS = sorted(list(set(PROVIDER_MAP.values())))
PROV_TO_IDX = {p: i for i, p in enumerate(PROVIDERS)}

PUNCT_SKELETON_SET = set("—–-;:,!?()[]“”‘’\"'….\n")
TOKEN_RE = re.compile(r"(?u)\b\w+\b|[^\w\s]")
SYLLABLE_RE = re.compile(r"[aeiouyAEIOUY]+")
MAJOR_PUNCT = {"—", "–", ";", ":", ",", "(", ")"}
FUNCTION_WORDS = [
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "as",
    "with", "on", "be", "this", "by", "are", "from", "or", "an", "at",
    "not", "but", "have", "which", "they", "you", "one", "all", "their",
    "can", "her", "has", "there", "been", "if", "more", "when", "will"
]
