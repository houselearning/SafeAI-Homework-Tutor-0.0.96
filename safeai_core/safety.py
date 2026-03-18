# safeai_core/safety.py
FORBIDDEN = [
    "write my entire assignment",
    "cheat on",
    "plagiarize",
    "bypass plagiarism",
    "exam answers",
]

SYSTEM_PREFIX = (
    "You are SafeAI Homework Tutor 0.0.96. "
    "You help students understand and practice. "
    "You refuse to do graded work, cheat, or enable plagiarism."
)

def check_request(msg: str) -> str | None:
    lower = msg.lower()
    if any(p in lower for p in FORBIDDEN):
        return (
            "I can’t do that. I can help you understand the material, outline ideas, "
            "or review your own work instead."
        )
    return None
