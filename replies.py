"""Customer reply drafts for reviews.

Without AI, a reply is filled in from a template using the themes detected in the
review. With AI (ai_service.draft_reply), the reply is written for the specific review.
Either way it is a draft for a person to check before sending.
"""

from __future__ import annotations

TONES = ("Friendly", "Professional", "Apologetic")

# How each theme is described in a reply ("We're sorry about ...").
THEME_PHRASES = {
    "Connectivity": "the connection problems",
    "Battery & Charging": "the battery and charging issues",
    "Shipping & Delivery": "the delivery experience",
    "Product Quality": "the build quality problems",
    "Customer Service": "the support you received",
    "Performance & Features": "the performance issues",
    "Comfort & Materials": "the comfort issues",
    "Price & Value": "the value for money",
    "Design & Usability": "how difficult it was to use",
}

OPENINGS = {
    "Friendly": "Hi there, thanks so much for taking the time to review the {product}!",
    "Professional": "Thank you for your review of the {product}.",
    "Apologetic": "Thank you for telling us about your experience with the {product}, and we're truly sorry it fell short.",
}
PROBLEM = {
    "Friendly": "We're sorry to hear about {issues}, that's not the experience we want you to have.",
    "Professional": "We regret to hear about {issues} and have shared your feedback with the product team.",
    "Apologetic": "We apologise for {issues}. You deserved better, and your feedback has gone straight to our product team.",
}
PRAISE = {
    "Friendly": "We're really glad you're enjoying {liked}!",
    "Professional": "We're pleased that {liked} met your expectations.",
    "Apologetic": "We're glad {liked} worked well for you.",
}
CLOSINGS = {
    "Friendly": "If you'd like a hand, just reach out to our support team and we'll help you sort it out.",
    "Professional": "Please contact our support team so we can look into this for you.",
    "Apologetic": "Please contact our support team so we can make this right for you.",
}
POSITIVE_CLOSINGS = {
    "Friendly": "Thanks again, and enjoy it!",
    "Professional": "Thank you for choosing us.",
    "Apologetic": "Thank you for your support.",
}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def template_reply(product: str, complaints: list[str], praise: list[str], tone: str = "Friendly",
                   signature: str = "") -> str:
    """Reply built from the detected complaint and praise themes."""
    tone = tone if tone in TONES else "Friendly"
    parts = [OPENINGS[tone].format(product=product)]
    if praise:
        parts.append(PRAISE[tone].format(liked=_join([t.split(" & ")[0].lower() for t in praise[:2]])))
    if complaints:
        issues = _join([THEME_PHRASES.get(t, f"the {t.lower()} issues") for t in complaints[:2]])
        parts.append(PROBLEM[tone].format(issues=issues))
        parts.append(CLOSINGS[tone])
    else:
        parts.append(POSITIVE_CLOSINGS[tone])
    return " ".join(parts) + f"\n\n{signature or f'The {product} team'}"
