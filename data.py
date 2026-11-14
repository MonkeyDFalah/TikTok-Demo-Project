"""Deterministic, fully synthetic data for a moderation quality work sample."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
POLICIES = {
    "allow": [
        "A creator welcomes viewers and asks what music they enjoy.",
        "The host shares a recipe and invites the audience to try it.",
        "A viewer asks for the name of the book shown on screen.",
        "The creator explains a beginner stretching routine.",
        "A guest describes how they organize a community event.",
        "The host answers questions about a public museum exhibit.",
        "A viewer compliments the performance and requests another song.",
        "The creator discusses a new phone case and its colors.",
        "The host reminds viewers to take breaks and drink water.",
        "A guest shares a general opinion about a recent film.",
    ],
    "scam": [
        "A message promises a prize if viewers send a small payment first.",
        "The host asks viewers to share a password to verify a giveaway.",
        "A comment links to a fake support page to recover an account.",
        "The speaker guarantees a large return after an urgent crypto transfer.",
        "A viewer is told to pay a fee to unlock a reward that does not exist.",
        "The host impersonates platform support and requests a login code.",
        "A message claims a limited offer expires unless money is sent now.",
        "The creator promotes an unverified investment with guaranteed profits.",
        "A comment requests bank details to deliver a surprise gift.",
        "The speaker directs viewers to a lookalike page to claim a refund.",
    ],
    "harassment": [
        "A viewer repeatedly targets another person with degrading insults.",
        "The speaker tells a named guest that they are worthless and unwelcome.",
        "A comment encourages the audience to pile on one specific creator.",
        "The host mocks a viewer with personal attacks after a disagreement.",
        "A message tells one person that everyone should ridicule them.",
        "A viewer directs repeated hostile name-calling at another participant.",
        "The speaker singles out a guest and urges others to shame them.",
        "A comment attacks a creator personally instead of discussing the topic.",
        "The host encourages viewers to send cruel messages to one person.",
        "A participant repeatedly insults another user during the discussion.",
    ],
    "dangerous_activity": [
        "The host urges viewers to copy an unsafe stunt near moving traffic.",
        "A speaker encourages an untrained person to attempt a risky rooftop jump.",
        "The creator demonstrates a hazardous challenge without safety precautions.",
        "A viewer is told to mix household chemicals for a dangerous reaction.",
        "The host pressures a participant to try an unsafe high-speed stunt.",
        "A message promotes a risky challenge likely to cause serious injury.",
        "The speaker shows how to bypass safety equipment during a stunt.",
        "A creator urges children to imitate a dangerous rooftop activity.",
        "The host encourages an unsafe fire trick in a crowded room.",
        "A participant is challenged to perform a hazardous act for attention.",
    ],
    "sexual_solicitation": [
        "A viewer pressures another adult to move to a private sexual chat.",
        "The host requests intimate images from a viewer in exchange for access.",
        "A comment repeatedly solicits sexual contact after being declined.",
        "The speaker offers money for explicit private content from a participant.",
        "A message directs viewers to arrange paid sexual encounters privately.",
        "The creator pressures a guest to share intimate material on camera.",
        "A viewer sends persistent sexual propositions to another participant.",
        "The host asks an audience member for private intimate images.",
        "A comment advertises sexual services and requests private contact.",
        "The speaker tries to coerce a viewer into a sexual conversation.",
    ],
    "hateful_abuse": [
        "A speaker says a protected group should be excluded from public life.",
        "A comment dehumanizes people because of their religious identity.",
        "The host calls for a nationality to be denied basic rights.",
        "A viewer attacks people based on a disability and urges exclusion.",
        "A message uses a slur against a protected group and endorses harm.",
        "The speaker claims an ethnic group is inferior and should be expelled.",
        "A comment celebrates discrimination against people of a certain race.",
        "The host directs hateful abuse at people because of their identity.",
        "A viewer argues a protected group should not be allowed to participate.",
        "The speaker promotes hostility toward people for their sexual orientation.",
    ],
}

ACTION = {
    "allow": "allow",
    "scam": "remove",
    "harassment": "restrict",
    "dangerous_activity": "restrict",
    "sexual_solicitation": "remove",
    "hateful_abuse": "remove",
}
MARKETS = ["UK", "US", "BR", "ID", "JP"]
QUEUES = ["LIVE_CHAT", "LIVE_AUDIO", "LIVE_CREATOR"]
TEAMS = ["BPO_ALPHA", "BPO_BRAVO", "BPO_CHARLIE", "INTERNAL_QA"]
DIFFICULTY = ["standard", "borderline", "context_dependent"]


def build_dataset(n_cases: int = 720, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return case-level gold/model features and long-form human annotations."""
    rng = np.random.default_rng(seed)
    categories = list(POLICIES)
    weights = np.array([0.28, 0.15, 0.16, 0.14, 0.12, 0.15])
    records: list[dict] = []
    annotations: list[dict] = []
    annotator_quality = {
        "BPO_ALPHA": 0.91,
        "BPO_BRAVO": 0.84,
        "BPO_CHARLIE": 0.79,
        "INTERNAL_QA": 0.96,
    }
    common_confusions = {
        "allow": ["allow", "allow", "allow", "restrict"],
        "scam": ["remove", "remove", "restrict", "allow"],
        "harassment": ["restrict", "restrict", "allow", "remove"],
        "dangerous_activity": ["restrict", "restrict", "allow", "remove"],
        "sexual_solicitation": ["remove", "remove", "restrict", "allow"],
        "hateful_abuse": ["remove", "remove", "restrict", "allow"],
    }
    for i in range(n_cases):
        category = str(rng.choice(categories, p=weights))
        text = str(rng.choice(POLICIES[category]))
        # Small benign transformations make the text corpus less templated.
        if rng.random() < 0.45:
            text = f"{text} {rng.choice(['Please review the context.', 'This happened during a live session.', 'A moderator flagged the message.', ''])}".strip()
        difficulty = str(rng.choice(DIFFICULTY, p=[0.56, 0.28, 0.16]))
        market, queue = str(rng.choice(MARKETS)), str(rng.choice(QUEUES))
        gold = ACTION[category]
        case_id = f"LG-{i+1:05d}"
        records.append({
            "case_id": case_id, "text": text, "policy_area": category,
            "gold_action": gold, "difficulty": difficulty, "market": market,
            "queue": queue, "week": f"2026-W{int(rng.integers(1, 13)):02d}",
            "is_edge_case": difficulty != "standard",
        })
        for team in TEAMS:
            accuracy = annotator_quality[team] - (0.15 if difficulty == "context_dependent" else 0.07 if difficulty == "borderline" else 0)
            if rng.random() < accuracy:
                label = gold
            else:
                label = str(rng.choice(common_confusions[category]))
            # Occasionally make edge-case failures more concentrated in specific teams.
            if team == "BPO_CHARLIE" and difficulty == "context_dependent" and rng.random() < 0.18:
                label = "allow" if gold != "allow" else "restrict"
            annotations.append({
                "case_id": case_id, "team": team, "label": label,
                "confidence": round(float(np.clip(rng.normal(0.82 if label == gold else 0.63, 0.14), 0.35, 0.99)), 3),
                "review_seconds": int(np.clip(rng.normal(31 if difficulty == "standard" else 53, 13), 8, 130)),
            })
    return pd.DataFrame(records), pd.DataFrame(annotations)

