TOPIC_KEYWORDS = {
    "tesla_business": [
        "tesla",
        "model y",
        "model 3",
        "cybertruck",
        "delivery",
        "deliveries",
        "production",
        "factory",
        "gigafactory",
        "supercharger",
    ],
    "autonomy_robotaxi": [
        "autopilot",
        "fsd",
        "full self-driving",
        "robotaxi",
        "self driving",
        "autonomous",
    ],
    "ai": [
        "ai",
        "artificial intelligence",
        "grok",
        "xai",
        "openai",
        "neural",
    ],
    "semiconductor": [
        "chip",
        "chips",
        "semiconductor",
        "nvidia",
        "gpu",
        "compute",
    ],
    "macro_economy": [
        "recession",
        "inflation",
        "interest rate",
        "rates",
        "fed",
        "economy",
        "dollar",
    ],
    "tariff": [
        "tariff",
        "tariffs",
        "duty",
        "duties",
        "import tax",
    ],
    "china": [
        "china",
        "chinese",
        "beijing",
        "xi",
    ],
    "trade_policy": [
        "trade",
        "deal",
        "export",
        "imports",
        "exports",
        "sanction",
        "sanctions",
    ],
    "fed_rates": [
        "powell",
        "federal reserve",
        "rate cut",
        "rate cuts",
        "interest rate",
        "interest rates",
    ],
    "jobs_economy": [
        "jobs",
        "unemployment",
        "wages",
        "manufacturing",
        "economy",
    ],
    "doge_budget_feud": [
        "doge",
        "budget",
        "subsidy",
        "subsidies",
        "government efficiency",
        "spending bill",
    ],
}

MUSK_TOPICS = [
    "tesla_business",
    "autonomy_robotaxi",
    "ai",
    "semiconductor",
    "macro_economy",
    "doge_budget_feud",
]

TRUMP_TOPICS = [
    "tariff",
    "china",
    "trade_policy",
    "fed_rates",
    "jobs_economy",
    "macro_economy",
    "doge_budget_feud",
]

MUSK_MARKET_WIDE_TOPICS = {"ai", "semiconductor", "macro_economy"}
TRUMP_QQQ_TOPICS = {"china", "tariff"}


def assign_topic(text: str, person: str) -> str:
    text_l = str(text).lower()
    candidates = MUSK_TOPICS if person == "Musk" else TRUMP_TOPICS
    for topic in candidates:
        if any(keyword in text_l for keyword in TOPIC_KEYWORDS[topic]):
            return topic
    return "other"


def is_market_relevant(text: str, person: str) -> bool:
    return assign_topic(text, person) != "other"


def map_ticker(person: str, topic: str) -> str:
    if person == "Musk":
        return "QQQ" if topic in MUSK_MARKET_WIDE_TOPICS else "TSLA"
    return "QQQ" if topic in TRUMP_QQQ_TOPICS else "SPY"
