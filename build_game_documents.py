from pathlib import Path
import re
import sqlite3

from steam_sqlite import load_games_from_sqlite


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"
MAX_GAMES = 50_000


def clean_text(x, max_chars=None):
    if x is None:
        return ""
    text = str(x)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars:
        text = text[:max_chars].strip()

    return text


def list_to_text(x, max_items=10):
    if isinstance(x, dict):
        return ", ".join(list(x.keys())[:max_items])
    if isinstance(x, list):
        return ", ".join(str(i) for i in x[:max_items])
    return clean_text(x)

def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def get_release_year(release_date):
    text = clean_text(release_date)
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def build_metadata(appid, raw):
    positive = safe_float(raw.get("positive"), 0)
    negative = safe_float(raw.get("negative"), 0)
    total_reviews = positive + negative

    positive_ratio = None
    if total_reviews > 0:
        positive_ratio = positive / total_reviews

    return {
        "appid": str(appid),
        "name": clean_text(raw.get("name")),
        "price": safe_float(raw.get("price")),
        "windows": bool(raw.get("windows")),
        "mac": bool(raw.get("mac")),
        "linux": bool(raw.get("linux")),
        "release_year": get_release_year(raw.get("release_date")),
        "required_age": int(safe_float(raw.get("required_age"), 0)),
        "positive": int(positive),
        "negative": int(negative),
        "positive_ratio": positive_ratio,
        "recommendations": int(safe_float(raw.get("recommendations"), 0)),
        "average_playtime_forever": int(safe_float(raw.get("average_playtime_forever"), 0)),
    }

def is_low_quality_review(text: str) -> bool:
    """Drop reviews that are mostly caps/punctuation or too short to inform."""
    if len(text) < 40:
        return True
    letters = sum(c.isalpha() for c in text)
    if letters < 0.5 * len(text):          # mostly symbols/emoji/numbers
        return True
    uppers = sum(c.isupper() for c in text)
    if letters and uppers / letters > 0.5:  # SHOUTING / meme reviews
        return True
    return False


def fetch_player_feedback(con, appid, max_reviews=4):
    rows = con.execute(
        """
        SELECT review
        FROM reviews
        WHERE appid = ?
          AND language = 'english'
          AND LENGTH(review) BETWEEN 80 AND 500
          AND votes_up >= 1
        ORDER BY weighted_vote_score DESC, votes_up DESC
        LIMIT ?
        """,
        (appid, max_reviews * 3),  # over-fetch, then filter down
    ).fetchall()

    reviews = []
    for row in rows:
        text = clean_text(row[0], 350)
        if is_low_quality_review(text):
            continue
        reviews.append(text)
        if len(reviews) >= max_reviews:
            break

    return " ".join(reviews)

def build_game_document(raw, player_feedback=""):
    name = clean_text(raw.get("name"))
    description = clean_text(raw.get("short_description"), 600)

    genres = list_to_text(raw.get("genres"), 8)
    tags = list_to_text(raw.get("tags"), 15)
    categories = list_to_text(raw.get("categories"), 10)

    # High-signal identity first (title + genres + tags), restate the title,
    # then the longer-form description and grounded player feedback.
    return f"""
GAME: {name}
GENRES: {genres}
TAGS: {tags}
CATEGORIES: {categories}

ABOUT {name}:
{description}

WHAT PLAYERS SAY:
{player_feedback}
""".strip()

def create_review_index(con):
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reviews_appid_language_score
        ON reviews(appid, language, weighted_vote_score DESC, votes_up DESC)
        """
    )
    con.commit()

def build_all_documents():
    documents = []

    con = sqlite3.connect(DB_PATH)

    try:
        create_review_index(con)

        for i, (appid, raw) in enumerate(load_games_from_sqlite(DB_PATH, MAX_GAMES), start=1):
            player_feedback = fetch_player_feedback(con, appid)
            doc = build_game_document(raw, player_feedback)
            metadata = build_metadata(appid, raw)

            documents.append({
                "appid": str(appid),
                "name": raw.get("name"),
                "document": doc,
                "metadata": metadata,
            })

            if i % 1000 == 0:
                print(f"Built {i} documents...")

    finally:
        con.close()

    return documents

if __name__ == "__main__":
    documents = build_all_documents()

    print(f"\nBuilt {len(documents)} game documents.")

    for item in documents[:5]:
        print("\n" + "=" * 80)
        print(f"APPID: {item['appid']}")
        print(f"NAME: {item['name']}")
        print("\nDOCUMENT:")
        print(item["document"][:1500])
        print("\nMETADATA:")
        print(item["metadata"])