#!/usr/bin/env python3
"""
KnowFlow Database Layer
Replaces JSON-file storage with Neon Postgres.
"""
import os, json, subprocess, sys
from datetime import datetime

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--break-system-packages", "--quiet", *packages])

# Auto-install deps
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("📦 Installing psycopg2...")
    pip_install("psycopg2-binary")
    import psycopg2
    from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pip_install("python-dotenv")
    from dotenv import load_dotenv
    load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL nicht gesetzt! Bitte .env Datei prüfen.")
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS creators (
                    slug TEXT PRIMARY KEY,
                    clerk_user_id TEXT UNIQUE,
                    email TEXT,
                    channel_name TEXT,
                    channel_url TEXT,
                    bio TEXT,
                    avatar TEXT,
                    booking_link TEXT,
                    greeting_video_url TEXT,
                    products JSONB DEFAULT '[]',
                    video_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMPTZ DEFAULT NOW(),
                    stripe_customer_id TEXT,
                    stripe_subscription_id TEXT,
                    subscription_status TEXT DEFAULT 'inactive',
                    plan TEXT DEFAULT 'free',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id SERIAL PRIMARY KEY,
                    slug TEXT REFERENCES creators(slug) ON DELETE CASCADE,
                    content TEXT,
                    word_count INTEGER DEFAULT 0,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE UNIQUE INDEX IF NOT EXISTS knowledge_base_slug_idx
                    ON knowledge_base(slug);

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    slug TEXT REFERENCES creators(slug) ON DELETE CASCADE,
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    message_count INTEGER DEFAULT 0,
                    is_warm BOOLEAN DEFAULT FALSE,
                    warm_at TIMESTAMPTZ,
                    diag_state JSONB DEFAULT '{}',
                    last_active TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS leads (
                    id SERIAL PRIMARY KEY,
                    slug TEXT REFERENCES creators(slug) ON DELETE CASCADE,
                    session_id TEXT,
                    email TEXT,
                    name TEXT,
                    is_warm BOOLEAN DEFAULT FALSE,
                    message_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()
    print("✅ Datenbank initialisiert")

# ── Creator CRUD ───────────────────────────────────────────────────────────────

def get_creator(slug: str) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM creators WHERE slug = %s", (slug,))
            row = cur.fetchone()
            if not row:
                return {}
            result = dict(row)
            # Ensure products is a list
            if isinstance(result.get("products"), str):
                result["products"] = json.loads(result["products"])
            return result

def get_creator_by_clerk_id(clerk_user_id: str) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM creators WHERE clerk_user_id = %s", (clerk_user_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

def save_creator(slug: str, data: dict) -> dict:
    """Upsert creator config — only updates fields that are explicitly provided."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Check if creator exists
            cur.execute("SELECT slug FROM creators WHERE slug = %s", (slug,))
            exists = cur.fetchone()

            if not exists:
                cur.execute("""
                    INSERT INTO creators (slug, channel_name, channel_url, bio, avatar,
                        booking_link, greeting_video_url, products, clerk_user_id, email)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    slug,
                    data.get("channel_name", slug),
                    data.get("channel_url", ""),
                    data.get("bio", ""),
                    data.get("avatar", ""),
                    data.get("booking_link", ""),
                    data.get("greeting_video_url", ""),
                    json.dumps(data.get("products", [])),
                    data.get("clerk_user_id"),
                    data.get("email"),
                ))
            else:
                # Build dynamic update — only update provided non-empty fields
                updates = []
                values = []
                field_map = {
                    "channel_name": "channel_name",
                    "channel_url": "channel_url",
                    "bio": "bio",
                    "avatar": "avatar",
                    "booking_link": "booking_link",
                    "greeting_video_url": "greeting_video_url",
                    "clerk_user_id": "clerk_user_id",
                    "email": "email",
                    "video_count": "video_count",
                }
                for key, col in field_map.items():
                    if key in data and data[key] is not None:
                        if data[key] != "" or key in ("greeting_video_url", "video_count"):
                            updates.append(f"{col} = %s")
                            values.append(data[key])

                # Products handled separately
                if "products" in data and (data["products"] or data.get("clear_products")):
                    updates.append("products = %s")
                    values.append(json.dumps(data["products"]))

                if updates:
                    updates.append("last_updated = NOW()")
                    values.append(slug)
                    cur.execute(
                        f"UPDATE creators SET {', '.join(updates)} WHERE slug = %s",
                        values
                    )
        conn.commit()
    return get_creator(slug)

def get_all_creators() -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.slug, c.channel_name, c.email, c.plan,
                       c.subscription_status, c.created_at,
                       COALESCE(k.word_count, 0) as word_count
                FROM creators c
                LEFT JOIN knowledge_base k ON k.slug = c.slug
                ORDER BY c.created_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]

def update_subscription(slug: str, stripe_customer_id: str,
                        stripe_subscription_id: str, status: str, plan: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE creators SET
                    stripe_customer_id = %s,
                    stripe_subscription_id = %s,
                    subscription_status = %s,
                    plan = %s,
                    last_updated = NOW()
                WHERE slug = %s
            """, (stripe_customer_id, stripe_subscription_id, status, plan, slug))
        conn.commit()

def update_subscription_by_customer(stripe_customer_id: str, status: str, plan: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE creators SET
                    subscription_status = %s,
                    plan = %s,
                    last_updated = NOW()
                WHERE stripe_customer_id = %s
            """, (status, plan, stripe_customer_id))
        conn.commit()

# ── Knowledge Base ─────────────────────────────────────────────────────────────

def get_knowledge(slug: str) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM knowledge_base WHERE slug = %s", (slug,))
            row = cur.fetchone()
            return row["content"][:80000] if row and row["content"] else ""

def save_knowledge(slug: str, content: str):
    word_count = len(content.split())
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO knowledge_base (slug, content, word_count, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (slug) DO UPDATE SET
                    content = EXCLUDED.content,
                    word_count = EXCLUDED.word_count,
                    updated_at = NOW()
            """, (slug, content, word_count))
        conn.commit()

def append_knowledge(slug: str, new_content: str):
    existing = get_knowledge(slug)
    combined = (existing + "\n\n" + new_content).strip() if existing else new_content
    save_knowledge(slug, combined)

def get_word_count(slug: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT word_count FROM knowledge_base WHERE slug = %s", (slug,))
            row = cur.fetchone()
            return row["word_count"] if row else 0

# ── Sessions / Leads ───────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else {}

def save_session(session_id: str, slug: str, message_count: int,
                 is_warm: bool, diag_state: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (session_id, slug, message_count, is_warm,
                    warm_at, diag_state, last_active)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (session_id) DO UPDATE SET
                    message_count = EXCLUDED.message_count,
                    is_warm = EXCLUDED.is_warm,
                    warm_at = CASE WHEN EXCLUDED.is_warm AND sessions.warm_at IS NULL
                                   THEN NOW() ELSE sessions.warm_at END,
                    diag_state = EXCLUDED.diag_state,
                    last_active = NOW()
            """, (
                session_id, slug, message_count, is_warm,
                datetime.now() if is_warm else None,
                json.dumps(diag_state)
            ))
        conn.commit()

def get_leads(slug: str) -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, is_warm, message_count, created_at, last_active
                FROM sessions
                WHERE slug = %s
                ORDER BY last_active DESC
                LIMIT 100
            """, (slug,))
            return [dict(r) for r in cur.fetchall()]

if __name__ == "__main__":
    print("🗄️ Initialisiere Datenbank...")
    init_db()
    print("✅ Fertig!")
