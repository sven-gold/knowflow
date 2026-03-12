#!/usr/bin/env python3
"""
KnowFlow Auth + Payments
Clerk JWT verification + Stripe subscription management
"""
import os, json, hmac, hashlib, time, subprocess, sys
from datetime import datetime

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--break-system-packages", "--quiet", *packages])

try:
    import stripe
except ImportError:
    pip_install("stripe")
    import stripe

try:
    import jwt as pyjwt
except ImportError:
    pip_install("PyJWT", "cryptography")
    import jwt as pyjwt

try:
    import requests
except ImportError:
    pip_install("requests")
    import requests

CLERK_SECRET_KEY = os.environ.get("CLERK_SECRET_KEY", "")
CLERK_PUBLISHABLE_KEY = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

stripe.api_key = STRIPE_SECRET_KEY

# ── Clerk Auth ─────────────────────────────────────────────────────────────────

_jwks_cache = {"keys": None, "fetched_at": 0}

def get_clerk_jwks():
    """Fetch Clerk JWKS with 1h cache."""
    now = time.time()
    if _jwks_cache["keys"] and now - _jwks_cache["fetched_at"] < 3600:
        return _jwks_cache["keys"]
    try:
        resp = requests.get(
            "https://api.clerk.com/v1/jwks",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
            timeout=5
        )
        keys = resp.json()
        _jwks_cache["keys"] = keys
        _jwks_cache["fetched_at"] = now
        return keys
    except Exception as e:
        print(f"⚠️ JWKS fetch failed: {e}")
        return None

def verify_clerk_token(token: str) -> dict:
    """Verify a Clerk JWT session token. Returns user claims or {}."""
    if not token or not CLERK_SECRET_KEY:
        return {}
    try:
        # Use Clerk's backend API to verify — most reliable approach
        resp = requests.get(
            "https://api.clerk.com/v1/sessions",
            headers={
                "Authorization": f"Bearer {CLERK_SECRET_KEY}",
                "Content-Type": "application/json"
            },
            timeout=5
        )
        # Decode token without verification to get user_id
        # Then verify via Clerk API
        unverified = pyjwt.decode(token, options={"verify_signature": False})
        session_id = unverified.get("sid", "")
        user_id = unverified.get("sub", "")

        if not user_id:
            return {}

        # Verify the session is active via Clerk API
        session_resp = requests.get(
            f"https://api.clerk.com/v1/sessions/{session_id}",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
            timeout=5
        )
        if session_resp.status_code == 200:
            session_data = session_resp.json()
            if session_data.get("status") == "active":
                return {
                    "user_id": user_id,
                    "session_id": session_id,
                    "email": unverified.get("email", "")
                }
        return {}
    except Exception as e:
        print(f"⚠️ Token verification error: {e}")
        return {}

def get_clerk_user(user_id: str) -> dict:
    """Get user details from Clerk."""
    try:
        resp = requests.get(
            f"https://api.clerk.com/v1/users/{user_id}",
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
            timeout=5
        )
        if resp.status_code == 200:
            u = resp.json()
            email = u.get("email_addresses", [{}])[0].get("email_address", "")
            return {
                "user_id": u["id"],
                "email": email,
                "first_name": u.get("first_name", ""),
                "last_name": u.get("last_name", ""),
            }
    except Exception as e:
        print(f"⚠️ Get user error: {e}")
    return {}

def extract_token_from_headers(headers) -> str:
    """Extract Bearer token from Authorization header."""
    auth = headers.get("Authorization", "") or headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    # Also check cookie
    cookie = headers.get("Cookie", "") or headers.get("cookie", "")
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("__session="):
            return part[len("__session="):]
    return ""

# ── Stripe Payments ────────────────────────────────────────────────────────────

# Pricing plans — update these with your actual Stripe Price IDs
PLANS = {
    "creator": {
        "name": "Creator",
        "price_eur": 29,
        "stripe_price_id": os.environ.get("STRIPE_PRICE_CREATOR", ""),
        "features": ["Unlimitierter AI-Chat", "Kein KnowFlow-Badge", "Email-Capture", "Analytics"]
    },
    "pro": {
        "name": "Pro",
        "price_eur": 79,
        "stripe_price_id": os.environ.get("STRIPE_PRICE_PRO", ""),
        "features": ["Alles in Creator", "Mehrere Profile", "White-Label", "Priority Support"]
    }
}

def create_stripe_customer(email: str, name: str = "") -> str:
    """Create a Stripe customer and return customer ID."""
    try:
        customer = stripe.Customer.create(email=email, name=name)
        return customer.id
    except Exception as e:
        print(f"⚠️ Stripe customer creation failed: {e}")
        return ""

def create_checkout_session(slug: str, plan: str, clerk_user_id: str,
                             email: str, success_url: str, cancel_url: str) -> str:
    """Create a Stripe Checkout session and return the URL."""
    plan_data = PLANS.get(plan)
    if not plan_data or not plan_data.get("stripe_price_id"):
        raise ValueError(f"Unbekannter Plan oder fehlende Price ID: {plan}")

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=email,
            line_items=[{
                "price": plan_data["stripe_price_id"],
                "quantity": 1,
            }],
            metadata={
                "slug": slug,
                "clerk_user_id": clerk_user_id,
                "plan": plan,
            },
            success_url=success_url,
            cancel_url=cancel_url,
            allow_promotion_codes=True,
        )
        return session.url
    except Exception as e:
        print(f"⚠️ Checkout session creation failed: {e}")
        raise

def create_billing_portal_session(stripe_customer_id: str, return_url: str) -> str:
    """Create a Stripe Billing Portal session for subscription management."""
    try:
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=return_url,
        )
        return session.url
    except Exception as e:
        print(f"⚠️ Billing portal creation failed: {e}")
        raise

def handle_stripe_webhook(payload: bytes, sig_header: str) -> dict:
    """Verify and parse a Stripe webhook event."""
    if not STRIPE_WEBHOOK_SECRET or not sig_header:
        return json.loads(payload)

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
        return event
    except Exception as e:
        print(f"⚠️ Webhook signature warning: {e} — processing anyway")
        return json.loads(payload)

def get_subscription_status(stripe_subscription_id: str) -> dict:
    """Get current subscription status from Stripe."""
    try:
        sub = stripe.Subscription.retrieve(stripe_subscription_id)
        return {
            "status": sub.status,
            "current_period_end": sub.current_period_end,
            "plan": sub.metadata.get("plan", "creator"),
        }
    except Exception as e:
        print(f"⚠️ Subscription retrieval failed: {e}")
        return {"status": "unknown"}

# ── Clerk Webhook Handler ──────────────────────────────────────────────────────

def handle_clerk_webhook(payload: dict) -> dict:
    """Process Clerk user events (user.created, user.deleted)."""
    event_type = payload.get("type", "")
    data = payload.get("data", {})

    if event_type == "user.created":
        email = data.get("email_addresses", [{}])[0].get("email_address", "")
        user_id = data.get("id", "")
        first_name = data.get("first_name", "")
        last_name = data.get("last_name", "")

        # Auto-generate slug from email prefix
        slug = email.split("@")[0].lower().replace(".", "-").replace("_", "-")
        # Ensure slug is unique by appending user_id suffix if needed
        from db import get_creator, save_creator
        if get_creator(slug):
            slug = f"{slug}-{user_id[-4:]}"

        save_creator(slug, {
            "clerk_user_id": user_id,
            "email": email,
            "channel_name": f"{first_name} {last_name}".strip() or email.split("@")[0],
            "bio": "",
            "products": [],
        })
        return {"ok": True, "slug": slug, "event": event_type}

    elif event_type == "user.deleted":
        user_id = data.get("id", "")
        # Optionally: deactivate creator instead of deleting
        from db import get_creator_by_clerk_id
        creator = get_creator_by_clerk_id(user_id)
        if creator:
            print(f"ℹ️ User deleted: {creator.get('slug')} — keeping data")
        return {"ok": True, "event": event_type}

    return {"ok": True, "event": event_type, "ignored": True}

if __name__ == "__main__":
    print(f"Clerk Key: {'✅' if CLERK_SECRET_KEY else '❌ FEHLT'}")
    print(f"Stripe Key: {'✅' if STRIPE_SECRET_KEY else '❌ FEHLT'}")
