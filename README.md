# KreatifPanel - Prototype (Starter Pack)

This repository is a **starter scaffold** for the KreatifPanel admin panel you requested.
It includes a minimal PHP + MySQL prototype plus API placeholders to integrate:
- OpenAI / ChatGPT (content generation)
- n8n (automation engine)
- ManyChat / WhatsApp / Instagram (DM bots)
- Payment providers (Stripe / Iyzico)

**Important:** This is a scaffold and not a full production-ready system. It saves you months of setup
by providing the basic structure, DB schema, and integration points. Follow the Installation guide below.

## What is included
- `admin/` : login, dashboard, customers, n8n embed page
- `api/` : openai.php (placeholder), n8n_webhook.php (example)
- `auth/` : simple login/register (very basic, not production-secure)
- `database/` : `schema.sql` to create necessary tables
- `assets/` : css + js skeleton
- `config.php` : central config for DB and API keys (edit before use)
- `install-instructions.md` : step-by-step install & deployment guide
- `.env.example` : environment variables example

## Quick installation (local, development)
1. Copy files to your PHP web root (e.g. `htdocs` or `/var/www/html/kreatifpanel`)
2. Create a MySQL database and user.
3. Edit `config.php` or create a `.env` file and set DB credentials and API keys.
4. Import `database/schema.sql` into your DB.
   ```
   mysql -u youruser -p kreatifpanel < database/schema.sql
   ```
5. Open `http://localhost/kreatifpanel/admin/login.php` and register an account.
6. To enable OpenAI features, set `OPENAI_API_KEY` in config.
7. To use n8n, deploy n8n (see install-instructions.md) and set `N8N_WEBHOOK_BASE` in config.

## Production & Security notes
- Replace the basic auth with a robust system (use password_hash, prepared statements, CSRF protection).
- Set HTTPS and secure PHP session settings.
- Do not store API keys in committed files; use environment variables.
- Rate-limit APIs and add usage billing.

## Files list (important files)
- admin/login.php
- admin/dashboard.php
- admin/customers.php
- admin/n8n_embed.php
- api/openai.php
- api/n8n_webhook.php
- config.php
- database/schema.sql
- install-instructions.md

-----
Generated automatically for you. After downloading, follow the `install-instructions.md`.
