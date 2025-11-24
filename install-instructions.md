# Install & Deployment Instructions (KreatifPanel Prototype)

## 1) Requirements
- PHP 8.0+ with cURL and PDO MySQL extensions
- MySQL 5.7+ or MariaDB
- Composer (optional, for future PHP packages)
- Web server: Apache or Nginx
- Recommended: VPS with HTTPS (Let's Encrypt)

## 2) Setup steps (development)
1. Place project into web root:
   ```
   /var/www/html/kreatifpanel
   ```
2. Edit `config.php` or set environment variables.
3. Create DB and import schema:
   ```
   mysql -u root -p
   CREATE DATABASE kreatifpanel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   CREATE USER 'kreatif'@'localhost' IDENTIFIED BY 'yourpassword';
   GRANT ALL PRIVILEGES ON kreatifpanel.* TO 'kreatif'@'localhost';
   FLUSH PRIVILEGES;
   exit
   mysql -u kreatif -p kreatifpanel < database/schema.sql
   ```
4. Visit `admin/register.php` to create your first user (register page included for dev only).
5. Open `admin/login.php` to access dashboard.

## 3) n8n setup (automation)
- Option A: Quick cloud deploy (recommended for prototyping)
  - Use n8n.cloud or another managed provider (requires account).
- Option B: Self-host on a VPS with Docker
  - Follow n8n docs: https://docs.n8n.io/getting-started/installation/
  - Expose your workflow webhooks (use HTTPS). Set `N8N_WEBHOOK_BASE` in config.php.
- To embed n8n UI inside the panel, ensure CORS and X-Frame-Options are set to allow embedding.
  Alternatively, use `admin/n8n_embed.php` which contains an iframe.

## 4) OpenAI / ChatGPT
- Get API key from OpenAI and set in environment (`OPENAI_API_KEY`).
- The file `api/openai.php` shows how to call the API (placeholder). Replace with your preferred client.

## 5) ManyChat / WhatsApp / Instagram
- Obtain required API keys and configure webhooks.
- Use `api/n8n_webhook.php` as an example endpoint to trigger workflows.

## 6) Production hardening (must do!)
- Use HTTPS (Let's Encrypt).
- Replace the simple auth with a secure system (password_hash, prepared statements).
- Protect admin pages with role checks.
- Move sensitive keys to environment variables and do not commit them.

## 7) Extending features
- Add queue worker (Redis + Supervisor) for background jobs (video generation, large exports).
- Add file storage (S3 or Wasabi) for media assets.
- Add billing via Stripe or iyzico.
- Add usage metering to prevent runaway OpenAI costs.

## 8) Where to go next
- I can generate the full PHP code for each module (customers, automations, n8n control, chat).
- I can also produce Docker + nginx setup for easy deployment.
