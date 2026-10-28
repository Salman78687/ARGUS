-- Sample schema + seed data matching db_sources.example.json.
-- Lets you try ARGUS's live-DB context without your own database:
--   createdb argus && psql argus -f examples/schema.sql

CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.error_logs (
    id          SERIAL PRIMARY KEY,
    application TEXT NOT NULL,
    action      TEXT NOT NULL,
    code        TEXT,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app.job_runs (
    id            SERIAL PRIMARY KEY,
    job_name      TEXT NOT NULL,
    status        TEXT NOT NULL,
    error_message TEXT,
    duration_ms   INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app.audit_logs (
    id          SERIAL PRIMARY KEY,
    subject     TEXT NOT NULL,
    type        TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO app.error_logs (application, action, code, description, created_at) VALUES
    ('Mail Connector', 'Send Notification', 'SMTP_AUTH_FAILED', 'Authentication rejected by SMTP server', now() - interval '2 hours'),
    ('Webhook', 'Deliver Event', 'HTTP_504', 'Upstream endpoint timed out after 30s', now() - interval '5 hours');

INSERT INTO app.job_runs (job_name, status, error_message, duration_ms, created_at) VALUES
    ('nightly-export', 'failed', 'Disk quota exceeded on export volume', 41230, now() - interval '9 hours'),
    ('user-sync', 'completed', NULL, 1820, now() - interval '1 hour');

INSERT INTO app.audit_logs (subject, type, description, created_at) VALUES
    ('Workflow "Onboarding"', 'updated', 'Step 3 condition changed by admin', now() - interval '3 hours'),
    ('Ticket #1042', 'assigned', 'Assigned to Tier 2 queue', now() - interval '30 minutes');

-- Recommended: give ARGUS a read-only role
-- CREATE ROLE argus_readonly LOGIN PASSWORD '<choose-a-password>';
-- GRANT USAGE ON SCHEMA app TO argus_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA app TO argus_readonly;
