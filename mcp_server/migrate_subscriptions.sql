-- ============================================================
-- 订阅管理 & 调度日志表
-- ============================================================

-- 订阅配置表
CREATE TABLE IF NOT EXISTS report_subscriptions (
    id              SERIAL PRIMARY KEY,
    user_id         VARCHAR(64)  NOT NULL,
    report_type     VARCHAR(64)  NOT NULL,          -- daily_sales / weekly_summary / monthly_finance / anomaly_digest / custom
    schedule_cron   VARCHAR(128) NOT NULL,           -- "0 9 * * 1" (每周一9点)
    recipients      TEXT[]        DEFAULT '{}',       -- 邮件接收人列表
    delivery_method VARCHAR(16)  DEFAULT 'email',    -- email / chat
    filters         JSONB         DEFAULT '{}',       -- 过滤条件: {"region":"华东","category":"电子产品"}
    description     TEXT          DEFAULT '',          -- 用户自定义描述
    is_active       BOOLEAN       DEFAULT TRUE,
    last_run_at     TIMESTAMP,
    next_run_at     TIMESTAMP,
    created_at      TIMESTAMP     DEFAULT NOW(),
    updated_at      TIMESTAMP     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subs_active_next ON report_subscriptions (is_active, next_run_at);

-- 调度执行日志表
CREATE TABLE IF NOT EXISTS scheduler_logs (
    id              SERIAL PRIMARY KEY,
    subscription_id INTEGER REFERENCES report_subscriptions(id) ON DELETE CASCADE,
    status          VARCHAR(16)  NOT NULL,           -- success / failed / skipped
    report_preview  TEXT,                             -- 报告摘要（用于审计）
    error_message   TEXT,
    executed_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sched_logs_sub ON scheduler_logs (subscription_id, executed_at DESC);
