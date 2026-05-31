-- ============================================================
-- 企业经营分析智能助手 - 业务数据库初始化脚本
-- 使用方法: psql -U your_user -d your_db -f seed_business_data.sql
-- ============================================================

-- 1. 产品表
CREATE TABLE IF NOT EXISTS products (
    product_id   SERIAL PRIMARY KEY,
    product_name VARCHAR(120) NOT NULL,
    category     VARCHAR(60)  NOT NULL,       -- 品类: 电子/服装/食品/家居
    sub_category VARCHAR(60)  DEFAULT '',
    cost_price   NUMERIC(12,2) NOT NULL DEFAULT 0,
    retail_price NUMERIC(12,2) NOT NULL DEFAULT 0,
    unit         VARCHAR(20)  DEFAULT '件',
    status       VARCHAR(20)  DEFAULT 'active', -- active / discontinued
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

-- 2. 客户表
CREATE TABLE IF NOT EXISTS customers (
    customer_id   SERIAL PRIMARY KEY,
    customer_name VARCHAR(120) NOT NULL,
    level         VARCHAR(20)  DEFAULT '普通',  -- 普通 / 银牌 / 金牌 / 钻石
    region        VARCHAR(60)  NOT NULL,         -- 华东/华南/华北/华中/西南/西北/东北
    city          VARCHAR(60)  DEFAULT '',
    contact_phone VARCHAR(30)  DEFAULT '',
    contact_email VARCHAR(120) DEFAULT '',
    first_order_date DATE     DEFAULT CURRENT_DATE,
    created_at    TIMESTAMPTZ  DEFAULT NOW()
);

-- 3. 销售订单表
CREATE TABLE IF NOT EXISTS sales_orders (
    order_id     SERIAL PRIMARY KEY,
    order_no     VARCHAR(30)  NOT NULL UNIQUE,   -- SO-20260501-001
    order_date   DATE         NOT NULL,
    customer_id  INT          NOT NULL REFERENCES customers(customer_id),
    product_id   INT          NOT NULL REFERENCES products(product_id),
    region       VARCHAR(60)  NOT NULL,
    quantity     INT          NOT NULL DEFAULT 1,
    unit_price   NUMERIC(12,2) NOT NULL DEFAULT 0,
    total_amount NUMERIC(14,2) NOT NULL DEFAULT 0,
    discount_pct NUMERIC(5,2)  DEFAULT 0,        -- 折扣百分比 0-100
    order_status VARCHAR(20)   DEFAULT '已完成',  -- 待处理/已发货/已完成/已退货
    created_at   TIMESTAMPTZ   DEFAULT NOW()
);

-- 4. 库存表
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id   SERIAL PRIMARY KEY,
    product_id     INT          NOT NULL REFERENCES products(product_id),
    warehouse      VARCHAR(60)  NOT NULL,         -- 仓库名: 华东仓/华南仓/华北仓/中央仓
    stock_qty      INT          NOT NULL DEFAULT 0,
    safety_stock   INT          NOT NULL DEFAULT 10,
    last_inbound   DATE         DEFAULT CURRENT_DATE,
    last_outbound  DATE         DEFAULT CURRENT_DATE,
    updated_at     TIMESTAMPTZ  DEFAULT NOW()
);

-- 5. 财务月报表
CREATE TABLE IF NOT EXISTS finance_monthly (
    id           SERIAL PRIMARY KEY,
    year_month   VARCHAR(7)    NOT NULL UNIQUE,   -- 2026-05
    revenue      NUMERIC(14,2) NOT NULL DEFAULT 0,
    cogs         NUMERIC(14,2) NOT NULL DEFAULT 0, -- 成本
    gross_profit NUMERIC(14,2) NOT NULL DEFAULT 0,
    opex         NUMERIC(14,2) NOT NULL DEFAULT 0, -- 运营费用
    net_profit   NUMERIC(14,2) NOT NULL DEFAULT 0,
    marketing    NUMERIC(14,2) DEFAULT 0,
    rd_cost      NUMERIC(14,2) DEFAULT 0,
    admin_cost   NUMERIC(14,2) DEFAULT 0,
    updated_at   TIMESTAMPTZ   DEFAULT NOW()
);

-- ============================================================
-- 示例数据
-- ============================================================

-- 产品数据 (20个SKU)
INSERT INTO products (product_name, category, sub_category, cost_price, retail_price, unit) VALUES
('智能手表 Pro',     '电子', '穿戴设备', 299.00, 599.00, '台'),
('无线蓝牙耳机',     '电子', '音频设备', 59.00,  129.00, '副'),
('4K超清投影仪',     '电子', '影音设备', 1299.00,2999.00, '台'),
('机械键盘 RGB',     '电子', '外设',     129.00, 259.00, '个'),
('USB-C 扩展坞',     '电子', '外设',     89.00,  179.00, '个'),
('纯棉圆领T恤',      '服装', '上装',     29.00,  79.00,  '件'),
('商务休闲西装',     '服装', '外套',     199.00, 599.00, '件'),
('运动速干裤',       '服装', '裤装',     49.00,  129.00, '条'),
('轻薄羽绒服',       '服装', '外套',     159.00, 459.00, '件'),
('真丝连衣裙',       '服装', '裙装',     199.00, 599.00, '件'),
('有机坚果礼盒',     '食品', '零食',     39.00,  89.00,  '盒'),
('即溶咖啡 100g',    '食品', '饮品',     15.00,  39.00,  '罐'),
('鲜榨果汁 1L',      '食品', '饮品',     8.00,   19.90,  '瓶'),
('全麦面包 500g',    '食品', '烘焙',     6.00,   14.90,  '袋'),
('进口牛排 300g',    '食品', '生鲜',     49.00,  119.00, '份'),
('北欧风台灯',       '家居', '照明',     59.00,  149.00, '个'),
('记忆棉枕头',       '家居', '寝具',     69.00,  169.00, '个'),
('智能加湿器',       '家居', '小家电',   99.00,  229.00, '台'),
('不锈钢保温杯',     '家居', '日用',     29.00,  69.00,  '个'),
('乳胶床垫 1.8m',    '家居', '寝具',     899.00, 2199.00,'张')
ON CONFLICT DO NOTHING;

-- 客户数据 (30个客户，覆盖7大区域)
INSERT INTO customers (customer_name, level, region, city, contact_email) VALUES
('张伟',     '金牌',  '华东', '上海', 'zhangwei@example.com'),
('李娜',     '普通',  '华东', '杭州', 'lina@example.com'),
('王强',     '钻石',  '华东', '南京', 'wangqiang@example.com'),
('赵敏',     '银牌',  '华东', '苏州', 'zhaomin@example.com'),
('陈刚',     '普通',  '华南', '广州', 'chengang@example.com'),
('林芳',     '金牌',  '华南', '深圳', 'linfang@example.com'),
('黄磊',     '普通',  '华南', '东莞', 'huanglei@example.com'),
('周杰',     '银牌',  '华南', '佛山', 'zhoujie@example.com'),
('吴芳',     '钻石',  '华北', '北京', 'wufang@example.com'),
('刘洋',     '普通',  '华北', '天津', 'liuyang@example.com'),
('孙丽',     '金牌',  '华北', '石家庄','sunli@example.com'),
('马超',     '银牌',  '华北', '济南', 'machao@example.com'),
('杨勇',     '普通',  '华中', '武汉', 'yangyong@example.com'),
('朱莉',     '金牌',  '华中', '长沙', 'zhuli@example.com'),
('何明',     '银牌',  '华中', '郑州', 'heming@example.com'),
('罗琳',     '普通',  '华中', '合肥', 'luolin@example.com'),
('谢飞',     '金牌',  '西南', '成都', 'xiefei@example.com'),
('邓丽',     '普通',  '西南', '重庆', 'dengli@example.com'),
('韩磊',     '银牌',  '西南', '昆明', 'hanlei@example.com'),
('唐伟',     '普通',  '西南', '贵阳', 'tangwei@example.com'),
('曹阳',     '金牌',  '西北', '西安', 'caoyang@example.com'),
('彭丽',     '普通',  '西北', '兰州', 'pengli@example.com'),
('蒋超',     '银牌',  '西北', '乌鲁木齐','jiangchao@example.com'),
('沈明',     '普通',  '东北', '沈阳', 'shenming@example.com'),
('宋丽',     '金牌',  '东北', '大连', 'songli@example.com'),
('潘伟',     '银牌',  '东北', '哈尔滨','panwei@example.com'),
('陆芳',     '普通',  '东北', '长春', 'lufang@example.com'),
('丁强',     '钻石',  '华东', '宁波', 'dingqiang@example.com'),
('魏丽',     '金牌',  '华南', '珠海', 'weili@example.com'),
('薛明',     '银牌',  '华北', '青岛', 'xueming@example.com')
ON CONFLICT DO NOTHING;

-- 库存数据 (每产品多仓)
INSERT INTO inventory (product_id, warehouse, stock_qty, safety_stock, last_inbound, last_outbound)
SELECT p.product_id, w.warehouse,
    CASE WHEN random() < 0.1 THEN 0                           -- 10%概率缺货
         WHEN random() < 0.2 THEN (random() * 8)::int + 2     -- 低库存
         ELSE (random() * 200)::int + 20 END,
    10,
    CURRENT_DATE - (random() * 30)::int,
    CURRENT_DATE - (random() * 7)::int
FROM products p
CROSS JOIN (VALUES ('华东仓'),('华南仓'),('华北仓'),('中央仓')) AS w(warehouse)
ON CONFLICT DO NOTHING;

-- 销售订单数据 (最近6个月，约500条)
INSERT INTO sales_orders (order_no, order_date, customer_id, product_id, region, quantity, unit_price, total_amount, discount_pct, order_status)
SELECT
    'SO-' || TO_CHAR(d, 'YYYYMMDD') || '-' || LPAD(ROW_NUMBER() OVER (PARTITION BY d ORDER BY random())::text, 3, '0'),
    d,
    (SELECT customer_id FROM customers ORDER BY random() LIMIT 1),
    (SELECT product_id FROM products ORDER BY random() LIMIT 1),
    (ARRAY['华东','华南','华北','华中','西南','西北','东北'])[floor(random()*7+1)::int],
    (random() * 10 + 1)::int,
    (random() * 2000 + 10)::numeric(12,2),
    0,
    CASE WHEN random() < 0.3 THEN (random() * 20)::numeric(5,2) ELSE 0 END,
    (ARRAY['已完成','已完成','已完成','已发货','已退货'])[floor(random()*5+1)::int]
FROM generate_series(
    CURRENT_DATE - INTERVAL '180 days',
    CURRENT_DATE,
    '1 day'::interval
) AS d
CROSS JOIN generate_series(1, 3) AS n  -- 每天约3单
LIMIT 500
ON CONFLICT DO NOTHING;

-- 修正 total_amount = quantity * unit_price * (1 - discount/100)
UPDATE sales_orders
SET total_amount = ROUND(quantity * unit_price * (1 - discount_pct / 100), 2)
WHERE total_amount = 0;

-- 财务月报 (最近6个月)
INSERT INTO finance_monthly (year_month, revenue, cogs, gross_profit, opex, net_profit, marketing, rd_cost, admin_cost)
SELECT
    TO_CHAR(d, 'YYYY-MM'),
    (random() * 500000 + 200000)::numeric(14,2),
    (random() * 300000 + 100000)::numeric(14,2),
    0,
    (random() * 80000 + 30000)::numeric(14,2),
    0,
    (random() * 30000 + 10000)::numeric(14,2),
    (random() * 25000 + 8000)::numeric(14,2),
    (random() * 15000 + 5000)::numeric(14,2)
FROM generate_series(
    DATE_TRUNC('month', CURRENT_DATE - INTERVAL '5 months'),
    DATE_TRUNC('month', CURRENT_DATE),
    '1 month'::interval
) AS d
ON CONFLICT DO NOTHING;

-- 修正毛利和净利润
UPDATE finance_monthly
SET gross_profit = revenue - cogs,
    net_profit   = revenue - cogs - opex
WHERE gross_profit = 0;

-- 常用查询索引
CREATE INDEX IF NOT EXISTS idx_orders_date     ON sales_orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_region   ON sales_orders(region);
CREATE INDEX IF NOT EXISTS idx_orders_product  ON sales_orders(product_id);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON sales_orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);

-- 数据库 Schema 注释（供 LLM 理解表结构）
COMMENT ON TABLE products       IS '产品表 - 包含所有在售/已停售商品信息';
COMMENT ON TABLE customers      IS '客户表 - 包含客户基本信息和等级';
COMMENT ON TABLE sales_orders   IS '销售订单表 - 每条记录为一个订单项';
COMMENT ON TABLE inventory      IS '库存表 - 按产品+仓库维度记录库存';
COMMENT ON TABLE finance_monthly IS '财务月报表 - 按月汇总的经营数据';

-- ============================================================
-- 订阅管理 & 调度日志表
-- ============================================================

CREATE TABLE IF NOT EXISTS report_subscriptions (
    id              SERIAL PRIMARY KEY,
    user_id         VARCHAR(64)  NOT NULL,
    report_type     VARCHAR(64)  NOT NULL,
    schedule_cron   VARCHAR(128) NOT NULL,
    recipients      TEXT[]        DEFAULT '{}',
    delivery_method VARCHAR(16)  DEFAULT 'email',
    filters         JSONB         DEFAULT '{}',
    description     TEXT          DEFAULT '',
    is_active       BOOLEAN       DEFAULT TRUE,
    last_run_at     TIMESTAMP,
    next_run_at     TIMESTAMP,
    created_at      TIMESTAMP     DEFAULT NOW(),
    updated_at      TIMESTAMP     DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_subs_active_next ON report_subscriptions (is_active, next_run_at);

CREATE TABLE IF NOT EXISTS scheduler_logs (
    id              SERIAL PRIMARY KEY,
    subscription_id INTEGER REFERENCES report_subscriptions(id) ON DELETE CASCADE,
    status          VARCHAR(16)  NOT NULL,
    report_preview  TEXT,
    error_message   TEXT,
    executed_at     TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sched_logs_sub ON scheduler_logs (subscription_id, executed_at DESC);

COMMENT ON TABLE report_subscriptions IS '报告订阅表 - 用户的定时报告订阅配置';
COMMENT ON TABLE scheduler_logs IS '调度日志表 - 定时报告的执行记录';

-- 完成提示
DO $$ BEGIN RAISE NOTICE '业务数据初始化完成！'; END $$;
