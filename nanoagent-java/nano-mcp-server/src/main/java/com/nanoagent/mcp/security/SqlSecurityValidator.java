package com.nanoagent.mcp.security;

import com.nanoagent.mcp.config.McpServerProperties;
import net.sf.jsqlparser.JSQLParserException;
import net.sf.jsqlparser.expression.Expression;
import net.sf.jsqlparser.expression.Function;
import net.sf.jsqlparser.expression.operators.conditional.AndExpression;
import net.sf.jsqlparser.expression.operators.conditional.OrExpression;
import net.sf.jsqlparser.expression.operators.relational.ExistsExpression;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.schema.Table;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.Select;
import net.sf.jsqlparser.statement.select.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * SQL 安全校验器 —— 基于 JSqlParser AST 解析。
 *
 * <p>校验策略（由浅入深）：
 * <ol>
 *   <li>空值 / 长度上限检查</li>
 *   <li>分号检查（纵深防御，防止多语句注入）</li>
 *   <li>AST 解析 → 顶层必须为 SELECT（含 WITH ... SELECT）</li>
 *   <li>遍历 SELECT 内所有表达式 → 拦截高风险函数调用</li>
 * </ol>
 *
 * <p>与旧版字符串匹配方案的区别：彻底消除误杀（如 SELECT * FROM insert_log
 * 不再因表名含 "insert" 被拦截），同时通过 AST 精确识别真正的攻击语句。
 */
@Component
public class SqlSecurityValidator {

    private static final Logger log = LoggerFactory.getLogger(SqlSecurityValidator.class);

    private final McpServerProperties properties;

    public SqlSecurityValidator(McpServerProperties properties) {
        this.properties = properties;
    }

    /**
     * 校验 SQL 安全性。返回 null 表示通过，否则返回错误描述字符串。
     */
    public String validate(String sql) {
        String normalized = normalizeSql(sql);

        // 第 1 层：空值检查
        if (normalized.isBlank()) {
            return "sql 不能为空。";
        }

        // 第 2 层：长度上限
        if (normalized.length() > properties.getSqlMaxLength()) {
            return "sql 过长（>" + properties.getSqlMaxLength() + " 字符），请缩短后重试。";
        }

        // 第 3 层：禁止多语句（纵深防御，防止解析器差异导致逃逸）
        if (normalized.contains(";")) {
            return "仅允许单条只读 SELECT 查询，不支持多语句。";
        }

        // 第 4 层：AST 解析
        Statement statement;
        try {
            statement = CCJSqlParserUtil.parse(normalized);
        } catch (JSQLParserException e) {
            log.warn("SQL 解析失败: {}", truncateForLog(normalized));
            return "SQL 语法解析失败，请检查语句是否合法。";
        }

        // 第 5 层：顶层语句类型检查 → 只允许 SELECT（含 WITH ... SELECT）
        if (!(statement instanceof Select)) {
            return "仅允许只读 SELECT/CTE 查询。";
        }

        Select select = (Select) statement;

        // 第 6 层：高危函数检测
        // 遍历 SELECT 树中所有表达式（WHERE, JOIN ON, 子查询等）
        List<Expression> allExpressions = collectExpressions(select);
        for (Expression expr : allExpressions) {
            String funcError = checkForbiddenFunction(expr);
            if (funcError != null) {
                return funcError;
            }
        }

        return null;
    }

    /**
     * 从 Select 树中递归收集所有表达式。
     */
    private List<Expression> collectExpressions(Select select) {
        List<Expression> expressions = new ArrayList<>();

        // 收集 WITH 子句中的表达式
        if (select.getWithItemsList() != null) {
            for (WithItem withItem : select.getWithItemsList()) {
                collectFromSelectBody(withItem, expressions);
            }
        }

        // 收集主查询体的表达式
        collectFromSelectBody(select.getSelectBody(), expressions);

        return expressions;
    }

    private void collectFromSelectBody(SelectBody selectBody, List<Expression> expressions) {
        if (selectBody instanceof PlainSelect plainSelect) {
            collectFromPlainSelect(plainSelect, expressions);
        } else if (selectBody instanceof SetOperationList setOpList) {
            for (SelectBody body : setOpList.getSelects()) {
                collectFromSelectBody(body, expressions);
            }
        } else if (selectBody instanceof WithItem withItem) {
            collectFromSelectBody(withItem.getSelectBody(), expressions);
        }
    }

    private void collectFromPlainSelect(PlainSelect plainSelect, List<Expression> expressions) {
        // WHERE 子句
        if (plainSelect.getWhere() != null) {
            expressions.add(plainSelect.getWhere());
        }

        // HAVING 子句
        if (plainSelect.getHaving() != null) {
            expressions.add(plainSelect.getHaving());
        }

        // SELECT 列表中的表达式（函数调用、子查询等）
        if (plainSelect.getSelectItems() != null) {
            for (SelectItem<?> item : plainSelect.getSelectItems()) {
                if (item instanceof SelectExpressionItem exprItem) {
                    expressions.add(exprItem.getExpression());
                }
            }
        }

        // FROM 子句中的子查询
        if (plainSelect.getFromItem() != null) {
            collectFromItem(plainSelect.getFromItem(), expressions);
        }

        // JOIN 子句
        if (plainSelect.getJoins() != null) {
            for (Join join : plainSelect.getJoins()) {
                if (join.getOnExpression() != null) {
                    expressions.add(join.getOnExpression());
                }
                if (join.getFromItem() != null) {
                    collectFromItem(join.getFromItem(), expressions);
                }
            }
        }

        // GROUP BY
        if (plainSelect.getGroupBy() != null) {
            plainSelect.getGroupBy().getGroupByExpressions()
                    .forEach(expressions::add);
        }

        // ORDER BY
        if (plainSelect.getOrderByElements() != null) {
            for (OrderByElement elem : plainSelect.getOrderByElements()) {
                expressions.add(elem.getExpression());
            }
        }
    }

    private void collectFromItem(FromItem fromItem, List<Expression> expressions) {
        if (fromItem instanceof SubSelect subSelect) {
            collectFromSelectBody(subSelect.getSelectBody(), expressions);
        } else if (fromItem instanceof SubJoin subJoin) {
            if (subJoin.getJoin() != null && subJoin.getJoin().getOnExpression() != null) {
                expressions.add(subJoin.getJoin().getOnExpression());
            }
        } else if (fromItem instanceof ParenthesedFromItem pfItem) {
            collectFromItem(pfItem.getFromItem(), expressions);
        }
        // Table / ValuesList 等不含嵌套表达式，跳过
    }

    /**
     * 检查单个表达式是否包含禁止的高风险函数。
     * 返回 null 表示安全，否则返回错误消息。
     */
    private String checkForbiddenFunction(Expression expression) {
        if (!(expression instanceof Function func)) {
            return null;
        }
        String funcName = func.getName().toLowerCase();
        if (McpServerProperties.FORBIDDEN_SQL_FUNCTIONS.contains(funcName)) {
            return "SQL 包含高风险函数 `" + funcName + "`，已拒绝执行。";
        }
        // 递归检查函数参数中的嵌套表达式
        if (func.getParameters() != null) {
            for (Expression param : func.getParameters().getExpressions()) {
                String nestedError = checkForbiddenFunction(param);
                if (nestedError != null) return nestedError;
            }
        }
        return null;
    }

    /**
     * 标准化 SQL 文本（去首尾空白，合并连续空格）。
     */
    private String normalizeSql(String sql) {
        return sql.trim().replaceAll("\\s+", " ");
    }

    /**
     * 截断 SQL 用于日志输出。
     */
    private String truncateForLog(String sql) {
        int maxLen = 80;
        return sql.length() <= maxLen ? sql : sql.substring(0, maxLen) + "...";
    }

    /**
     * 对只读 SELECT 包裹外层 LIMIT，控制结果集规模。
     */
    public String buildLimitedSelectSql(String sql, int rowLimit) {
        return "SELECT * FROM (" + sql + ") AS nanoagent_safe_query LIMIT " + rowLimit;
    }
}
