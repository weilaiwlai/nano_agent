package com.nanoagent.mcp.security;

import com.nanoagent.mcp.config.McpServerProperties;
import org.springframework.stereotype.Component;

import java.util.regex.Pattern;

@Component
public class SqlSecurityValidator {

    private final McpServerProperties properties;
    public SqlSecurityValidator(McpServerProperties properties) {
        this.properties = properties;
    }

    public String validate(String sql) {
        String normalized = normalizeSql(sql);
        if (normalized.isBlank()) {
            return "sql 不能为空。";
        }

        if (normalized.length() > properties.getSqlMaxLength()) {
            return "sql 过长（>" + properties.getSqlMaxLength() + " 字符），请缩短后重试。";
        }

        if (normalized.contains(";")) {
            return "仅允许单条只读 SELECT 查询，不支持多语句。";
        }

        String lower = normalized.toLowerCase();
        if (!lower.startsWith("select ") && !lower.startsWith("with ")) {
            return "仅允许只读 SELECT/CTE 查询。";
        }

        String[] forbiddenTokens = {
                " insert ", " update ", " delete ", " drop ", " alter ",
                " create ", " truncate ", " grant ", " revoke ", " merge ",
                " call ", " execute ", " do ", " copy "
        };
        String padded = " " + lower + " ";
        for (String token : forbiddenTokens) {
            if (padded.contains(token)) {
                return "仅允许只读 SELECT 查询；增删改和 DDL 已被禁用。";
            }
        }

        for (String funcName : McpServerProperties.FORBIDDEN_SQL_FUNCTIONS) {
            Pattern pattern = Pattern.compile("\\b" + Pattern.quote(funcName) + "\\s*\\(");
            if (pattern.matcher(lower).find()) {
                return "SQL 包含高风险函数 `" + funcName + "`，已拒绝执行。";
            }
        }

        return null;
    }

    public String buildLimitedSelectSql(String sql, int rowLimit) {
        return "SELECT * FROM (" + sql + ") AS nanoagent_safe_query LIMIT " + rowLimit;
    }

    private String normalizeSql(String sql) {
        return sql.trim().replaceAll("\\s+", " ");
    }
}