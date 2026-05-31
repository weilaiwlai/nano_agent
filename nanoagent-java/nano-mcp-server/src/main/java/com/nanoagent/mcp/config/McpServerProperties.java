package com.nanoagent.mcp.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.Set;

@Configuration
@ConfigurationProperties(prefix = "nanoagent.mcp")
public class McpServerProperties {

    public static final Set<String> FORBIDDEN_SQL_FUNCTIONS = Set.of(
            "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
            "pg_sleep", "pg_terminate_backend", "pg_cancel_backend",
            "lo_import", "lo_export", "current_setting",
            "set_config", "pg_advisory_lock", "pg_advisory_unlock"
    );

    private boolean requireAuth = true;
    private String serviceToken = "";
    private int maxLogTextLength = 240;
    private int queryRowLimit = 200;
    private int queryTimeoutMs = 3000;
    private int sqlMaxLength = 4000;

    private ReportConfig report = new ReportConfig();
    private SmtpConfig smtp = new SmtpConfig();

    public boolean isRequireAuth() { return requireAuth; }
    public void setRequireAuth(boolean requireAuth) { this.requireAuth = requireAuth; }
    public String getServiceToken() { return serviceToken; }
    public void setServiceToken(String serviceToken) { this.serviceToken = serviceToken; }
    public int getMaxLogTextLength() { return maxLogTextLength; }
    public void setMaxLogTextLength(int maxLogTextLength) { this.maxLogTextLength = maxLogTextLength; }
    public int getQueryRowLimit() { return queryRowLimit; }
    public void setQueryRowLimit(int queryRowLimit) { this.queryRowLimit = queryRowLimit; }
    public int getQueryTimeoutMs() { return queryTimeoutMs; }
    public void setQueryTimeoutMs(int queryTimeoutMs) { this.queryTimeoutMs = queryTimeoutMs; }
    public int getSqlMaxLength() { return sqlMaxLength; }
    public void setSqlMaxLength(int sqlMaxLength) { this.sqlMaxLength = sqlMaxLength; }
    public ReportConfig getReport() { return report; }
    public void setReport(ReportConfig report) { this.report = report; }
    public SmtpConfig getSmtp() { return smtp; }
    public void setSmtp(SmtpConfig smtp) { this.smtp = smtp; }

    public static class ReportConfig {
        private String provider = "mock";
        private int maxContentChars = 12000;
        private int softBodyChars = 2000;
        private int summaryPreviewChars = 500;
        private boolean attachOverflow = true;
        private String attachmentPrefix = "nanoagent_report";
        private String subject = "NanoAgent 自动报告";
        private Set<String> allowedEmailDomains = Set.of();

        public String getProvider() { return provider; }
        public void setProvider(String provider) { this.provider = provider; }
        public int getMaxContentChars() { return maxContentChars; }
        public void setMaxContentChars(int maxContentChars) { this.maxContentChars = maxContentChars; }
        public int getSoftBodyChars() { return softBodyChars; }
        public void setSoftBodyChars(int softBodyChars) { this.softBodyChars = softBodyChars; }
        public int getSummaryPreviewChars() { return summaryPreviewChars; }
        public void setSummaryPreviewChars(int summaryPreviewChars) { this.summaryPreviewChars = summaryPreviewChars; }
        public boolean isAttachOverflow() { return attachOverflow; }
        public void setAttachOverflow(boolean attachOverflow) { this.attachOverflow = attachOverflow; }
        public String getAttachmentPrefix() { return attachmentPrefix; }
        public void setAttachmentPrefix(String attachmentPrefix) { this.attachmentPrefix = attachmentPrefix; }
        public String getSubject() { return subject; }
        public void setSubject(String subject) { this.subject = subject; }
        public Set<String> getAllowedEmailDomains() { return allowedEmailDomains; }
        public void setAllowedEmailDomains(Set<String> allowedEmailDomains) { this.allowedEmailDomains = allowedEmailDomains; }
    }

    public static class SmtpConfig {
        private String host = "";
        private int port = 587;
        private String username = "";
        private String password = "";
        private String from = "";
        private boolean useTls = true;
        private boolean useSsl = false;
        private int timeoutSeconds = 10;

        public String getHost() { return host; }
        public void setHost(String host) { this.host = host; }
        public int getPort() { return port; }
        public void setPort(int port) { this.port = port; }
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getFrom() { return from; }
        public void setFrom(String from) { this.from = from; }
        public boolean isUseTls() { return useTls; }
        public void setUseTls(boolean useTls) { this.useTls = useTls; }
        public boolean isUseSsl() { return useSsl; }
        public void setUseSsl(boolean useSsl) { this.useSsl = useSsl; }
        public int getTimeoutSeconds() { return timeoutSeconds; }
        public void setTimeoutSeconds(int timeoutSeconds) { this.timeoutSeconds = timeoutSeconds; }
    }

    public static final Set<String> ALLOWED_SETTING_KEYS = Set.of(
            "report_language", "career_direction", "timezone", "notification_channel"
    );
}