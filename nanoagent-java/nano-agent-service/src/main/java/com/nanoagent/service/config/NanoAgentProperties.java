package com.nanoagent.service.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.List;
import java.util.Set;

@Configuration
@ConfigurationProperties(prefix = "nanoagent")
public class NanoAgentProperties {

    private String environment = "development";
    private boolean debugMode = false;
    private int autoMemoryMaxLen = 120;
    private int graphRecursionLimit = 25;
    private int maxModelHistoryMessages = 6;
    private int reportContentSoftLimit = 8000;
    private int emailDraftTargetChars = 2000;
    private int maxToolPayloadLength = 400;
    private String allowedLlmBaseUrls = "";
    private String tavilyApiKey = "";
    private String graphCheckpointerBackend = "memory";
    private String graphCheckpointerRedisUrl = "";
    private String graphCheckpointerPrefix = "nanoagent";
    private String graphCheckpointerTableName = "graph_checkpoints";

    private McpConfig mcp = new McpConfig();
    private AuthConfig auth = new AuthConfig();
    private CorsConfig cors = new CorsConfig();
    private SessionConfig session = new SessionConfig();
    private EmbeddingConfig embedding = new EmbeddingConfig();

    public String getEnvironment() { return environment; }
    public void setEnvironment(String environment) { this.environment = environment; }
    public boolean isDebugMode() { return debugMode; }
    public void setDebugMode(boolean debugMode) { this.debugMode = debugMode; }
    public int getAutoMemoryMaxLen() { return autoMemoryMaxLen; }
    public void setAutoMemoryMaxLen(int autoMemoryMaxLen) { this.autoMemoryMaxLen = autoMemoryMaxLen; }
    public int getGraphRecursionLimit() { return graphRecursionLimit; }
    public void setGraphRecursionLimit(int graphRecursionLimit) { this.graphRecursionLimit = graphRecursionLimit; }
    public int getMaxModelHistoryMessages() { return maxModelHistoryMessages; }
    public void setMaxModelHistoryMessages(int maxModelHistoryMessages) { this.maxModelHistoryMessages = maxModelHistoryMessages; }
    public int getReportContentSoftLimit() { return reportContentSoftLimit; }
    public void setReportContentSoftLimit(int reportContentSoftLimit) { this.reportContentSoftLimit = reportContentSoftLimit; }
    public int getEmailDraftTargetChars() { return emailDraftTargetChars; }
    public void setEmailDraftTargetChars(int emailDraftTargetChars) { this.emailDraftTargetChars = emailDraftTargetChars; }
    public int getMaxToolPayloadLength() { return maxToolPayloadLength; }
    public void setMaxToolPayloadLength(int maxToolPayloadLength) { this.maxToolPayloadLength = maxToolPayloadLength; }
    public String getAllowedLlmBaseUrls() { return allowedLlmBaseUrls; }
    public void setAllowedLlmBaseUrls(String allowedLlmBaseUrls) { this.allowedLlmBaseUrls = allowedLlmBaseUrls; }
    public String getTavilyApiKey() { return tavilyApiKey; }
    public void setTavilyApiKey(String tavilyApiKey) { this.tavilyApiKey = tavilyApiKey; }
    public String getGraphCheckpointerBackend() { return graphCheckpointerBackend; }
    public void setGraphCheckpointerBackend(String graphCheckpointerBackend) { this.graphCheckpointerBackend = graphCheckpointerBackend; }
    public String getGraphCheckpointerRedisUrl() { return graphCheckpointerRedisUrl; }
    public void setGraphCheckpointerRedisUrl(String graphCheckpointerRedisUrl) { this.graphCheckpointerRedisUrl = graphCheckpointerRedisUrl; }
    public String getGraphCheckpointerPrefix() { return graphCheckpointerPrefix; }
    public void setGraphCheckpointerPrefix(String graphCheckpointerPrefix) { this.graphCheckpointerPrefix = graphCheckpointerPrefix; }
    public String getGraphCheckpointerTableName() { return graphCheckpointerTableName; }
    public void setGraphCheckpointerTableName(String graphCheckpointerTableName) { this.graphCheckpointerTableName = graphCheckpointerTableName; }
    public McpConfig getMcp() { return mcp; }
    public void setMcp(McpConfig mcp) { this.mcp = mcp; }
    public AuthConfig getAuth() { return auth; }
    public void setAuth(AuthConfig auth) { this.auth = auth; }
    public CorsConfig getCors() { return cors; }
    public void setCors(CorsConfig cors) { this.cors = cors; }
    public SessionConfig getSession() { return session; }
    public void setSession(SessionConfig session) { this.session = session; }
    public EmbeddingConfig getEmbedding() { return embedding; }
    public void setEmbedding(EmbeddingConfig embedding) { this.embedding = embedding; }

    public static class McpConfig {
        private String baseUrl = "http://localhost:8000";
        private String serviceToken = "";

        public String getBaseUrl() { return baseUrl; }
        public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
        public String getServiceToken() { return serviceToken; }
        public void setServiceToken(String serviceToken) { this.serviceToken = serviceToken; }
    }

    public static class AuthConfig {
        private boolean requireApiAuth = false;
        private boolean requireUserSub = true;
        private boolean allowApiKeyFallback = false;
        private List<String> allowedApiKeys = List.of();
        private String jwtJwksUrl = "";
        private String jwtIssuer = "";
        private String jwtAudience = "";
        private String jwtHs256Secret = "";
        private List<String> jwtAlgorithms = List.of("RS256", "ES256", "HS256");
        private int jwtLeewaySeconds = 30;

        public boolean isRequireApiAuth() { return requireApiAuth; }
        public void setRequireApiAuth(boolean requireApiAuth) { this.requireApiAuth = requireApiAuth; }
        public boolean isRequireUserSub() { return requireUserSub; }
        public void setRequireUserSub(boolean requireUserSub) { this.requireUserSub = requireUserSub; }
        public boolean isAllowApiKeyFallback() { return allowApiKeyFallback; }
        public void setAllowApiKeyFallback(boolean allowApiKeyFallback) { this.allowApiKeyFallback = allowApiKeyFallback; }
        public List<String> getAllowedApiKeys() { return allowedApiKeys; }
        public void setAllowedApiKeys(List<String> allowedApiKeys) { this.allowedApiKeys = allowedApiKeys; }
        public String getJwtJwksUrl() { return jwtJwksUrl; }
        public void setJwtJwksUrl(String jwtJwksUrl) { this.jwtJwksUrl = jwtJwksUrl; }
        public String getJwtIssuer() { return jwtIssuer; }
        public void setJwtIssuer(String jwtIssuer) { this.jwtIssuer = jwtIssuer; }
        public String getJwtAudience() { return jwtAudience; }
        public void setJwtAudience(String jwtAudience) { this.jwtAudience = jwtAudience; }
        public String getJwtHs256Secret() { return jwtHs256Secret; }
        public void setJwtHs256Secret(String jwtHs256Secret) { this.jwtHs256Secret = jwtHs256Secret; }
        public List<String> getJwtAlgorithms() { return jwtAlgorithms; }
        public void setJwtAlgorithms(List<String> jwtAlgorithms) { this.jwtAlgorithms = jwtAlgorithms; }
        public int getJwtLeewaySeconds() { return jwtLeewaySeconds; }
        public void setJwtLeewaySeconds(int jwtLeewaySeconds) { this.jwtLeewaySeconds = jwtLeewaySeconds; }
    }

    public static class CorsConfig {
        private List<String> allowOrigins = List.of("http://localhost:8501", "http://127.0.0.1:8501");
        private boolean allowCredentials = true;

        public List<String> getAllowOrigins() { return allowOrigins; }
        public void setAllowOrigins(List<String> allowOrigins) { this.allowOrigins = allowOrigins; }
        public boolean isAllowCredentials() { return allowCredentials; }
        public void setAllowCredentials(boolean allowCredentials) { this.allowCredentials = allowCredentials; }
    }

    public static class SessionConfig {
        private String masterKey = "";
        private int ttlSeconds = 3600;
        private int maxTtlSeconds = 86400;
        private boolean requireLlmSession = true;

        public String getMasterKey() { return masterKey; }
        public void setMasterKey(String masterKey) { this.masterKey = masterKey; }
        public int getTtlSeconds() { return ttlSeconds; }
        public void setTtlSeconds(int ttlSeconds) { this.ttlSeconds = ttlSeconds; }
        public int getMaxTtlSeconds() { return maxTtlSeconds; }
        public void setMaxTtlSeconds(int maxTtlSeconds) { this.maxTtlSeconds = maxTtlSeconds; }
        public boolean isRequireLlmSession() { return requireLlmSession; }
        public void setRequireLlmSession(boolean requireLlmSession) { this.requireLlmSession = requireLlmSession; }
    }

    public static class EmbeddingConfig {
        private String defaultModel = "text-embedding-v3";

        public String getDefaultModel() { return defaultModel; }
        public void setDefaultModel(String defaultModel) { this.defaultModel = defaultModel; }
    }

    public Set<String> getProductionEnvAliases() {
        return Set.of("production", "prod");
    }

    public boolean isProduction() {
        return getProductionEnvAliases().contains(environment.toLowerCase());
    }
}