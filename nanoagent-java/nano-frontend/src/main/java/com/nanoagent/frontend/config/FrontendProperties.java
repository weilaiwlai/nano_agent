package com.nanoagent.frontend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "nanoagent.frontend")
public class FrontendProperties {

    private String agentApiBaseUrl = "http://localhost:8080";
    private String agentApiToken = "";
    private String pageTitle = "NanoAgent";

    public String getAgentApiBaseUrl() { return agentApiBaseUrl; }
    public void setAgentApiBaseUrl(String agentApiBaseUrl) { this.agentApiBaseUrl = agentApiBaseUrl; }
    public String getAgentApiToken() { return agentApiToken; }
    public void setAgentApiToken(String agentApiToken) { this.agentApiToken = agentApiToken; }
    public String getPageTitle() { return pageTitle; }
    public void setPageTitle(String pageTitle) { this.pageTitle = pageTitle; }
}