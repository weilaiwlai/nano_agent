package com.nanoagent.mcp.config;

import com.nanoagent.mcp.security.ServiceAuthFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.reactive.EnableWebFluxSecurity;
import org.springframework.security.config.web.server.SecurityWebFiltersOrder;
import org.springframework.security.config.web.server.ServerHttpSecurity;
import org.springframework.security.web.server.SecurityWebFilterChain;

@Configuration
@EnableWebFluxSecurity
public class SecurityConfig {

    private final ServiceAuthFilter serviceAuthFilter;
    public SecurityConfig(ServiceAuthFilter serviceAuthFilter) {
        this.serviceAuthFilter = serviceAuthFilter;
    }

    @Bean
    public SecurityWebFilterChain securityWebFilterChain(ServerHttpSecurity http) {
        return http
                .csrf(ServerHttpSecurity.CsrfSpec::disable)
                .authorizeExchange(exchanges -> exchanges
                        .pathMatchers("/health", "/health/mcp").permitAll()
                        .anyExchange().permitAll()
                )
                .addFilterBefore(serviceAuthFilter, SecurityWebFiltersOrder.AUTHENTICATION)
                .build();
    }
}