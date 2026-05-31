package com.nanoagent.service.config;

import com.nanoagent.service.auth.AuthWebFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.List;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private final NanoAgentProperties properties;
    private final AuthWebFilter authWebFilter;

    public SecurityConfig(NanoAgentProperties properties, AuthWebFilter authWebFilter) {
        this.properties = properties;
        this.authWebFilter = authWebFilter;
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        return http
                .csrf(csrf -> csrf.disable())
                .cors(cors -> cors.configurationSource(corsConfigurationSource()))
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/health", "/health/mcp").permitAll()
                        .anyRequest().permitAll()
                )
                .addFilterBefore(authWebFilter, UsernamePasswordAuthenticationFilter.class)
                .build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        NanoAgentProperties.CorsConfig cors = properties.getCors();

        if (cors.getAllowOrigins().contains("*")) {
            config.addAllowedOriginPattern("*");
            config.setAllowCredentials(false);
        } else {
            config.setAllowedOrigins(cors.getAllowOrigins());
            config.setAllowCredentials(cors.isAllowCredentials());
        }

        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*"));
        config.setMaxAge(3600L);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return source;
    }
}