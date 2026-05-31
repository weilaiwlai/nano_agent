package com.nanoagent.service.auth;

import com.nanoagent.service.config.NanoAgentProperties;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
public class AuthWebFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(AuthWebFilter.class);

    private final AuthService authService;
    private final NanoAgentProperties properties;

    public AuthWebFilter(AuthService authService, NanoAgentProperties properties) {
        this.authService = authService;
        this.properties = properties;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {
        if (!properties.getAuth().isRequireApiAuth()) {
            filterChain.doFilter(request, response);
            return;
        }

        String path = request.getRequestURI();

        if (path.equals("/health") || path.equals("/health/mcp")) {
            filterChain.doFilter(request, response);
            return;
        }

        try {
            AuthContext authContext = authService.authenticate(request);
            request.setAttribute("authContext", authContext);
            filterChain.doFilter(request, response);
        } catch (AuthException e) {
            response.setStatus(e.getStatusCode());
            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
            response.getWriter().write("{\"detail\":\"" + e.getMessage() + "\"}");
        }
    }
}