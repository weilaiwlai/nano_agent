package com.nanoagent.mcp.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nanoagent.mcp.config.McpServerProperties;
import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class EmailService {

    private static final Logger log = LoggerFactory.getLogger(EmailService.class);

    private final McpServerProperties properties;
    private final JavaMailSender mailSender;
    private final ObjectMapper objectMapper;
    public EmailService(McpServerProperties properties, JavaMailSender mailSender, ObjectMapper objectMapper) {
        this.properties = properties;
        this.mailSender = mailSender;
        this.objectMapper = objectMapper;
    }

    private static final Pattern EMAIL_PATTERN =
            Pattern.compile("^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$");

    public Mono<String> sendReport(String email, String content) {
        String normalizedEmail = email.trim();
        String normalizedContent = content.trim();
        String maskedEmail = maskEmail(normalizedEmail);

        if (!isValidEmail(normalizedEmail)) {
            return jsonResponse(Map.of("status", "error", "message", "邮箱格式无效，请提供合法邮箱地址。"));
        }
        if (normalizedContent.isBlank()) {
            return jsonResponse(Map.of("status", "error", "message", "content 不能为空。"));
        }
        if (normalizedContent.length() > properties.getReport().getMaxContentChars()) {
            return jsonResponse(Map.of(
                    "status", "error",
                    "message", "content 过长（>" + properties.getReport().getMaxContentChars() + " 字符），请缩短后重试。"
            ));
        }

        if (!isReportEmailAllowed(normalizedEmail)) {
            return jsonResponse(Map.of("status", "error", "message", "目标邮箱域名不在白名单内，已拒绝发送。"));
        }

        String provider = properties.getReport().getProvider();
        if ("smtp".equalsIgnoreCase(provider)) {
            return sendSmtpEmail(normalizedEmail, maskedEmail, normalizedContent);
        }

        return sendMockEmail(normalizedEmail, maskedEmail, normalizedContent);
    }

    private Mono<String> sendSmtpEmail(String email, String maskedEmail, String content) {
        McpServerProperties.SmtpConfig smtp = properties.getSmtp();
        if (smtp.getHost().isBlank() || smtp.getFrom().isBlank()) {
            return jsonResponse(Map.of("status", "error", "message", "SMTP 配置不完整"));
        }

        return Mono.fromCallable(() -> {
            try {
                MimeMessage message = mailSender.createMimeMessage();
                MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
                helper.setFrom(smtp.getFrom());
                helper.setTo(email);
                helper.setSubject(properties.getReport().getSubject());
                helper.setText(content);

                mailSender.send(message);

                log.info("SMTP report sent | email={} | content_length={}", maskedEmail, content.length());
                return jsonResponseSync(Map.of(
                        "status", "success", "provider", "smtp",
                        "delivery", "sent", "email", maskedEmail,
                        "timestamp", Instant.now().toString(),
                        "message", "报告发送成功。"
                ));
            } catch (MessagingException e) {
                log.error("SMTP send failed | email={} | error={}", maskedEmail, e.getMessage());
                return jsonResponseSync(Map.of(
                        "status", "error", "message", "SMTP 发送失败，请检查邮件服务配置或网络连通性。"
                ));
            }
        }).subscribeOn(Schedulers.boundedElastic());
    }

    private Mono<String> sendMockEmail(String email, String maskedEmail, String content) {
        log.info("Mock report sent | email={} | content_length={}", maskedEmail, content.length());
        return jsonResponse(Map.of(
                "status", "success", "provider", "mock",
                "delivery", "simulated", "email", maskedEmail,
                "timestamp", Instant.now().toString(),
                "message", "模拟发送成功。"
        ));
    }

    private boolean isValidEmail(String email) {
        return EMAIL_PATTERN.matcher(email).matches();
    }

    private boolean isReportEmailAllowed(String email) {
        var allowedDomains = properties.getReport().getAllowedEmailDomains();
        if (allowedDomains == null || allowedDomains.isEmpty()) {
            return true;
        }
        String domain = email.substring(email.indexOf('@') + 1).toLowerCase();
        return allowedDomains.contains(domain);
    }

    private String maskEmail(String email) {
        int atIndex = email.indexOf('@');
        if (atIndex <= 0) return "***";
        String local = email.substring(0, atIndex);
        String domain = email.substring(atIndex);
        int keep = local.length() < 4 ? 1 : 2;
        return local.substring(0, keep) + "***" + domain;
    }

    private Mono<String> jsonResponse(Map<String, Object> payload) {
        try {
            return Mono.just(objectMapper.writeValueAsString(payload));
        } catch (JsonProcessingException e) {
            return Mono.just("{\"status\":\"error\",\"message\":\"Serialization error\"}");
        }
    }

    private String jsonResponseSync(Map<String, Object> payload) {
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException e) {
            return "{\"status\":\"error\",\"message\":\"Serialization error\"}";
        }
    }
}