package com.nanoagent.service.auth;

public class AuthContext {
    private AuthType authType;
    private String subject;

    public enum AuthType {
        JWT, API_KEY, DISABLED
    }

    public AuthContext() {}

    public AuthContext(AuthType authType, String subject) {
        this.authType = authType;
        this.subject = subject;
    }

    public AuthType getAuthType() { return authType; }
    public void setAuthType(AuthType authType) { this.authType = authType; }
    public String getSubject() { return subject; }
    public void setSubject(String subject) { this.subject = subject; }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private AuthType authType;
        private String subject;

        public Builder authType(AuthType authType) { this.authType = authType; return this; }
        public Builder subject(String subject) { this.subject = subject; return this; }
        public AuthContext build() { return new AuthContext(authType, subject); }
    }

    public String requireSubject() {
        if (subject == null || subject.isBlank()) {
            throw new AuthException(403, "当前接口要求用户身份认证（JWT）");
        }
        return subject;
    }
}