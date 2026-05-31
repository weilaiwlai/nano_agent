package com.nanoagent.service.memory;

public class MemoryProviderException extends RuntimeException {
    private final String reason;
    private final boolean retriable;

    public MemoryProviderException(String message, String reason, boolean retriable) {
        super(message);
        this.reason = reason;
        this.retriable = retriable;
    }

    public String getReason() {
        return reason;
    }

    public boolean isRetriable() {
        return retriable;
    }
}