package com.nanoagent.service.graph;

public enum SupervisorDecision {
    KNOWLEDGE_WORKER,
    REPORTER,
    ASSISTANT,
    FINISH;

    public static SupervisorDecision fromText(String text) {
        if (text == null) return FINISH;
        String upper = text.trim().replace("\"", "").replace("'", "").toUpperCase();
        if (upper.contains("KNOWLEDGEWORKER")) return KNOWLEDGE_WORKER;
        if (upper.contains("REPORTER")) return REPORTER;
        if (upper.contains("ASSISTANT")) return ASSISTANT;
        if (upper.contains("FINISH")) return FINISH;
        return FINISH;
    }
}