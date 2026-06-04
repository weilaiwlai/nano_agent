package com.nanoagent.service.graph;

public enum OrchestratorDecision {
    DATA_ANALYST,
    REPORTER,
    ASSISTANT,
    FINISH;

    public static OrchestratorDecision fromText(String text) {
        if (text == null) return FINISH;
        String lower = text.trim().replace("\"", "").replace("'", "").toLowerCase();
        if (lower.contains("data_analyst") || lower.contains("dataanalyst") || lower.contains("knowledge_worker")) {
            return DATA_ANALYST;
        }
        if (lower.contains("reporter")) return REPORTER;
        if (lower.contains("assistant")) return ASSISTANT;
        if (lower.contains("finish")) return FINISH;
        return FINISH;
    }
}
