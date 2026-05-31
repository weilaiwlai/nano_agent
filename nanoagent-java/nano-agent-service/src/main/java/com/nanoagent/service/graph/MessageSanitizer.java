package com.nanoagent.service.graph;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public final class MessageSanitizer {

    private static final Logger log = LoggerFactory.getLogger(MessageSanitizer.class);

    private MessageSanitizer() {}

    public static List<AgentState.Message> sanitizeForModel(List<AgentState.Message> messages, int maxMessages) {
        if (messages == null || messages.isEmpty()) {
            return List.of();
        }

        List<AgentState.Message> sanitized = new ArrayList<>();
        int droppedCount = 0;
        int index = 0;
        int total = messages.size();

        while (index < total) {
            AgentState.Message message = messages.get(index);

            if (message.getType() == AgentState.Message.MessageType.AI && message.getToolCalls() != null && !message.getToolCalls().isEmpty()) {
                List<String> toolCallIds = new ArrayList<>();
                for (AgentState.ToolCall tc : message.getToolCalls()) {
                    if (tc.getId() != null && !tc.getId().isBlank()) {
                        toolCallIds.add(tc.getId());
                    }
                }

                int nextIndex = index + 1;
                List<AgentState.Message> followingTools = new ArrayList<>();
                while (nextIndex < total && messages.get(nextIndex).getType() == AgentState.Message.MessageType.TOOL) {
                    followingTools.add(messages.get(nextIndex));
                    nextIndex++;
                }

                if (toolCallIds.isEmpty()) {
                    droppedCount++;
                    index = nextIndex;
                    continue;
                }

                boolean allFound = true;
                for (String reqId : toolCallIds) {
                    boolean found = false;
                    for (AgentState.Message tm : followingTools) {
                        String tci = tm.getToolCallId();
                        if (tci != null && tci.equals(reqId)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        allFound = false;
                        break;
                    }
                }

                if (allFound) {
                    sanitized.add(message);
                    for (AgentState.Message tm : followingTools) {
                        String tci = tm.getToolCallId();
                        if (tci != null && toolCallIds.contains(tci)) {
                            sanitized.add(tm);
                        } else {
                            droppedCount++;
                        }
                    }
                    index = nextIndex;
                    continue;
                }

                droppedCount += 1 + followingTools.size();
                index = nextIndex;
                continue;
            }

            if (message.getType() == AgentState.Message.MessageType.TOOL) {
                droppedCount++;
                index++;
                continue;
            }

            sanitized.add(message);
            index++;
        }

        if (maxMessages > 0 && sanitized.size() > maxMessages) {
            log.info("History truncated | max_messages={} | original={}", maxMessages, sanitized.size());
            sanitized = new ArrayList<>(sanitized.subList(sanitized.size() - maxMessages, sanitized.size()));
        }

        if (droppedCount > 0) {
            log.info("History sanitization complete | dropped={} | kept={}", droppedCount, sanitized.size());
        }

        return sanitized;
    }
}