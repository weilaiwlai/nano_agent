package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.NodeAction;
import com.nanoagent.service.config.NanoAgentProperties;
import com.nanoagent.service.memory.UserMemoryManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RetrieveMemoryNode implements NodeAction {

    private static final Logger log = LoggerFactory.getLogger(RetrieveMemoryNode.class);

    private final UserMemoryManager memoryManager;
    private final NanoAgentProperties properties;

    public RetrieveMemoryNode(UserMemoryManager memoryManager, NanoAgentProperties properties) {
        this.memoryManager = memoryManager;
        this.properties = properties;
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> apply(OverAllState state) throws Exception {
        String userId = (String) state.value("userId").orElse("");
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());

        String latestQuery = "";
        for (int i = messages.size() - 1; i >= 0; i--) {
            if (messages.get(i).getType() == AgentState.Message.MessageType.HUMAN) {
                latestQuery = messages.get(i).getContent() != null ? messages.get(i).getContent().trim() : "";
                break;
            }
        }

        log.info("Node start | retrieve_memory_node | user_id={}", userId);

        String memoryContext;
        try {
            memoryContext = memoryManager.retrieveRelevantMemories(userId, latestQuery);
        } catch (Exception e) {
            log.error("Memory retrieval error | user_id={} | error={}", userId, e.getMessage());
            memoryContext = "";
        }

        log.info("Node end | retrieve_memory_node | user_id={} | memory_len={}", userId, memoryContext.length());

        Map<String, Object> result = new HashMap<>();
        result.put("messages", messages);
        result.put("memoryContext", memoryContext);
        result.put("sender", "");
        return result;
    }
}