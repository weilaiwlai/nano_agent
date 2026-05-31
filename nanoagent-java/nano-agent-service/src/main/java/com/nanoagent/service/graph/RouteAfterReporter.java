package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class RouteAfterReporter implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterReporter.class);

    @Override
    @SuppressWarnings("unchecked")
    public String apply(OverAllState state) {
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        if (messages.isEmpty()) {
            log.info("Route | reporter -> END | reason=no_messages");
            return "FINISH";
        }

        AgentState.Message lastMessage = messages.get(messages.size() - 1);
        if (lastMessage.getType() == AgentState.Message.MessageType.AI
                && lastMessage.getToolCalls() != null
                && !lastMessage.getToolCalls().isEmpty()) {
            log.info("Route | reporter -> permission_tools | tool_calls={}", lastMessage.getToolCalls().size());
            return "PERMISSION_TOOLS";
        }

        log.info("Route | reporter -> END | reason=no_tool_calls");
        return "FINISH";
    }
}