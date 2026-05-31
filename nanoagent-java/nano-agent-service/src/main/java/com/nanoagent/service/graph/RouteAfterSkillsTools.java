package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class RouteAfterSkillsTools implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterSkillsTools.class);

    @Override
    @SuppressWarnings("unchecked")
    public String apply(OverAllState state) {
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
        if (!messages.isEmpty()) {
            AgentState.Message lastMessage = messages.get(messages.size() - 1);
            if (lastMessage.getType() == AgentState.Message.MessageType.AI
                    && lastMessage.getToolCalls() != null
                    && !lastMessage.getToolCalls().isEmpty()) {
                log.info("Route | skills_tools -> skill_tools_executor | tool_calls={}", lastMessage.getToolCalls().size());
                return "SKILL_TOOLS_EXECUTOR";
            }
        }

        log.info("Route | skills_tools -> assistant");
        return "ASSISTANT";
    }
}