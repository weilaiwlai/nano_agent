package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class RouteAfterAssistant implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterAssistant.class);

    @Override
    @SuppressWarnings("unchecked")
    public String apply(OverAllState state) {
        String activeSkill = (String) state.value("activeSkill").orElse("");
        List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());

        if (messages.isEmpty()) {
            log.info("Route | assistant -> END | reason=no_messages");
            return "FINISH";
        }

        AgentState.Message lastMessage = messages.get(messages.size() - 1);
        if (lastMessage.getType() == AgentState.Message.MessageType.AI
                && lastMessage.getToolCalls() != null
                && !lastMessage.getToolCalls().isEmpty()) {
            log.info("Route | assistant -> skills_tools | tool_calls={}", lastMessage.getToolCalls().size());
            return "SKILLS_TOOLS";
        }

        if (activeSkill != null && !activeSkill.isBlank()) {
            log.info("Route | assistant -> skills_tools | active_skill={}", activeSkill);
            return "SKILLS_TOOLS";
        }

        log.info("Route | assistant -> END");
        return "FINISH";
    }
}