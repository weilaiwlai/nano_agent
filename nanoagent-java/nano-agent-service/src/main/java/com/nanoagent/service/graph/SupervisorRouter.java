package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class SupervisorRouter implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(SupervisorRouter.class);

    @Override
    @SuppressWarnings("unchecked")
    public String apply(OverAllState state) {
        String supervisorDecision = (String) state.value("supervisorDecision").orElse("");

        if (supervisorDecision == null || supervisorDecision.isBlank()) {
            List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
            if (!messages.isEmpty()) {
                AgentState.Message last = messages.get(messages.size() - 1);
                if (last.getType() == AgentState.Message.MessageType.AI) {
                    supervisorDecision = last.getContent();
                }
            }
        }

        SupervisorDecision decision = SupervisorDecision.fromText(supervisorDecision);

        String route = switch (decision) {
            case KNOWLEDGE_WORKER -> "KNOWLEDGE_WORKER";
            case REPORTER -> "REPORTER";
            case ASSISTANT -> "ASSISTANT";
            case FINISH -> "FINISH";
        };

        log.info("Router | supervisor -> {}", route);

        return route;
    }
}