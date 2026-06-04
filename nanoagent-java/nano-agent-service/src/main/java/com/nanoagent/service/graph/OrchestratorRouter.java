package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class OrchestratorRouter implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(OrchestratorRouter.class);

    @Override
    @SuppressWarnings("unchecked")
    public String apply(OverAllState state) {
        String orchestratorDecision = (String) state.value("orchestratorDecision").orElse("");

        if (orchestratorDecision == null || orchestratorDecision.isBlank()) {
            List<AgentState.Message> messages = (List<AgentState.Message>) state.value("messages").orElse(List.of());
            if (!messages.isEmpty()) {
                AgentState.Message last = messages.get(messages.size() - 1);
                if (last.getType() == AgentState.Message.MessageType.AI) {
                    orchestratorDecision = last.getContent();
                }
            }
        }

        OrchestratorDecision decision = OrchestratorDecision.fromText(orchestratorDecision);

        String route = switch (decision) {
            case DATA_ANALYST -> "DATA_ANALYST";
            case REPORTER -> "REPORTER";
            case ASSISTANT -> "ASSISTANT";
            case FINISH -> "FINISH";
        };

        log.info("Router | orchestrator -> {}", route);

        return route;
    }
}
