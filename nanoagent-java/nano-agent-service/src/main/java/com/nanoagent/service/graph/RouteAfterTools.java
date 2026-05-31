package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RouteAfterTools implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterTools.class);

    @Override
    public String apply(OverAllState state) {
        String sender = (String) state.value("sender").orElse("");
        log.info("Route | tools_node -> {} | sender={}", sender.isBlank() ? "knowledge_worker" : "knowledge_worker", sender);
        return "KNOWLEDGE_WORKER";
    }
}