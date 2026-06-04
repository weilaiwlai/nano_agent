package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RouteAfterHighRiskTools implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterHighRiskTools.class);

    @Override
    public String apply(OverAllState state) {
        String currentAgent = (String) state.value("currentAgent").orElse("");

        if ("reporter".equals(currentAgent)) {
            log.info("Route | high_risk_tools -> reporter | current_agent={}", currentAgent);
            return "REPORTER";
        }

        // 默认回跳到 data_analyst
        log.info("Route | high_risk_tools -> data_analyst | current_agent={}", currentAgent);
        return "DATA_ANALYST";
    }
}
