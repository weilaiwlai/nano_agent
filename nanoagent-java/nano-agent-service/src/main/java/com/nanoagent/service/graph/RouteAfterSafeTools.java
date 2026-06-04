package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RouteAfterSafeTools implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterSafeTools.class);

    @Override
    public String apply(OverAllState state) {
        log.info("Route | safe_tools -> assistant");
        return "ASSISTANT";
    }
}
