package com.nanoagent.service.graph;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.EdgeAction;
import com.nanoagent.service.graph.skills.SkillRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;

public class RouteAfterAssistant implements EdgeAction {

    private static final Logger log = LoggerFactory.getLogger(RouteAfterAssistant.class);
    private final SkillRegistry skillRegistry = new SkillRegistry();

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
            log.info("Route | assistant -> safe_tools | tool_calls={}", lastMessage.getToolCalls().size());
            return "SAFE_TOOLS";
        }

        // 检查是否是技能名称（文本内容）
        if (lastMessage.getType() == AgentState.Message.MessageType.AI) {
            String content = lastMessage.getContent() != null ? lastMessage.getContent().trim() : "";
            skillRegistry.refresh();
            List<String> skillNames = skillRegistry.listSkills().stream()
                    .map(s -> s.get("name"))
                    .collect(Collectors.toList());
            if (skillNames.contains(content)) {
                log.info("Route | assistant -> safe_tools | skill_name={}", content);
                return "SAFE_TOOLS";
            }
        }

        log.info("Route | assistant -> END");
        return "FINISH";
    }
}
