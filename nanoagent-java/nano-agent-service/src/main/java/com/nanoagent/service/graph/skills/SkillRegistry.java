package com.nanoagent.service.graph.skills;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class SkillRegistry {

    private static final Logger log = LoggerFactory.getLogger(SkillRegistry.class);

    private static final Path SKILLS_ROOT = Paths.get("skills");

    private final Map<String, AgentSkill> skills = new ConcurrentHashMap<>();

    public SkillRegistry() {
        try {
            Files.createDirectories(SKILLS_ROOT);
        } catch (IOException e) {
            log.warn("Failed to create skills root directory: {}", e.getMessage());
        }
        refresh();
    }

    public void refresh() {
        skills.clear();
        if (!Files.exists(SKILLS_ROOT) || !Files.isDirectory(SKILLS_ROOT)) {
            return;
        }

        try {
            Files.list(SKILLS_ROOT).forEach(item -> {
                if (Files.isDirectory(item) && Files.exists(item.resolve("SKILL.md"))) {
                    try {
                        AgentSkill skill = new AgentSkill(item);
                        skills.put(skill.getName(), skill);
                        log.info("Loaded skill: {}", skill.getName());
                    } catch (Exception e) {
                        log.warn("Error loading skill {}: {}", item.getFileName(), e.getMessage());
                    }
                }
            });
        } catch (IOException e) {
            log.warn("Error listing skills directory: {}", e.getMessage());
        }
    }

    public AgentSkill getSkill(String name) {
        return skills.get(name);
    }

    public List<Map<String, String>> listSkills() {
        List<Map<String, String>> result = new ArrayList<>();
        for (AgentSkill skill : skills.values()) {
            result.add(skill.getMetadata());
        }
        return result;
    }

    public boolean isEmpty() {
        return skills.isEmpty();
    }

    public int size() {
        return skills.size();
    }
}