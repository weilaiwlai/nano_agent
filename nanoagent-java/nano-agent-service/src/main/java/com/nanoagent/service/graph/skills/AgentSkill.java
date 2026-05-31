package com.nanoagent.service.graph.skills;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AgentSkill {

    private static final Pattern FRONTMATTER_PATTERN = Pattern.compile(
            "^---\\s*\\n(.*?)\\n---\\s*\\n(.*)", Pattern.DOTALL);

    private final Path rootPath;
    private final Path skillFile;
    private String name;
    private String description;
    private String version;
    private String author;
    private String instructions;

    public AgentSkill(Path rootPath) throws IOException {
        this.rootPath = rootPath;
        this.skillFile = rootPath.resolve("SKILL.md");
        load();
    }

    private void load() throws IOException {
        if (!Files.exists(skillFile)) {
            throw new IOException("Missing SKILL.md in " + rootPath);
        }

        String content = Files.readString(skillFile, StandardCharsets.UTF_8);
        Matcher matcher = FRONTMATTER_PATTERN.matcher(content);

        if (matcher.find()) {
            String frontmatter = matcher.group(1);
            this.instructions = matcher.group(2).trim();

            Map<String, String> metadata = parseYamlSimple(frontmatter);
            this.name = metadata.getOrDefault("name", rootPath.getFileName().toString());
            this.description = metadata.getOrDefault("description", "No description provided.");
            this.version = metadata.getOrDefault("version", "1.0");
            this.author = metadata.getOrDefault("author", "Unknown");
        } else {
            this.name = rootPath.getFileName().toString();
            this.description = "No description provided.";
            this.version = "1.0";
            this.author = "Unknown";
            this.instructions = content.trim();
        }
    }

    private Map<String, String> parseYamlSimple(String yaml) {
        Map<String, String> result = new LinkedHashMap<>();
        for (String line : yaml.split("\n")) {
            int colonIdx = line.indexOf(':');
            if (colonIdx > 0) {
                String key = line.substring(0, colonIdx).trim();
                String value = line.substring(colonIdx + 1).trim();
                if (value.startsWith("\"") && value.endsWith("\"")) {
                    value = value.substring(1, value.length() - 1);
                }
                result.put(key, value);
            }
        }
        return result;
    }

    public Path getRootPath() { return rootPath; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public String getVersion() { return version; }
    public String getAuthor() { return author; }
    public String getInstructions() { return instructions; }

    public Map<String, String> getMetadata() {
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("name", name);
        meta.put("description", description);
        meta.put("version", version);
        meta.put("author", author);
        return meta;
    }
}