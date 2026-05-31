package com.nanoagent.service.graph.skills;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class SkillTools {

    private static final Logger log = LoggerFactory.getLogger(SkillTools.class);

    private static Path activeSkillPath;

    public static void setActivePath(Path path) {
        activeSkillPath = path;
        if (path != null) {
            log.info("Context switched to skill path: {}", path);
        }
    }

    public static Path getActivePath() {
        return activeSkillPath;
    }

    public static String runSkillScript(String scriptName, List<String> args) {
        log.info("Tool Call: run_skill_script | Script: {} | Args: {}", scriptName, args);

        if (activeSkillPath == null) {
            String msg = "Error: No active skill path set.";
            log.error(msg);
            return msg;
        }

        Path scriptsDir = activeSkillPath.resolve("scripts").toAbsolutePath().normalize();
        Path scriptFile = scriptsDir.resolve(scriptName).toAbsolutePath().normalize();

        if (!Files.exists(scriptsDir)) {
            return "Error: Directory not found: " + scriptsDir;
        }

        if (!Files.exists(scriptFile)) {
            List<String> existing = new ArrayList<>();
            try {
                Files.list(scriptsDir).forEach(f -> existing.add(f.getFileName().toString()));
            } catch (IOException ignored) {}
            return "Error: Script not found: " + scriptFile + ". Existing files: " + existing;
        }

        try {
            List<String> cmd = new ArrayList<>();
            cmd.add("python");
            cmd.add(scriptFile.toString());
            if (args != null) {
                cmd.addAll(args);
            }

            log.debug("Executing command: {}", String.join(" ", cmd));

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.directory(scriptsDir.toFile());
            pb.redirectErrorStream(true);

            Map<String, String> env = pb.environment();
            env.putAll(System.getenv());

            Process process = pb.start();

            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            boolean finished = process.waitFor(60, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                return "Error: Script execution timed out (60s).";
            }

            String result = output.toString().trim();
            if (result.isEmpty()) {
                result = "[No Output]";
            }

            if (process.exitValue() == 0) {
                log.info("Script Success. Output length: {}", result.length());
                return "Success:\n" + result;
            } else {
                log.error("Script Failed (Code {})", process.exitValue());
                return "Error (Code " + process.exitValue() + "):\n" + result;
            }

        } catch (Exception e) {
            log.error("Exception in run_skill_script: {}", e.getMessage(), e);
            return "System Execution Error: " + e.getMessage();
        }
    }

    public static String readReference(String filename) {
        if (activeSkillPath == null) {
            return "Error: No active skill context.";
        }

        Path refDir = activeSkillPath.resolve("references");
        Path filePath = refDir.resolve(filename);

        log.info("Tool Call: read_reference | File: {}", filename);

        if (!Files.exists(refDir)) {
            return "Error: This skill has no 'references' folder.";
        }

        if (!Files.exists(filePath)) {
            List<String> existing = new ArrayList<>();
            try {
                Files.list(refDir)
                        .filter(Files::isRegularFile)
                        .forEach(f -> existing.add(f.getFileName().toString()));
            } catch (IOException ignored) {}
            return "Error: File '" + filename + "' not found. Available files: " + existing;
        }

        try {
            String content = Files.readString(filePath, StandardCharsets.UTF_8);
            if (content.length() > 10000) {
                return content.substring(0, 10000) + "\n\n[Content truncated because it is too long]";
            }
            return content;
        } catch (IOException e) {
            return "Error reading file: " + e.getMessage();
        }
    }
}