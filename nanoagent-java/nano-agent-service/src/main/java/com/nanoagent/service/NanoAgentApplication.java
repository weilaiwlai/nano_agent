package com.nanoagent.service;

import org.springframework.ai.autoconfigure.openai.OpenAiAutoConfiguration;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(exclude = {
        OpenAiAutoConfiguration.class
})
public class NanoAgentApplication {

    public static void main(String[] args) {
        SpringApplication.run(NanoAgentApplication.class, args);
    }
}