package com.nanoagent.frontend.controller;

import com.nanoagent.frontend.config.FrontendProperties;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class PageController {

    private final FrontendProperties properties;
    public PageController(FrontendProperties properties) {
        this.properties = properties;
    }

    @GetMapping("/")
    public String index(Model model) {
        model.addAttribute("agentApiBaseUrl", properties.getAgentApiBaseUrl());
        model.addAttribute("agentApiToken", properties.getAgentApiToken());
        model.addAttribute("pageTitle", properties.getPageTitle());
        return "index";
    }
}