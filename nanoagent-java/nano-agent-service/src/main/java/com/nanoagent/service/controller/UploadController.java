package com.nanoagent.service.controller;

import com.nanoagent.service.auth.AuthContext;
import com.nanoagent.service.auth.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

@RestController
@RequestMapping("/api/v1")
public class UploadController {

    private static final Logger log = LoggerFactory.getLogger(UploadController.class);

    private static final Set<String> ALLOWED_TYPES = Set.of(
            "application/pdf", "text/plain", "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv", "image/jpeg", "image/png"
    );

    private static final long MAX_SIZE = 10 * 1024 * 1024;

    private final AuthService authService;

    public UploadController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/upload")
    public Map<String, Object> uploadFile(@RequestParam("file") MultipartFile file,
                                           HttpServletRequest httpRequest) {
        AuthContext authContext = authService.authenticate(httpRequest);
        String tokenSubject = authContext.requireSubject();
        String userId = httpRequest.getParameter("user_id");
        if (userId == null || userId.isBlank()) {
            userId = tokenSubject;
        }
        String resolvedUserId = authService.resolveEffectiveUserId(tokenSubject, userId);

        if (file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "未找到上传的文件");
        }

        String contentType = file.getContentType();
        if (contentType == null || !ALLOWED_TYPES.contains(contentType)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "不允许的文件类型: " + contentType);
        }

        if (file.getSize() > MAX_SIZE) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "文件大小超过限制 (最大10MB)");
        }

        try {
            Path uploadDir = Paths.get("./uploads", resolvedUserId);
            Files.createDirectories(uploadDir);

            String filename = file.getOriginalFilename();
            if (filename == null || filename.isBlank()) {
                filename = "uploaded_file";
            }
            Path filePath = uploadDir.resolve(filename);
            file.transferTo(filePath.toFile());

            log.info("文件上传成功 | user_id={} | file_path={} | size={}",
                    resolvedUserId, filePath, file.getSize());

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("status", "success");
            result.put("message", "文件上传成功");
            result.put("file_path", filePath.toString());
            result.put("filename", filename);
            result.put("size", file.getSize());
            result.put("content_type", contentType);
            return result;

        } catch (IOException e) {
            log.error("文件上传失败 | user_id={} | error={}", resolvedUserId, e.getMessage());
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "文件上传失败");
        }
    }
}