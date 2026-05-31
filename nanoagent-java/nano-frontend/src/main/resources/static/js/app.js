(function () {
    'use strict';

    const API_BASE = window.AGENT_API_BASE_URL || 'http://localhost:8080';
    const API_TOKEN = window.AGENT_API_TOKEN || '';

    function api(path, options = {}) {
        const headers = { 'Content-Type': 'application/json', ...options.headers };
        if (API_TOKEN) headers['Authorization'] = 'Bearer ' + API_TOKEN;
        return fetch(API_BASE + path, { ...options, headers });
    }

    // ==================== State ====================
    const state = {
        userId: localStorage.getItem('nano_user_id') || 'user_001',
        conversations: [],
        activeConvId: null,
        messages: [],
        isStreaming: false,
        abortController: null,
        pendingInterrupt: null,
        providers: [],
        models: [],
        activeSession: null,
        memories: [],
        backendHealthy: false,
        healthChecked: false
    };

    // ==================== DOM Refs ====================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const dom = {
        sidebar: $('#sidebar'),
        sidebarToggle: $('#sidebarToggle'),
        healthIndicator: $('#healthIndicator'),
        userIdInput: $('#userIdInput'),
        userIdHint: $('#userIdHint'),
        newConvBtn: $('#newConvBtn'),
        convList: $('#convList'),
        deleteConvBtn: $('#deleteConvBtn'),
        providerSelect: $('#providerSelect'),
        modelSelect: $('#modelSelect'),
        customModelInput: $('#customModelInput'),
        apiKeyInput: $('#apiKeyInput'),
        baseUrlInput: $('#baseUrlInput'),
        baseUrlGroup: $('#baseUrlGroup'),
        embeddingModelInput: $('#embeddingModelInput'),
        validateBtn: $('#validateBtn'),
        activateBtn: $('#activateBtn'),
        validationResult: $('#validationResult'),
        activeSessionInfo: $('#activeSessionInfo'),
        refreshProvidersBtn: $('#refreshProvidersBtn'),
        memoryText: $('#memoryText'),
        saveMemoryBtn: $('#saveMemoryBtn'),
        refreshMemoryBtn: $('#refreshMemoryBtn'),
        memoryList: $('#memoryList'),
        chatMessages: $('#chatMessages'),
        chatInput: $('#chatInput'),
        sendBtn: $('#sendBtn'),
        stopBtn: $('#stopBtn'),
        interruptPanel: $('#interruptPanel'),
        interruptCalls: $('#interruptCalls'),
        approveBtn: $('#approveBtn'),
        rejectBtn: $('#rejectBtn'),
        chatUserDisplay: $('#chatUserDisplay'),
        chatSessionDisplay: $('#chatSessionDisplay')
    };

    // ==================== Health Check ====================
    async function checkHealth() {
        try {
            const resp = await fetch(API_BASE + '/health', { signal: AbortSignal.timeout(5000) });
            state.backendHealthy = resp.ok;
        } catch {
            state.backendHealthy = false;
        }
        state.healthChecked = true;
        updateHealthUI();
    }

    function updateHealthUI() {
        const dot = dom.healthIndicator.querySelector('.health-dot');
        const text = dom.healthIndicator.querySelector('.health-text');
        dot.className = 'health-dot ' + (state.backendHealthy ? 'healthy' : 'unhealthy');
        text.textContent = state.backendHealthy
            ? '后端服务正常 (' + API_BASE + ')'
            : '后端服务不可达 (' + API_BASE + ')';
        setInputsEnabled(state.backendHealthy);
    }

    function setInputsEnabled(enabled) {
        dom.chatInput.disabled = !enabled;
        dom.sendBtn.disabled = !enabled;
        if (!enabled) {
            dom.chatInput.placeholder = '后端服务不可达，请检查服务状态...';
        } else {
            dom.chatInput.placeholder = '请输入你的问题...';
        }
    }

    // ==================== User ID ====================
    function initUserId() {
        dom.userIdInput.value = state.userId;
        dom.userIdInput.addEventListener('change', () => {
            state.userId = dom.userIdInput.value.trim() || 'user_001';
            localStorage.setItem('nano_user_id', state.userId);
            dom.userIdHint.textContent = '手动用户 ID 模式（本地调试）';
            updateChatMeta();
            loadConversations();
            loadMemories();
        });
    }

    function updateChatMeta() {
        dom.chatUserDisplay.textContent = '用户: ' + state.userId;
        dom.chatSessionDisplay.textContent = state.activeConvId
            ? '会话: ' + state.activeConvId.substring(0, 8) + '...'
            : '';
    }

    // ==================== Sidebar ====================
    function initSidebar() {
        dom.sidebarToggle.addEventListener('click', () => {
            dom.sidebar.classList.toggle('collapsed');
            dom.sidebarToggle.textContent = dom.sidebar.classList.contains('collapsed') ? '▶' : '☰';
        });

        $$('.section-header').forEach(header => {
            header.addEventListener('click', () => {
                const panelId = header.dataset.panel;
                const panel = $('#' + panelId);
                const arrow = header.querySelector('.section-arrow');
                panel.classList.toggle('collapsed');
                arrow.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
            });
        });
    }

    // ==================== Conversations ====================
    async function loadConversations() {
        try {
            const resp = await api('/api/conversations?user_id=' + encodeURIComponent(state.userId));
            if (!resp.ok) throw new Error('Failed to load conversations');
            const data = await resp.json();
            state.conversations = data.conversations || [];
            renderConvList();
        } catch (err) {
            console.error('Load conversations error:', err);
        }
    }

    function renderConvList() {
        dom.convList.innerHTML = '';
        state.conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conv-item' + (conv.id === state.activeConvId ? ' active' : '');
            item.innerHTML = '<span class="conv-item-title">' + escapeHtml(conv.title || '新对话') + '</span>'
                + '<span class="conv-item-badge">' + (conv.message_count || 0) + '</span>';
            item.addEventListener('click', () => switchConversation(conv.id));
            dom.convList.appendChild(item);
        });
        dom.deleteConvBtn.style.display = state.activeConvId ? 'block' : 'none';
    }

    async function switchConversation(convId) {
        state.activeConvId = convId;
        state.messages = [];
        dom.chatMessages.innerHTML = '';
        renderConvList();
        updateChatMeta();

        try {
            const resp = await api('/api/conversations/' + convId + '/messages?user_id=' + encodeURIComponent(state.userId));
            if (resp.ok) {
                const data = await resp.json();
                state.messages = data.messages || [];
                state.messages.forEach(msg => renderMessage(msg));
            }
        } catch (err) {
            console.error('Load messages error:', err);
        }
        scrollToBottom();
    }

    async function createConversation() {
        try {
            const resp = await api('/api/conversations', {
                method: 'POST',
                body: JSON.stringify({ user_id: state.userId, title: '新对话' })
            });
            if (resp.ok) {
                const data = await resp.json();
                state.activeConvId = data.id;
                state.messages = [];
                dom.chatMessages.innerHTML = '<div class="welcome-message"><div class="welcome-icon">🤖</div><h3>新对话已创建</h3><p>开始提问吧！</p></div>';
                await loadConversations();
                updateChatMeta();
            }
        } catch (err) {
            console.error('Create conversation error:', err);
        }
    }

    async function deleteConversation() {
        if (!state.activeConvId) return;
        if (!confirm('确定要删除当前对话吗？')) return;
        try {
            await api('/api/conversations/' + state.activeConvId + '?user_id=' + encodeURIComponent(state.userId), {
                method: 'DELETE'
            });
            state.activeConvId = null;
            state.messages = [];
            dom.chatMessages.innerHTML = '<div class="welcome-message"><div class="welcome-icon">🤖</div><h3>欢迎使用 NanoAgent</h3><p>多智能体协作系统，支持知识检索、报告生成、数据分析等功能。</p></div>';
            await loadConversations();
            updateChatMeta();
        } catch (err) {
            console.error('Delete conversation error:', err);
        }
    }

    // ==================== AI Config ====================
    async function loadProviders() {
        try {
            const resp = await api('/api/llm/providers');
            if (!resp.ok) throw new Error('Failed to load providers');
            const data = await resp.json();
            state.providers = data.providers || [];
            renderProviderSelect();
        } catch (err) {
            console.error('Load providers error:', err);
        }
    }

    function renderProviderSelect() {
        dom.providerSelect.innerHTML = '<option value="">选择提供商...</option>';
        state.providers.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.name;
            dom.providerSelect.appendChild(opt);
        });
    }

    dom.providerSelect.addEventListener('change', async () => {
        const providerId = dom.providerSelect.value;
        dom.modelSelect.innerHTML = '<option value="">选择模型...</option>';
        dom.customModelInput.style.display = 'none';
        dom.baseUrlGroup.style.display = 'none';

        if (!providerId) return;

        const provider = state.providers.find(p => p.id === providerId);
        if (provider) {
            if (provider.models && provider.models.length > 0) {
                provider.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    dom.modelSelect.appendChild(opt);
                });
                const customOpt = document.createElement('option');
                customOpt.value = '__custom__';
                customOpt.textContent = '自定义模型...';
                dom.modelSelect.appendChild(customOpt);
            }
            if (provider.requires_base_url) {
                dom.baseUrlGroup.style.display = 'block';
            }
        }
    });

    dom.modelSelect.addEventListener('change', () => {
        dom.customModelInput.style.display = dom.modelSelect.value === '__custom__' ? 'block' : 'none';
    });

    async function validateConnection() {
        const provider = dom.providerSelect.value;
        const model = dom.modelSelect.value === '__custom__' ? dom.customModelInput.value.trim() : dom.modelSelect.value;
        const apiKey = dom.apiKeyInput.value.trim();
        const baseUrl = dom.baseUrlInput.value.trim();

        if (!provider || !model || !apiKey) {
            showValidationResult('请填写完整的配置信息', 'error');
            return;
        }

        dom.validateBtn.disabled = true;
        dom.validateBtn.textContent = '验证中...';
        dom.validationResult.className = 'validation-result';
        dom.validationResult.style.display = 'none';

        try {
            const resp = await api('/api/llm/validate', {
                method: 'POST',
                body: JSON.stringify({ provider, model, api_key: apiKey, base_url: baseUrl || undefined })
            });
            const data = await resp.json();
            if (resp.ok && data.valid) {
                showValidationResult('连接验证成功！模型可用。', 'success');
            } else {
                showValidationResult('验证失败: ' + (data.error || '未知错误'), 'error');
            }
        } catch (err) {
            showValidationResult('验证请求失败: ' + err.message, 'error');
        } finally {
            dom.validateBtn.disabled = false;
            dom.validateBtn.textContent = '验证连接';
        }
    }

    async function activateSession() {
        const provider = dom.providerSelect.value;
        const model = dom.modelSelect.value === '__custom__' ? dom.customModelInput.value.trim() : dom.modelSelect.value;
        const apiKey = dom.apiKeyInput.value.trim();
        const baseUrl = dom.baseUrlInput.value.trim();
        const embeddingModel = dom.embeddingModelInput.value.trim();

        if (!provider || !model || !apiKey) {
            showValidationResult('请填写完整的配置信息', 'error');
            return;
        }

        dom.activateBtn.disabled = true;
        dom.activateBtn.textContent = '保存中...';

        try {
            const resp = await api('/api/llm/sessions', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: state.userId,
                    provider,
                    model,
                    api_key: apiKey,
                    base_url: baseUrl || undefined,
                    embedding_model: embeddingModel || undefined
                })
            });
            const data = await resp.json();
            if (resp.ok) {
                state.activeSession = data;
                showActiveSessionInfo(data);
                showValidationResult('配置已保存并激活！', 'success');
            } else {
                showValidationResult('保存失败: ' + (data.error || '未知错误'), 'error');
            }
        } catch (err) {
            showValidationResult('保存请求失败: ' + err.message, 'error');
        } finally {
            dom.activateBtn.disabled = false;
            dom.activateBtn.textContent = '保存并启用';
        }
    }

    function showValidationResult(msg, type) {
        dom.validationResult.textContent = msg;
        dom.validationResult.className = 'validation-result ' + type;
        dom.validationResult.style.display = 'block';
    }

    function showActiveSessionInfo(session) {
        dom.activeSessionInfo.innerHTML = '已激活: <strong>' + escapeHtml(session.provider) + ' / ' + escapeHtml(session.model) + '</strong>'
            + (session.embedding_model ? ' (Embedding: ' + escapeHtml(session.embedding_model) + ')' : '');
        dom.activeSessionInfo.style.display = 'block';
    }

    // ==================== Memory ====================
    async function loadMemories() {
        try {
            const resp = await api('/api/memory?user_id=' + encodeURIComponent(state.userId));
            if (!resp.ok) throw new Error('Failed to load memories');
            const data = await resp.json();
            state.memories = data.memories || [];
            renderMemoryList();
        } catch (err) {
            console.error('Load memories error:', err);
        }
    }

    function renderMemoryList() {
        dom.memoryList.innerHTML = '';
        state.memories.forEach((mem, idx) => {
            const item = document.createElement('div');
            item.className = 'memory-item';
            item.innerHTML = '<span class="memory-item-text">' + escapeHtml(mem.content || mem.text || '')
                + '</span><button class="memory-item-delete" data-idx="' + idx + '">✕</button>';
            item.querySelector('.memory-item-delete').addEventListener('click', () => deleteMemory(idx));
            dom.memoryList.appendChild(item);
        });
    }

    async function saveMemory() {
        const text = dom.memoryText.value.trim();
        if (!text) return;

        dom.saveMemoryBtn.disabled = true;
        try {
            await api('/api/memory', {
                method: 'POST',
                body: JSON.stringify({ user_id: state.userId, content: text })
            });
            dom.memoryText.value = '';
            await loadMemories();
        } catch (err) {
            console.error('Save memory error:', err);
        } finally {
            dom.saveMemoryBtn.disabled = false;
        }
    }

    async function deleteMemory(idx) {
        const mem = state.memories[idx];
        if (!mem) return;
        try {
            await api('/api/memory/' + (mem.id || idx) + '?user_id=' + encodeURIComponent(state.userId), {
                method: 'DELETE'
            });
            await loadMemories();
        } catch (err) {
            console.error('Delete memory error:', err);
        }
    }

    // ==================== Chat ====================
    function renderMessage(msg) {
        const isUser = msg.role === 'user';
        const div = document.createElement('div');
        div.className = 'message ' + (isUser ? 'user' : 'assistant');
        div.innerHTML = '<div class="message-avatar">' + (isUser ? '👤' : '🤖') + '</div>'
            + '<div><div class="message-bubble">' + (isUser ? escapeHtml(msg.content) : renderMarkdown(msg.content)) + '</div>'
            + '<div class="message-meta">' + formatTime(msg.timestamp) + '</div></div>';
        dom.chatMessages.appendChild(div);
    }

    // ==================== JWT Parsing ====================
    function parseJwtSubject(token) {
        if (!token) return '';
        const normalized = token.trim();
        const parts = normalized.split('.');
        if (parts.length !== 2 && parts.length !== 3) return '';
        try {
            const payloadSegment = parts[1];
            const padding = '='.repeat((4 - payloadSegment.length % 4) % 4);
            const payloadRaw = atob(payloadSegment.replace(/-/g, '+').replace(/_/g, '/') + padding);
            const payload = JSON.parse(payloadRaw);
            const subject = payload.sub;
            return typeof subject === 'string' ? subject.trim() : '';
        } catch (e) {
            return '';
        }
    }

    function updateUserIdFromToken() {
        if (API_TOKEN) {
            const subject = parseJwtSubject(API_TOKEN);
            if (subject) {
                state.userId = subject;
                dom.userIdInput.value = subject;
                dom.userIdHint.textContent = '从 JWT Token 自动解析（sub: ' + subject + '）';
                localStorage.setItem('nano_user_id', subject);
            }
        }
    }

    // ==================== Model Presets ====================
    const MODEL_PRESETS = {
        qwen: ['qwen3.5-plus', 'qwen-plus-latest', 'qwen-max-latest'],
        openai: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini'],
        deepseek: ['deepseek-chat', 'deepseek-reasoner'],
        groq: ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
        other: ['自定义输入']
    };

    function getPresetModels(providerId) {
        return MODEL_PRESETS[providerId] || MODEL_PRESETS['other'];
    }

    // ==================== Thread Management ====================
    function getThreadId() {
        return state.activeConvId || (state.userId + '_' + Date.now().toString(36));
    }

    function updateThreadDisplay() {
        const threadId = getThreadId();
        dom.chatSessionDisplay.textContent = '线程: ' + threadId.substring(0, 8) + '...';
    }

    // ==================== DSML Processing ====================
    function processDsml(text) {
        if (!text) return text;

        const dsmlPattern = /<dsml\s+type="([^"]*)"(?:\s+[^>]*)?>([\s\S]*?)<\/dsml>/gi;
        let processed = text;

        processed = processed.replace(dsmlPattern, (match, type, content) => {
            switch (type) {
                case 'tool_call':
                    return '<div class="dsml-tool-call"><div class="dsml-tool-header">🔧 工具调用</div><pre>' + escapeHtml(content.trim()) + '</pre></div>';
                case 'tool_result':
                    return '<div class="dsml-tool-result"><div class="dsml-tool-header">📋 工具结果</div><pre>' + escapeHtml(content.trim()) + '</pre></div>';
                case 'thinking':
                    return '<div class="dsml-thinking"><div class="dsml-thinking-header">💭 思考过程</div><div>' + renderMarkdown(content.trim()) + '</div></div>';
                case 'error':
                    return '<div class="dsml-error"><div class="dsml-error-header">❌ 错误</div><pre>' + escapeHtml(content.trim()) + '</pre></div>';
                default:
                    return '<div class="dsml-block"><pre>' + escapeHtml(content.trim()) + '</pre></div>';
            }
        });

        return processed;
    }

    function renderMarkdown(text) {
        if (!text) return '';
        text = processDsml(text);
        let html = escapeHtml(text);

        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
        html = html.replace(/\n{2,}/g, '</p><p>');
        html = '<p>' + html + '</p>';

        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatTime(ts) {
        if (!ts) return '';
        const d = new Date(ts);
        return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    }

    function scrollToBottom() {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    }

    async function sendMessage() {
        if (state.isStreaming) return;
        const content = dom.chatInput.value.trim();
        if (!content) return;

        if (!state.activeConvId) {
            await createConversation();
            if (!state.activeConvId) return;
        }

        const userMsg = { role: 'user', content, timestamp: new Date().toISOString() };
        state.messages.push(userMsg);
        renderMessage(userMsg);
        dom.chatInput.value = '';
        dom.chatInput.style.height = 'auto';
        scrollToBottom();

        state.isStreaming = true;
        dom.sendBtn.style.display = 'none';
        dom.stopBtn.style.display = 'inline-block';
        dom.chatInput.disabled = true;

        const assistantMsg = { role: 'assistant', content: '', timestamp: new Date().toISOString() };
        state.messages.push(assistantMsg);

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.innerHTML = '<div class="message-avatar">🤖</div>'
            + '<div><div class="message-bubble streaming-cursor" id="streamingBubble"></div>'
            + '<div class="message-meta">' + formatTime(assistantMsg.timestamp) + '</div></div>';
        dom.chatMessages.appendChild(msgDiv);
        const bubble = msgDiv.querySelector('#streamingBubble');

        state.abortController = new AbortController();

        try {
            const resp = await fetch(API_BASE + '/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_TOKEN ? { 'Authorization': 'Bearer ' + API_TOKEN } : {})
                },
                body: JSON.stringify({
                    user_id: state.userId,
                    conversation_id: state.activeConvId,
                    message: content
                }),
                signal: state.abortController.signal
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.error || '请求失败 (' + resp.status + ')');
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6).trim();
                        if (data === '[DONE]') continue;

                        try {
                            const parsed = JSON.parse(data);

                            if (parsed.type === 'interrupt') {
                                handleInterrupt(parsed);
                                continue;
                            }

                            if (parsed.type === 'error') {
                                throw new Error(parsed.error || '流式错误');
                            }

                            const delta = parsed.content || parsed.delta || '';
                            assistantMsg.content += delta;
                            bubble.innerHTML = renderMarkdown(assistantMsg.content);
                            scrollToBottom();
                        } catch (parseErr) {
                            if (parseErr.message && !parseErr.message.includes('JSON')) {
                                throw parseErr;
                            }
                        }
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                assistantMsg.content += '\n\n*[已暂停]*';
            } else {
                assistantMsg.content += '\n\n*[错误: ' + escapeHtml(err.message) + ']*';
            }
            bubble.innerHTML = renderMarkdown(assistantMsg.content);
        } finally {
            bubble.classList.remove('streaming-cursor');
            bubble.removeAttribute('id');
            state.isStreaming = false;
            state.abortController = null;
            dom.sendBtn.style.display = 'inline-block';
            dom.stopBtn.style.display = 'none';
            dom.chatInput.disabled = !state.backendHealthy;
            dom.chatInput.focus();
            scrollToBottom();
            await loadConversations();
        }
    }

    function stopStreaming() {
        if (state.abortController) {
            state.abortController.abort();
        }
    }

    // ==================== Interrupt Handling ====================
    function handleInterrupt(data) {
        state.pendingInterrupt = data;
        dom.interruptPanel.style.display = 'block';
        dom.interruptCalls.innerHTML = '';

        const calls = data.tool_calls || data.calls || [];
        calls.forEach(call => {
            const div = document.createElement('div');
            div.className = 'interrupt-call';
            div.innerHTML = '<div class="interrupt-call-name">🔧 ' + escapeHtml(call.name || call.function_name || 'unknown') + '</div>'
                + '<div class="interrupt-call-args">' + escapeHtml(JSON.stringify(call.arguments || call.args || {}, null, 2)) + '</div>';
            dom.interruptCalls.appendChild(div);
        });

        dom.chatInput.disabled = true;
        dom.sendBtn.disabled = true;
    }

    async function approveInterrupt() {
        if (!state.pendingInterrupt) return;
        dom.interruptPanel.style.display = 'none';

        try {
            await api('/api/chat/approve', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: state.userId,
                    conversation_id: state.activeConvId,
                    interrupt_id: state.pendingInterrupt.interrupt_id || state.pendingInterrupt.id,
                    approved: true
                })
            });
        } catch (err) {
            console.error('Approve error:', err);
        }

        state.pendingInterrupt = null;
        dom.chatInput.disabled = !state.backendHealthy;
        dom.sendBtn.disabled = !state.backendHealthy;
    }

    async function rejectInterrupt() {
        if (!state.pendingInterrupt) return;
        dom.interruptPanel.style.display = 'none';

        try {
            await api('/api/chat/approve', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: state.userId,
                    conversation_id: state.activeConvId,
                    interrupt_id: state.pendingInterrupt.interrupt_id || state.pendingInterrupt.id,
                    approved: false
                })
            });
        } catch (err) {
            console.error('Reject error:', err);
        }

        state.pendingInterrupt = null;
        dom.chatInput.disabled = !state.backendHealthy;
        dom.sendBtn.disabled = !state.backendHealthy;
    }

    // ==================== Event Bindings ====================
    function bindEvents() {
        dom.newConvBtn.addEventListener('click', createConversation);
        dom.deleteConvBtn.addEventListener('click', deleteConversation);
        dom.sendBtn.addEventListener('click', sendMessage);
        dom.stopBtn.addEventListener('click', stopStreaming);
        dom.approveBtn.addEventListener('click', approveInterrupt);
        dom.rejectBtn.addEventListener('click', rejectInterrupt);
        dom.validateBtn.addEventListener('click', validateConnection);
        dom.activateBtn.addEventListener('click', activateSession);
        dom.saveMemoryBtn.addEventListener('click', saveMemory);
        dom.refreshMemoryBtn.addEventListener('click', loadMemories);
        dom.refreshProvidersBtn.addEventListener('click', loadProviders);

        dom.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        dom.chatInput.addEventListener('input', () => {
            dom.chatInput.style.height = 'auto';
            dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 150) + 'px';
        });
    }

    // ==================== Init ====================
    async function init() {
        bindEvents();
        initSidebar();
        initUserId();
        updateUserIdFromToken();
        updateChatMeta();
        updateThreadDisplay();
        checkHealth();
        loadConversations();
        loadProviders();
        loadMemories();
    }

    document.addEventListener('DOMContentLoaded', init);
})();