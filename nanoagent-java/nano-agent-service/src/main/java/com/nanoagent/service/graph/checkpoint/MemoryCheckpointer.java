package com.nanoagent.service.graph.checkpoint;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MemoryCheckpointer implements GraphCheckpointer {

    private static final Logger log = LoggerFactory.getLogger(MemoryCheckpointer.class);

    private final Map<String, Map<String, Object>> store = new ConcurrentHashMap<>();

    @Override
    public void put(String threadId, Map<String, Object> state) {
        store.put(threadId, state);
    }

    @Override
    public Map<String, Object> get(String threadId) {
        return store.get(threadId);
    }

    @Override
    public void delete(String threadId) {
        store.remove(threadId);
    }

    @Override
    public boolean exists(String threadId) {
        return store.containsKey(threadId);
    }

    @Override
    public String getBackendName() {
        return "memory";
    }

    @Override
    public void close() {
        store.clear();
        log.info("Memory checkpointer closed");
    }
}