package com.nanoagent.service.graph.checkpoint;

import com.alibaba.cloud.ai.graph.OverAllState;

import java.util.Map;

public interface GraphCheckpointer extends AutoCloseable {

    void put(String threadId, Map<String, Object> state);

    Map<String, Object> get(String threadId);

    void delete(String threadId);

    boolean exists(String threadId);

    String getBackendName();

    @Override
    default void close() throws Exception {}
}