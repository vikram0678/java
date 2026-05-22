package com.mlops.sentiment.model;

import java.util.List;

public class BatchResponse {
    private int total;
    private List<SentimentResponse> results;

    public BatchResponse(int total, List<SentimentResponse> results) {
        this.total = total;
        this.results = results;
    }

    public int getTotal() { return total; }
    public List<SentimentResponse> getResults() { return results; }
}