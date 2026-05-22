package com.mlops.sentiment.model;

import java.util.List;

public class BatchRequest {
    private List<String> texts;

    public List<String> getTexts() { return texts; }
    public void setTexts(List<String> texts) { this.texts = texts; }
}