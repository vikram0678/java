package com.mlops.sentiment.model;

public class SentimentResponse {
    private String text;
    private String label;
    
    public SentimentResponse(String text, String label) {
        this.text = text;
        this.label = label;
    }
    public String getText() { return text; }
    public String getLabel() { return label; }
}
