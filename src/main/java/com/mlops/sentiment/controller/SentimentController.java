package com.mlops.sentiment.controller;

import com.mlops.sentiment.model.SentimentRequest;
import com.mlops.sentiment.model.SentimentResponse;
import com.mlops.sentiment.service.InferenceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/sentiment")
public class SentimentController {

    @Autowired
    private InferenceService inferenceService;

    @PostMapping("/predict")
    public ResponseEntity<SentimentResponse> predict(@RequestBody SentimentRequest request) {
        String label = inferenceService.predict(request.getText());
        return ResponseEntity.ok(new SentimentResponse(request.getText(), label));
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("{\"status\":\"healthy\"}");
    }
}
