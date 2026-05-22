package com.mlops.sentiment.controller;

import com.mlops.sentiment.model.SentimentRequest;
import com.mlops.sentiment.model.SentimentResponse;
import com.mlops.sentiment.model.BatchRequest;
import com.mlops.sentiment.model.BatchResponse;
import com.mlops.sentiment.service.InferenceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/v1/sentiment")
public class SentimentController {

    @Autowired
    private InferenceService inferenceService;

    // Single prediction endpoint
    @PostMapping("/predict")
    public ResponseEntity<SentimentResponse> predict(@RequestBody SentimentRequest request) {
        String label = inferenceService.predict(request.getText());
        return ResponseEntity.ok(new SentimentResponse(request.getText(), label));
    }

    // Batch endpoint - accepts JSON list of texts
    @PostMapping("/batch")
    public ResponseEntity<BatchResponse> batch(@RequestBody BatchRequest request) {
        List<String> texts = request.getTexts();
        List<String> labels = inferenceService.predictBatch(texts);

        List<SentimentResponse> results = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            results.add(new SentimentResponse(texts.get(i), labels.get(i)));
        }

        return ResponseEntity.ok(new BatchResponse(results.size(), results));
    }

    // Batch endpoint - accepts CSV file upload
    @PostMapping("/batch/csv")
    public ResponseEntity<BatchResponse> batchCsv(@RequestParam("file") MultipartFile file) {
        List<String> texts = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(file.getInputStream()))) {

            String line;
            boolean firstLine = true;
            while ((line = reader.readLine()) != null) {
                if (firstLine) { firstLine = false; continue; } // skip header
                if (!line.trim().isEmpty()) {
                    texts.add(line.trim());
                }
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }

        List<String> labels = inferenceService.predictBatch(texts);

        List<SentimentResponse> results = new ArrayList<>();
        for (int i = 0; i < texts.size(); i++) {
            results.add(new SentimentResponse(texts.get(i), labels.get(i)));
        }

        return ResponseEntity.ok(new BatchResponse(results.size(), results));
    }

    // Health check
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("{\"status\":\"healthy\"}");
    }
}