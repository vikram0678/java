package com.mlops.sentiment.service;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.huggingface.translator.TextClassificationTranslatorFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.nio.file.Paths;
import java.util.List;
import java.util.ArrayList;

@Service
public class InferenceService {

    @Value("${model.path:/app/model_output}")
    private String modelPath;

    private ZooModel<String, Classifications> model;
    private Predictor<String, Classifications> predictor;

    @PostConstruct
    public void init() throws Exception {
        Criteria<String, Classifications> criteria = Criteria.builder()
                .setTypes(String.class, Classifications.class)
                .optModelPath(Paths.get(modelPath))
                .optEngine("PyTorch")
                .optTranslatorFactory(new TextClassificationTranslatorFactory())
                .build();

        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
        System.out.println(">>> SUCCESS: Model loaded with TextClassificationTranslatorFactory!");
    }

    // Single prediction
    public String predict(String text) {
        try {
            Classifications classifications = predictor.predict(text);
            return classifications.best().getClassName();
        } catch (Exception e) {
            return "Error during JVM inference: " + e.getMessage();
        }
    }

    // Batch prediction - processes a list of texts at once
    public List<String> predictBatch(List<String> texts) {
        List<String> results = new ArrayList<>();
        for (String text : texts) {
            results.add(predict(text));
        }
        return results;
    }

    @PreDestroy
    public void cleanup() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
    }
}