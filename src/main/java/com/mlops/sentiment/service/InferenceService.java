package com.mlops.sentiment.service;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import java.nio.file.Paths;

@Service
public class InferenceService {

    @Value("${model.path:/app/model_output}")
    private String modelPath;

    private ZooModel<String, Classifications> model;
    private Predictor<String, Classifications> predictor;

    @PostConstruct
    public void init() throws Exception {
        // We configure DJL to explicitly handle HuggingFace Safetensors 
        Criteria<String, Classifications> criteria = Criteria.builder()
                .setTypes(String.class, Classifications.class)
                .optModelPath(Paths.get(modelPath))
                .optEngine("PyTorch")
                // This option forces DJL to look for the model.safetensors file instead of a .pt file
                .optArgument("model_type", "safetensors") 
                .build();
                
        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
        System.out.println(">>> SUCCESS: Real model.safetensors loaded into Java JVM memory!");
    }

    public String predict(String text) {
        try {
            Classifications classifications = predictor.predict(text);
            return classifications.best().getClassName();
        } catch (Exception e) {
            return "Error during JVM inference: " + e.getMessage();
        }
    }

    @PreDestroy
    public void cleanup() {
        if (predictor != null) predictor.close();
        if (model != null) model.close();
    }
}
