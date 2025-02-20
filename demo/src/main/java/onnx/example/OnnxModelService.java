package onnx.example;

import ai.onnxruntime.*;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.ObjectMapper;




public class OnnxModelService {
    private OrtEnvironment env;
    private OrtSession session;


    public OnnxModelService() throws OrtException, IOException {
        // Initialize ONNX Runtime environment
        env = OrtEnvironment.getEnvironment();
        // Load the model from the classpath
        session = env.createSession("demo/src/main/java/onnx/example/logreg_iris.onnx", new OrtSession.SessionOptions());


        // Sample input data 
        float[][] inputData = {
            {3.1f,5.5f,4.4f,2.2f},
        };

        // Convert the 2D float array into FloatBuffer
        FloatBuffer inputBuffer = FloatBuffer.allocate(inputData.length * inputData[0].length);
        for(float[] row : inputData) {
            inputBuffer.put(row);
        }
        inputBuffer.rewind();

        // Create Onnx Tensor
        long[] shape = {1,4}; 
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, shape);

        // Prepare input map
        Map<String, OnnxTensor> inputMap = Collections.singletonMap("float_input", inputTensor);

        OrtSession.Result results = session.run(inputMap);

        // Retrieve output
        long[] output = (long[]) results.get(0).getValue();
        System.out.println("Predicted class: " + output[0]);

        inputTensor.close();
        results.close();
        session.close();
        env.close();
    }
    public static void main(String[] args) throws IOException {
        try {
            new OnnxModelService();
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
    
}