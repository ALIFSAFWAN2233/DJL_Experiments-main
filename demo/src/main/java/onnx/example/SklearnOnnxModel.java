package onnx.example;

import ai.onnxruntime.*;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;



public class SklearnOnnxModel {
    private OrtEnvironment env;
    private OrtSession session;

    // Function definition
    public SklearnOnnxModel() throws OrtException, IOException {
        // Initialize ONNX Runtime environment
        env = OrtEnvironment.getEnvironment();
        // Load the model from the classpath
        session = env.createSession("demo/src/main/java/onnx/example/logreg_iris.onnx", new OrtSession.SessionOptions());

        /*
         * Preprocess and format the input data so that it is compatible with the input tensor of the model
         */
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



    public static void main(String[] args) {
        try {
            // run the model
            new SklearnOnnxModel();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }






}







