package onnx.example;

import ai.onnxruntime.*;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;
import javax.imageio.ImageIO;

public class PytorchOnnxModel {
    private OrtEnvironment env;
    private OrtSession session;

    public PytorchOnnxModel(String imagePath) throws OrtException, IOException {
        // Initialize ONNX Runtime environment
        env = OrtEnvironment.getEnvironment();
        session = env.createSession("demo/src/main/java/onnx/example/resnet_ONNX_exported.onnx", new OrtSession.SessionOptions());

        // Load and preprocess the image from file
        float[] inputData = preprocessImage(imagePath, 96, 96);

        // Convert inputData to FloatBuffer
        FloatBuffer inputBuffer = FloatBuffer.wrap(inputData);

        // Define input tensor shape: (1, 3, 96, 96)
        long[] shape = {1, 3, 96, 96};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, shape);

        // Prepare input map (Ensure the input name matches your ONNX model's expected name)
        Map<String, OnnxTensor> inputMap = Collections.singletonMap("input", inputTensor);

        // Run inference
        OrtSession.Result results = session.run(inputMap);

        // Extract output tensor (assuming model outputs probabilities)
        float[][] outputArray = (float[][]) results.get(0).getValue();

        // Find the class with the highest probability
        int predictedLabel = argmax(outputArray[0]);

        // Class mapping
        String[] classMapping = {"airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"};
        String predictedClass = classMapping[predictedLabel];

        System.out.println("Predicted Class: " + predictedClass);

        // Cleanup resources
        inputTensor.close();
        results.close();
        session.close();
        env.close();
    }

    // Helper function to find the index of the max value
    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // Image Preprocessing: Load, resize, and normalize the image
    private float[] preprocessImage(String imagePath, int targetWidth, int targetHeight) throws IOException {
        File imageFile = new File(imagePath);

        // ✅ Check if the file exists before processing
        if (!imageFile.exists()) {
            throw new IOException("Error: Image file not found at " + imagePath);
        }

        BufferedImage img = ImageIO.read(imageFile);

        // ✅ Check if ImageIO successfully loaded the image
        if (img == null) {
            throw new IOException("Error: Could not read the image. Unsupported format or corrupted file.");
        }

        // Resize the image to match the model's expected input size (96x96)
        BufferedImage resizedImg = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImg.createGraphics();
        g.drawImage(img.getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH), 0, 0, targetWidth, targetHeight, null);
        g.dispose();

        // Convert image to float array (No normalization, keeping values in [0,255])
        float[] floatArray = new float[3 * targetWidth * targetHeight];
        int[] rgbArray = new int[targetWidth * targetHeight];
        resizedImg.getRGB(0, 0, targetWidth, targetHeight, rgbArray, 0, targetWidth);

        for (int i = 0; i < rgbArray.length; i++) {
            int pixel = rgbArray[i];
            int r = (pixel >> 16) & 0xFF;  // Extract Red channel
            int gVal = (pixel >> 8) & 0xFF; // Extract Green channel
            int b = pixel & 0xFF;          // Extract Blue channel

            // Store raw pixel values (DO NOT normalize to [0,1])
            floatArray[i] = (float) r;       // Red channel
            floatArray[i + targetWidth * targetHeight] = (float) gVal;  // Green channel
            floatArray[i + 2 * targetWidth * targetHeight] = (float) b;  // Blue channel
        }
        return floatArray;
    }

    public static void main(String[] args) {
        try {
            // Specify the image file path
            String imagePath = "demo/src/main/java/onnx/example/sampleImages/airplane.jpeg"; 
            new PytorchOnnxModel(imagePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
