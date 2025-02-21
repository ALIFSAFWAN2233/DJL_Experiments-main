package onnx.example;

import ai.onnxruntime.*;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;





// Test loading a sklearn trained model in onnx format
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


// Test loading onnx converted pytorch model 
# Load the ONNX model
sess = rt.InferenceSession("resnet.onnx", providers=["CPUExecutionProvider"])

# Get input and output names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name  # This is the output tensor

# Create a dummy input (batch size 1, 3 channels, 96x96 image)
dummy_input = np.random.rand(1, 3, 96, 96).astype(np.float32)

# Run inference
output_tensor = sess.run([output_name], {input_name: dummy_input})[0]

# Check the shape of the output tensor
print(f"Output Tensor Shape: {output_tensor.shape}")  # Expected: (1, 1000)