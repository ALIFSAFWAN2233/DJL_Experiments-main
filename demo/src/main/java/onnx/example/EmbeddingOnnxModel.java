package onnx.example;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;


import ai.onnxruntime.*;


/*
 * 
 *  tokenizer from huggingface is not supported  
 *  need to have a helper script to tokenize the sample text input
 *
 */

public class EmbeddingOnnxModel {
    private OrtEnvironment env;
    private OrtSession session;


    public EmbeddingOnnxModel(String text) throws OrtException, IOException{
    // Initialize ONNX Runtime environment
    env = OrtEnvironment.getEnvironment();
    session = env.createSession("demo/src/main/java/onnx/example/mxbai_embed_large_v1.onnx", new OrtSession.SessionOptions());
    
    //Parse the json and proprocess the data
    // this embedding model receives inputId and attentionMask 
    Gson gson = new Gson();
    JsonObject jsonObj = gson.fromJson(text, JsonObject.class);

    // Get input_ids and attention_mask arrays (assuming single-batch input)
    JsonArray inputIdsJson = jsonObj.getAsJsonArray("input_ids");
    JsonArray attentionMaskJson = jsonObj.getAsJsonArray("attention_mask");

    // Convert the nested JSON arrays to 2D long arrays
    long[][] inputIds = jsonArrayToLong2D(inputIdsJson);
    long[][] attentionMask = jsonArrayToLong2D(attentionMaskJson);
    
    //Create inputTensor
    OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIds);
    OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMask);
    
    Map<String, OnnxTensor> inputs = new HashMap<>();
    inputs.put("input_ids", inputIdsTensor);
    inputs.put("attention_mask", attentionMaskTensor);

    //Run inference
    OrtSession.Result results = session.run(inputs);

    //Extract output tensor
    Object outputValue = results.get(0).getValue();
    if (outputValue instanceof float[][][]) {
        float[][][] sequenceEmbeddings = (float[][][]) outputValue;
        // For example, select the embedding for the first token of the first batch:
        float[] pooledEmbedding = sequenceEmbeddings[0][0]; 
        System.out.println("Pooled Embedding (first token):");
        for (float f : pooledEmbedding) {
            System.out.print(f + " ");
        }
        System.out.println();
    } else if (outputValue instanceof float[][]) {
        float[][] pooledEmbedding = (float[][]) outputValue;
        System.out.println("Pooled Embedding:");
        for (float f : pooledEmbedding[0]) {
            System.out.print(f + " ");
        }
        System.out.println();
    } else {
        System.out.println("Unexpected output type: " + outputValue.getClass().getName());
    }

    //print result
    inputIdsTensor.close();
    attentionMaskTensor.close();
    results.close();
    session.close();
    env.close();

    }



    // Helper function to get the tokenized text which will execute the python helper script
    public static String textTokenizer(String text) throws Exception{
        //call the helper script
        ProcessBuilder pb = new ProcessBuilder("python", "demo/src/main/java/onnx/example/tokenizerHelper.py", text);
        Process process = pb.start();

        //get the output from the helper
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

        //checks if it is null
        String line;
        String jsonOutput = null;
        while((line = reader.readLine()) != null){
           if(line.trim().startsWith("{")){
            jsonOutput = line.trim();
           }
        }
        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Tokenization failed with exit code " + exitCode);
        }
        return jsonOutput;

    }

    //method to convert json array to [long] array from normal array
    private long[][] jsonArrayToLong2D(JsonArray jsonArray) {
        int outerSize = jsonArray.size();
        long[][] result = new long[outerSize][];
        for (int i = 0; i < outerSize; i++) {
            JsonArray innerArray = jsonArray.get(i).getAsJsonArray();
            int innerSize = innerArray.size();
            long[] innerResult = new long[innerSize];
            for (int j = 0; j < innerSize; j++) {
                innerResult[j] = innerArray.get(j).getAsLong();
            }
            result[i] = innerResult;
        }
        return result;
    }

    public static void main(String[] args) {
        try {
            // Specify the input text
            String input_text = "Hye Hazrul!!!"; 

            //tokenize the input text
            String tokenized_text = textTokenizer(input_text);

            //inference
            new EmbeddingOnnxModel(tokenized_text);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }



}
