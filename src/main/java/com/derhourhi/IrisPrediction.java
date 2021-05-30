package com.derhourhi;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IrisPrediction {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model=ModelSerializer.restoreMultiLayerNetwork(new File("model_iris.zip"));

        String[] types={"iris-setosa","iris-versicolor","iris-virginica"};

        INDArray predictionDta=Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2}
        });

        INDArray output=model.output(predictionDta);
        System.out.println("Prediction result");
        int [] classes=output.argMax(1).toIntVector();
        System.out.println(output);

        for (int i = 0; i < classes.length; i++) {
                System.out.println("Class: "+classes[i]);
                System.out.println("Type: "+types[classes[i]]);
        }
    }
    
}
