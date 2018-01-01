/*
 * Name: Vijayagopal Krishnan (VJ), Israel Quiroz
 * Date: 12/10/17
 * Class: CS 4375.501
 * Course Project 
 */

/*
 * This program/project uses Neural nets to build a model from the training file and use that model to predict the class values for the test file
 * The data for both training and test files are stored in 2D String ArrayLists, which are then parsed and transferred to 2D double arrays. The values are then
 * normalized and stored in 2D Integer ArrayLists. An integer array storing the class labels for the training is also created, along with a double array
 * containing weights that have been randomly initialized. For the neural net algorithm, we had our learning rate set to 0.5 and kept an array of minimum errors
 * in order to invoke the stopping condition that all the values in the minimum errors array are 0. In case if it takes a long time for all the values in the
 * minimum array to be 0, we also kept track of the number iterations and decide to stop when we get to 2,000,000. During each iteration, the sigmoid of the sum
 * of the product of weights and attributes are calculated, along with the error values and then the weights are updated. During the weight update the derivative of the error for each attribute
 * is calculated and stored into the errorValues array. After the algorithm, the updated weights are then applied to the test data to predict the class value for the test file.
 */

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

public class NeuralProject
{
    public static void main(String[] args) throws IOException
    {
            ArrayList<ArrayList<String>> file_Data = new ArrayList<>(); //store data from train.txt
            ArrayList<ArrayList<String>> testFile_data = new ArrayList<>(); //store data from final-noclass.txt


           ArrayList<ArrayList<Integer>> data = new ArrayList<>(); //store the numeric discrete/cateogrical data from train.txt
           ArrayList<ArrayList<Integer>> test_data = new ArrayList<>(); //store the numeric discrete/categorical data from final-noclass.txt



        try {
            Scanner file = new Scanner(new File(args[0])); //is using the first file name which is the training data file, "train.txt"
            while (file.hasNextLine()) {
                ArrayList<String> a_Line = new ArrayList<>(); //this stores the current line of the file
                String line = file.nextLine();
                if (!line.isEmpty()) {   //checks if a line is empty and skips over it
                    String[] clean_array = line.split("\\s+");
                    Collections.addAll(a_Line, clean_array);
                    file_Data.add(a_Line);
                    Arrays.fill(clean_array, null); //these cleans helper array
                }
            }
        } catch (Exception e) {
            System.exit(0);
        }

        try {
            Scanner file = new Scanner(new File(args[1])); //is using the second file name which is the test data file, "final-noclass.txt"
            while (file.hasNextLine()) {
                ArrayList<String> a_Line = new ArrayList<>(); //this stores the current line of the file
                String line = file.nextLine();
                if (!line.isEmpty()) {   //checks if a line is empty and skips over it
                    String[] clean_array = line.split("\\s+");
                    Collections.addAll(a_Line, clean_array);
                    testFile_data.add(a_Line);
                    Arrays.fill(clean_array, null); //these cleans helper array
                }
            }
        } catch (Exception e) {
            System.exit(0);
        }

        Double[][] dub_array = new Double[file_Data.size()][file_Data.get(0).size()-1]; //store the values into the array for the training file. Last column contains class value for the training set
        Double[][] test_dub_array = new Double[testFile_data.size()][testFile_data.get(0).size()]; //same for the test file except there's no class label


       Integer[][] array = new Integer[file_Data.size()][file_Data.get(0).size()-1]; //store the normalized values bar the class label for the training file
        Integer[][] test_array = new Integer[testFile_data.size()][testFile_data.get(0).size()]; //store the normalized values for the test file






        for (int i = 0; i < dub_array.length; i++) {
            for (int j = 0; j < dub_array[i].length; j++) {
                dub_array[i][j] = Double.parseDouble(file_Data.get(i).get(j));
            }
        }


        for (int i = 0; i < test_dub_array.length; i++) {
            for (int j = 0; j < test_dub_array[i].length; j++) {
                test_dub_array[i][j] = Double.parseDouble(testFile_data.get(i).get(j));
            }
        }


        for(int i = 0; i < array.length; i++)
        {
            for(int j = 0; j < array[i].length; j++)
            {
                array[i][j] = normalizeValue(dub_array[i][j]); //normalize all the values minus the class label into 1s and 0s except the class label
            }
        }


        for(int i = 0; i < test_array.length; i++)
        {
            for(int j = 0; j < test_array[i].length; j++)
            {
                test_array[i][j] = normalizeValue(test_dub_array[i][j]); //same for the test file
            }
        }


        for (Integer [] a_array: array)
        {
            ArrayList<Integer> tempList = new ArrayList<>();
            Collections.addAll(tempList,a_array);
            data.add(tempList); //
        }

        for (Integer [] a_test_array: test_array)
        {
            ArrayList<Integer> tempList = new ArrayList<>();
            Collections.addAll(tempList,a_test_array);
            test_data.add(tempList); //add each row of the 2D array to the 2D arraylist for the test file
        }


        int labels[] = new int[file_Data.size()]; //store the class labels for the training file

        double weights [] = new double[file_Data.get(0).size()-1];//array of weights to use for the training and test data

        for(int i = 0; i < weights.length; i++)
        {
            weights[i] = ThreadLocalRandom.current().nextDouble((-1/Math.sqrt(array[i].length)), (1/Math.sqrt(array[i].length))); //initially assign random weights
        }


        double learningRate = 0.5; //learning rate of choice

        double errorValues [] = new double[file_Data.get(0).size()-1]; //store the error values for checking for convergence


        for (int i = 0; i < labels.length; i++) {
            labels[i] = Integer.parseInt(file_Data.get(i).get(file_Data.get(i).size() - 1)); //store the class labels of train.txt into the array
        }



        int iteration = 0; //keep track of the number of iterations
        do {
            for (int i = 0; i < data.size(); i++)
            {
              double sigmoidValue = getSigmoid(getSum(data.get(i),weights)); //store the sigmoid of the sum of the product of weights and attributes
              double derivativeValue = getDerivative(sigmoidValue); //store the derivative of the sigmoid value
              double errorVal = getError(sigmoidValue,labels[i]); //store the error value
              updateWeights(data.get(i),weights,learningRate,errorVal,derivativeValue, errorValues); //update the weights based on the error value, and product of weights and attributes
              iteration++; //increment iteration
            }
        } while(!isErrorMin(errorValues) || iteration < 2000000); //check if the error for all the attributes are 0 or if the procedure went over 2 million iterations


        PrintWriter writer = new PrintWriter("output.txt"); //Printwriter object to write to the file "output.txt"

        for (ArrayList<Integer> aTest_data : test_data) {
            if (getSigmoid(getSum(aTest_data, weights)) < 0.5)  //uses the same weights computed from the algorithim to predict the class values for the test file
                writer.println(0);
            else
                writer.println(1);
        }
        writer.close(); //closes the file
    }



    //gets the sum of the product of weights and attributes for dataPoint
    private static double getSum(ArrayList<Integer> dataPoint, double weights[])
    {
        double sum = 0;
        for (int i = 0; i < dataPoint.size(); i++)
        {
            sum += dataPoint.get(i) * weights[i];
        }
        return sum;
    }


    //returns the sigmoid of the sum calculated from above
    private static double getSigmoid(double sum)
    {
        return 1/(1 + Math.pow(Math.E, (-1 * sum)));
    }

    //returns  the derivative value of the sigmoid value calculated from above
    private static double getDerivative(double sigmoidValue)
    {
        return sigmoidValue * (1 - sigmoidValue);
    }

    //returns the difference between actual value and the predicted value for training
    private static double getError(double sigmoidValue, int actualValue)
    {
        return actualValue - sigmoidValue;
    }

    //updates the weights of all the attributes based on the error, initial weights and the sum of the product of weights and attributes
    private static void updateWeights(ArrayList<Integer> dataPoint, double weights [], double learningRate, double errorValue, double derivativeSigmoidValue, double errorVals[])
    {
        for (int i = 0; i < dataPoint.size(); i++)
        {
            errorVals[i] = errorValue * dataPoint.get(i) * derivativeSigmoidValue; //store the derivative error for each weight
            if(errorVals[i] != 0) //check if the derivative error is equal to zero or not
            {
                double weight = weights[i] + learningRate * errorValue * dataPoint.get(i) * derivativeSigmoidValue; //update the weight
                weights[i] = weight; //store the new updated weight
            }
        }
    }

    //check if the derivative error with respect to the weights is equal to 0 for all weights
    private static boolean isErrorMin (double errorVals[])
    {
        for (double val: errorVals)
        {
            if (val != 0)
                return false;
        }
        return true; //if they are all 0 then we reached the minimum error and a stopping point for the algorithim
    }


//normalize the  data into 0s and 1s based on their value
 private static int normalizeValue(double value)
 {
     if(value >= 1) //if the value is greater than equal to 1 then it's a 1
         return 1;
     else
         return 0; //otherwise 0
 }

}