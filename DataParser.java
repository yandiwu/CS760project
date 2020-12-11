import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/**
 * This class contains a static method to parse csv files and 
 * generates an matrix containing all the training data in the file. 
 * Here, a matrix is implemented as a list of double arrays. 
 * @author Hyejin Jenny Yeon
 */
public class DataParser {
	
    /**
     * This method parses csv file and returns a matrix as ArrayList<double[]>.
     * @param filePath is the path of the file. 
     * @return ArrayList<double[]> a matrix containing all the training data
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static List<ArrayList<Integer>> parse2019TrainingRecords(String filePath) throws FileNotFoundException, IOException {
    	List<ArrayList<Integer>> data = new ArrayList<ArrayList<Integer>>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line = "";
        reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] stringValues = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
            int colSize = stringValues.length;
            ArrayList<Integer> intValues = new ArrayList<Integer>();
            //Ignores population
            for (int i = 5 ; i < colSize ; i ++) {
            intValues.add((int) Math.round(Double.parseDouble(stringValues[i])));	
            }
            data.add(intValues);
            //Add Labels
            if (stringValues[1].equals("\"Rep\"")) {
           		intValues.add(2); //REP
           	}
           	else {
               	intValues.add(4); //DEM
           	}
        }
        reader.close();
    return data;
    }
    
    /**
     * This method parses csv file and returns a matrix as ArrayList<double[]>.
     * @param filePath is the path of the file. 
     * @return ArrayList<double[]> a matrix containing all the training data
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static List<ArrayList<Integer>> parse2019TestRecords(String filePath) throws FileNotFoundException, IOException {
    	List<ArrayList<Integer>> data = new ArrayList<ArrayList<Integer>>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line = "";
        reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] stringValues = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
            int colSize = stringValues.length;
            ArrayList<Integer> intValues = new ArrayList<Integer>();
            //Ignores population
            for (int i = 5 ; i < colSize ; i ++) {
            intValues.add((int) Math.round(Double.parseDouble(stringValues[i])));	
            }
            data.add(intValues);
        }
        reader.close();
    return data;
    }
    
    /**
     * This method parses csv file and returns a matrix as ArrayList<double[]>.
     * @param filePath is the path of the file. 
     * @return ArrayList<double[]> a matrix containing all the training data
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static List<ArrayList<Integer>> parse2020TrainingRecords(String filePath) throws FileNotFoundException, IOException {
    	List<ArrayList<Integer>> data = new ArrayList<ArrayList<Integer>>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line = "";
        reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] stringValues = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
            int colSize = stringValues.length;
            ArrayList<Integer> intValues = new ArrayList<Integer>();
            //Ignores population
            for (int i = 5 ; i < colSize ; i ++) {
            intValues.add((int) Math.round(Double.parseDouble(stringValues[i])));	
            }
            data.add(intValues);
            //Add Labels
            if (stringValues[0].equalsIgnoreCase("republican")) {
           		intValues.add(2); //REP
           	}
           	else {
               	intValues.add(4); //DEM
           	}
        }
        reader.close();
    return data;
    }
    /**
     * This method parses csv file and returns a matrix as ArrayList<double[]>.
     * @param filePath is the path of the file. 
     * @return ArrayList<double[]> a matrix containing all the training data
     * @throws FileNotFoundException
     * @throws IOException
     */
    public static List<ArrayList<Integer>> parse2020TestingRecords(String filePath) throws FileNotFoundException, IOException {
    	List<ArrayList<Integer>> data = new ArrayList<ArrayList<Integer>>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line = "";
        reader.readLine();
        while ((line = reader.readLine()) != null) {
            String[] stringValues = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
            int colSize = stringValues.length;
            ArrayList<Integer> intValues = new ArrayList<Integer>();
            //Ignores population
            for (int i = 5 ; i < colSize ; i ++) {
            intValues.add((int) Math.round(Double.parseDouble(stringValues[i])));	
            }
            data.add(intValues);
        }
        reader.close();
    return data;
    }
    

}
