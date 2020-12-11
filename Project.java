import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
/**
 * This is a class to make various trees. One can also use
 * public methods in classes DecTree.java such as
 * cross-validation, print tree etc. 
 * @author Hyejin Jenny Yeon
 */
public class Project {

	public static void main(String[] args) throws FileNotFoundException, IOException {
		//Only de-comment one of the two lines below
	//	DecTree test1 = new DecTree("CoveringClimateNow_2020_States_Districts_Only_Parties.csv", 2020);
		DecTree county = new DecTree("CoveringClimateNow_2020.csv", 2020);

	//	System.out.println("--------------------------------");
	//	county.printTree();
	//	test1.processTestData("PartisanMapData_20190218.01.txt");
	//	System.out.println("Depth of the non-pruned tree: " + test1.getDepth());
	//	System.out.println("--------------------------------");
	//	test1.pruneTree(1);
	//	System.out.println(test1.testResult().toString());
	//	test1.printAccuracy();
	//	test1.printTree();
	//	System.out.println("Depth of the pruned tree: " + test1.getDepth());
		
		ArrayList<Integer> result = county.crossValidation(10, "CoveringClimateNow_2020.csv");
		for (int i : result) {
			if (i==2)
			System.out.println("REP");
			else System.out.println("DEM");
		}
		
	//	test1.printCVAccuracy(5, "CoveringClimateNow_2020_States_Districts_Only_Parties.csv");
		county.printCVAccuracy(10, "CoveringClimateNow_2020.csv");


	}
}
	
