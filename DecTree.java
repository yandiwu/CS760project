import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
/**
 * This class is a modification of the class with the same name I wrote for CS760. 
 * This class can build a decision tree, do a cross validation, and print the tree,
 * and can also tests and print accuracy result
 * @author Hyejin Jenny Yeon
 * 
 */
public class DecTree {
	
	private int numAttr;
	private ArrayList<Integer> features;
	public List<ArrayList<Integer>> trainData;
	public DecTreeNode root;
	//Test data format: (x1, x2, .. , xn) where xi's are values of the feature.
	public List<ArrayList<Integer>> testData;
	private int counterfor2 = 0;
	private int counterfor4 = 0;
	private int year = 0;
	

	/**
	 * Constructor for using all available features and takes data as a file
	 * @param filePath
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public DecTree(String filePath, int year) throws FileNotFoundException, IOException {
		this.year = year;
		if (year == 2019) {
			this.trainData = DataParser.parse2019TrainingRecords(filePath);
			this.testData = DataParser.parse2019TestRecords(filePath);
		}
		else {
			this.trainData = DataParser.parse2020TrainingRecords(filePath);
			this.testData = DataParser.parse2020TestingRecords(filePath);
		}
		this.numAttr = trainData.get(0).size()-1;
		this.features = new ArrayList<Integer>(); 
		for (int i = 1; i< numAttr+1 ; i ++) {
			features.add(i);
		}
		this.root = buildTree(trainData);	
	}
	
	/**
	 * Allows the user to change the test set using their own file
	 * @param filePath
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void processTestData(String filePath) throws FileNotFoundException, IOException {
		if (this.year == 2019) this.testData = DataParser.parse2019TestRecords(filePath);
		else this.testData = DataParser.parse2020TestingRecords(filePath);
	}
	
	/**
	 * Allows the user to change the test set by entering as list of list
	 * @param testData
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void processTestData(List<ArrayList<Integer>> testData) throws FileNotFoundException, IOException {
		this.testData = this.selectData(testData);
	}
	
	/**
	 * Helper method to select only few features
	 * @param rawData
	 * @return
	 */
	private List<ArrayList<Integer>> selectData(List<ArrayList<Integer>> rawData) {
		List<ArrayList<Integer>> subData = new ArrayList<ArrayList<Integer>>();
		for (int i = 0 ; i < rawData.size() ; i ++) {
			ArrayList<Integer> rowi = new ArrayList<Integer>();
			for (Integer f : this.features) {
				rowi.add(rawData.get(i).get(f-1));
			}	
			subData.add(rowi);
		}

		return subData;
	}
	
	/**
	 * Method for computing entropy. This is very useful because
	 * we primarily work with binary features/ labels
	 * @param p0
	 * @return entropy
	 */
	public static double entropy(double p0){ 
		if (p0 == 0 || p0 == 1) return 0;
		
		double p1 = 1 - p0;
		return -(p0*Math.log(p0)/Math.log(2) + p1*Math.log(p1)/Math.log(2));
	}
	
	/**
	 * Method for computing information gain. This computes I(y,x) which is equal to I(x,y)
	 * because I(.,.) is symmetric. 
	 * Information gain is also known as mutual information
	 * @param dataSet
	 * @param feature
	 * @param threshold
	 * @return information gain
	 */
	public static double informationGain(List<ArrayList<Integer>> dataSet, int feature, int threshold){
		int dataSize = dataSet.size();
		// first computing H(Y)
		int count = 0;
		for (List<Integer> data: dataSet) {
			if (data.get(data.size()-1) == 2) { // if label equals 2, last column is label
				count++;
			}
		}
		double Hy = entropy((double) count / dataSize); // since there are only two labels, this
														// works just fine for H(y). 
		
		// Onto computing H(Y|X)
		double Hyx = 0;
		int countLess = 0;
		int countGreater = 0;
		int countLessAndPositive = 0; //Positive means survived
		int countGreaterAndPositive = 0;
		for (List<Integer> data: dataSet){
			if (data.get(feature) <= threshold) {
				countLess++;
				// below needed for conditional probability P(positive|feature <= split point)
				if (data.get(data.size()-1) == 2) countLessAndPositive++; // last column is y
			} else {
				countGreater++;
				if (data.get(data.size()-1) == 2) countGreaterAndPositive++;
			}
		}
		double prob1 = (double)countLess/dataSize; // P(feature <= split point)
		double prob2 = (double)countGreater/dataSize;
		// One of prob1, prob2 may be zero, if so, skip these to save some time.
		if (prob1>0){
			//Here, we are using the fact that there are only two labels
			Hyx = Hyx + prob1 * entropy(((double)countLessAndPositive)/countLess);
		}
		if(prob2>0){
			Hyx = Hyx + prob2 * entropy(((double)countGreaterAndPositive)/countGreater);
		}
		// return difference between entropy and conditional entropy
		return Hy - Hyx;
	}

	/**
	 * Builds a tree using recursive algorithm when there is only one feature used.
	 * @param dataSet
	 * @return a pointer to the root node. 
	 */
	private DecTreeNode buildTreeWithOne(List<ArrayList<Integer>> dataSet) {
		int bestAttr = -1;
		int bestThres = Integer.MIN_VALUE;
		double bestScore = Double.NEGATIVE_INFINITY;
		DecTreeNode node = null;
			// Determines where to split using information gain. 
			for (int j = 0; j < numAttr; j++) {
				int threshhold = Collections.max(this.trainData.get(j));
				for (int i=0; i<threshhold; i++){ 
					double score = informationGain(dataSet, j, i); 
					if (score > bestScore) {
						bestScore = score;
						bestAttr = j;
						bestThres = i; 
					}
				}
			}
		node = new DecTreeNode(-1, bestAttr, bestThres);			
			// Now the feature is binary.
			// Split the entire # of instances into two groups 
		    // based on the threshold value (<= and >)
			List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
			List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
			for (ArrayList<Integer> data : dataSet) {
				if (data.get(bestAttr) <= bestThres) {
					leftList.add(data);
				}
				else {
					rightList.add(data);
				}
			}
		
			int count = 0;
			for(List<Integer> data: leftList) {
				if (data.get(data.size()-1) == 2)
				count += 1;
			}
			
			if (count >= leftList.size() - count) {
				node.left = new DecTreeNode(2, -1, -1); // assign label 2 to the leaf node
			}
			else {
				node.left = new DecTreeNode(4, -1, -1); // assign label 4 to the leaf node
			}
			count = 0;
			for(List<Integer> data: rightList) {
				if (data.get(data.size()-1) == 2)
				count += 1;
			}
			
			if (count >= rightList.size() - count) {
				node.right = new DecTreeNode(2, -1, -1); // assign label 2 to the right node
			}
			else {
				node.right = new DecTreeNode(4, -1, -1); // assign label 4 to the right node
			}
		return node;
	}
	
	/**
	 * Builds a tree using recursive algorithm
	 * Comments are left out because this is similar to the method buildTreeWithOne
	 * @param dataSet
	 * @return a pointer to the root node. 
	 */
	private DecTreeNode buildTree(List<ArrayList<Integer>> dataSet) {

		int numData = dataSet.size();
		int bestAttr = -1;
		int bestThres = Integer.MIN_VALUE;
		double bestScore = Double.NEGATIVE_INFINITY;
		boolean leaf = false;
		DecTreeNode node = null;
		
		// Compute the best feature with the best split:
		// Determines which feature with what spiting point to use
	    // based on information gain. 
		if (!leaf) {
			for (int j = 0; j < numAttr; j++) {
				int threshhold = Collections.max(this.trainData.get(j));
				for (int i=0; i<threshhold; i++){
					double score = informationGain(dataSet, j, i); 
					if (score > bestScore) {
						bestScore = score;
						bestAttr = j;
						bestThres = i; 
					}
				}
			}
			if (bestScore == 0) {
				leaf = true;
			}
			List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
			List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
			for (ArrayList<Integer> data : dataSet) {
				if (data.get(bestAttr) <= bestThres) {
					leftList.add(data);
				}
				else {
					rightList.add(data);
				}
			}
			
			if (leftList.size() == 0 || rightList.size() == 0) {
				leaf = true;
			}
			if (!leaf) {
				node = new DecTreeNode(-1, bestAttr, bestThres);
				node.left = buildTree(leftList);
				node.right = buildTree(rightList);
			}
		}
		if (leaf) {
			int count = 0;
			for(List<Integer> data: dataSet) {
				if (data.get(data.size()-1) == 2)
				count += 1;
			}
			
			if (count >= numData - count) {
				node = new DecTreeNode(2, -1, -1); // assign label 2 to the leaf node
			}
			else {
				node = new DecTreeNode(4, -1, -1); // assign label 4 to the leaf node
			}
		}
		return node;
	}

	/**
	 * Helper method for printing. 
	 * It prints in the format that can be read from left to right to accommodate
	 * limited horizontal space
	 * @param node
	 * @param numSpaces
	 */
	private void printTreeHelp(DecTreeNode node, int numSpaces) {

		if (node == null ) return;
		Integer trueFeature = -1;
		
		if (node.feature != -1) {
			trueFeature = this.features.get(node.feature);
			//TODO:remove this
		//	System.out.println(this.features);
		//	System.out.print("if (x" + node.feature+ " <= ");
			for (int i = 0 ; i < numSpaces - this.maxDepth(node)+1;i++) {
				System.out.print(" ");
			}
			System.out.print("("+ (this.maxDepth(node)-1) +") ");
			System.out.print("if ( x" + trueFeature+ " <= ");
			if (node.left.feature == -1 ) {
				System.out.print(node.threshold + " )");
			}
			else {
				System.out.println(node.threshold + " )");
			}
		}
		else {
		//	System.out.print("else ");
			String message = "";
			if (node.classLabel==2) message = "REP"; 
			else message = "DEM";
			System.out.println(" Result: " + message +".");
			return;

		}

		printTreeHelp(node.left, this.getDepth());
		if (node.right.feature == -1 ) {
			for (int i = 0 ; i < numSpaces -this.maxDepth(node)+1;i++) {
				System.out.print(" ");
			}
			System.out.print("("+ (this.maxDepth(node)-1) +") ");
			System.out.print("else ");
		}
		else {
			for (int i = 0 ; i < numSpaces -this.maxDepth(node)+1;i++) {
				System.out.print(" ");
			}
			System.out.print("("+ (this.maxDepth(node)-1) +") ");
			System.out.println("else ");
		}
		printTreeHelp(node.right, this.getDepth()); 

	}
	
	/**
	 * Maximum Depth of a tree: 
	 * the number of nodes along the longest path from the root node  
	 * @param node
	 * @return depth
	 */ 
	private int maxDepth(DecTreeNode node)  { 
     if (node == null) return 0; 
     else { 
         /* compute the depth of each subtree */
         int lDepth = maxDepth(node.left); 
         int rDepth = maxDepth(node.right); 

         /* use the larger one */
         if (lDepth > rDepth) 
             return (lDepth + 1); 
          else 
             return (rDepth + 1); 
     } 
 } 
	/**
	 * This is to perform n-fold cross validation. 
	 * This cross Validation method currently works only when all features are selected
	 * @param fold number of folds, i.e. "n"
	 * @param originalFilePath
	 * @return result of testing
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public ArrayList<Integer> crossValidation(int fold, String originalFilePath) throws FileNotFoundException, IOException {
		int sizeOfSet = this.trainData.size()/fold;
		ArrayList<Integer> results = new ArrayList<Integer>();
		this.trainData = DataParser.parse2020TrainingRecords(originalFilePath);
		this.testData = DataParser.parse2020TestingRecords(originalFilePath);
		for (int i = 0 ; i < fold ; i ++) {
			List<ArrayList<Integer>> trainingSubset = new ArrayList<ArrayList<Integer>>();
			List<ArrayList<Integer>> testSubset = new ArrayList<ArrayList<Integer>>();
			if (i == (fold - 1)) {
				testSubset = this.testData.subList(sizeOfSet*i, this.trainData.size());		
				trainingSubset = this.trainData.subList(0, sizeOfSet*i);
				this.trainData = trainingSubset;

			}
			else {
				for (int j = 0 ; j < sizeOfSet ; j++) {
					trainData.remove(sizeOfSet*i);
				}
				testSubset = this.testData.subList(sizeOfSet*i, sizeOfSet*(i+1));
			}

			this.root=this.buildTree(trainData);
			this.testData = testSubset;
			//System.out.println("how many? " + this.trainData.size());
			ArrayList<Integer> partialResults = this.testResult();
			
			for (Integer r : partialResults) {
				results.add(r);
			}
			this.printTree();
			System.out.println("===========");
			
			this.testData = DataParser.parse2020TestingRecords(originalFilePath);
			this.trainData = DataParser.parse2020TrainingRecords(originalFilePath);

		}
		this.testData = DataParser.parse2020TestingRecords(originalFilePath);
		this.trainData = DataParser.parse2020TrainingRecords(originalFilePath);
		return results;		
	}
	
	public int getDepth() {
		return maxDepth(this.root)-1; // the node with classification shouldn't be included
	}
	
	// In case people wants a simpler printing method, go head and uncomment this:	
	/*
	private void simplePrintTreeHelp(DecTreeNode node) {

		if (node==null) return;
		Integer trueFeature = -1;
		if (node.feature != -1) {
			trueFeature = this.features.get(node.feature);
		}
		if (node.feature!=-1) System.out.println(trueFeature);
		printTreeHelp(node.left);
		printTreeHelp(node.right); 
		
	}
	*/

	
	/**
	 * Print Tree in a nice format
	 */
	public void printTree() {
		printTreeHelp(this.root, this.getDepth());
	}
		
	/**
	 * This retuns a list of labels predicted by the tree in the class. 
	 * @return test results
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public ArrayList<Integer> testResult() throws FileNotFoundException, IOException{
		ArrayList<Integer> results = new ArrayList<Integer>();
		for (ArrayList<Integer> list : this.testData) {
			boolean leaf = false;
			DecTreeNode current=this.root;
			while (!leaf) {
				if (current.feature == -1) {
					leaf = true;
					if (current.classLabel==2) results.add(2);
					else results.add(4);
				}
				else {
						if (list.get(current.feature) <= current.threshold) {
						current = current.left;
					}
					else current = current.right;
					
				}			
			}
		
		}
		return results;
	}
	
	/**
	 * This prints number of correctly identified data points in the format of
	 * "705 correct out of 887"
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void printAccuracy() throws FileNotFoundException, IOException {
		ArrayList<Integer> result = this.testResult();
		int count = 0;
		int dataSize = this.trainData.size();
		for (int i = 0 ; i < dataSize ; i++) {
			if (this.trainData.get(i).get(this.numAttr) == result.get(i)) count++;
		}
		System.out.println(count + " correct out of " + dataSize);
		
	}
	/**
	 * This both performs and prints results of cross-validation 
	 * @param fold
	 * @param originalFilePath
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void printCVAccuracy(int fold, String originalFilePath) throws FileNotFoundException, IOException {
		ArrayList<Integer> result  = this.crossValidation(fold, originalFilePath);
		int count = 0;
		int dataSize = this.trainData.size();
		for (int i = 0 ; i < dataSize ; i++) {
			if (this.trainData.get(i).get(this.numAttr) == result.get(i)) count++;
		}
		System.out.println(count + " correct out of " + dataSize);
		
	}

    /**
     * Helper for tree pruning
     * @param root
     * @param level
     * @return
     */
    private DecTreeNode pruneTreeHelp (DecTreeNode root, int level) { 

        if (root == null) return null;
        if (level == 1) {

        	this.majLabel(root);
        	if (this.counterfor2 >= this.counterfor4) {
        		counterfor2 = 0;
        		counterfor4 = 0;
 
        		root.left = null;
        		root.right = null;
        		root = new DecTreeNode(2, -1, -1);
        		return root;
        	}
        	else {
        		counterfor2 = 0;
        		counterfor4 = 0;
        		root.left = null;
        		root.right = null;
        		root = new DecTreeNode(4, -1, -1);
        		return root;
        	}
        	
        }

        else if (level > 1) 
        { 
        	root.left = pruneTreeHelp(root.left, level-1); 
        	root.right = pruneTreeHelp(root.right, level-1); 
        } 
        return root;

    }
    
    /**
     * The user can prune the tree.
     * @param level that the user want to prune the tree
     */
    public void pruneTree(int level) {
    	//pruneTreeHelp (this.root, level+1);
    	this.root = pruneTreeHelp (this.root, level+1);
    }
    
    /**
     * Helper method for tree pruning. Calculates majority label 
     * @param node
     */
    private void majLabel(DecTreeNode node) {
		if (node == null ) return;
		if (node.feature != -1) {
		}
		else {
			if (node.classLabel == 2) this.counterfor2++;
			else this.counterfor4++;
			return;
		}
		majLabel(node.left);
		majLabel(node.right); 
		return;
    }

}