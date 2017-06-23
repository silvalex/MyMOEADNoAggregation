package moead;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.w3c.dom.Text;
import org.xml.sax.SAXException;

import representation.IndirectCrossoverOperator;
import representation.IndirectIndividual;
import representation.IndirectMutationOperator;

/**
 * The entry class for the program. The main algorithm is executed once MOEAD is instantiated.
 *
 * @author sawczualex
 */
public class MOEAD {
	// Parameter settings
	public static long seed = 1;
	public static int generations = 51;
	public static int popSize = 500;
	public static int numObjectives = 2;
	public static int numNeighbours = 30;
	public static double crossoverProbability = 0.8;
	public static double mutationProbability = 0.2;
	public static StoppingCriteria stopCrit = new GenerationStoppingCriteria(generations);
	public static Individual indType = new IndirectIndividual();
	public static MutationOperator mutOperator = new IndirectMutationOperator();
	public static CrossoverOperator crossOperator = new IndirectCrossoverOperator();
	public static String outFileName = "out.stat";
	public static String frontFileName = "front.stat";
	public static String serviceRepository = "services-output.xml";
	public static String serviceTaxonomy = "taxonomy.xml";
	public static String serviceTask = "problem.xml";
	// Fitness weights
	public static double w1 = 0.25;
	public static double w2 = 0.25;
	public static double w3 = 0.25;
	public static double w4 = 0.25;

	// Constants
	public static final int AVAILABILITY = 0;
	public static final int RELIABILITY = 1;
	public static final int TIME = 2;
	public static final int COST = 3;

	// Internal state
	private Individual[] population = new Individual[popSize];
	private double[][] weights = new double[popSize][numObjectives];
	private int[][] neighbourhood = new int[popSize][numNeighbours];
	private Map<String, Service> serviceMap = new HashMap<String, Service>();
	public Map<String, TaxonomyNode> taxonomyMap = new HashMap<String, TaxonomyNode>();
	public Set<String> taskInput;
	public Set<String> taskOutput;
	public Service startServ;
	public Set<Service> relevant;
	public List<Service> relevantList;
	public Service endServ;
	public Random random;
	public int numLayers;
	// Normalisation bounds
	public double minAvailability = 0.0;
	public double maxAvailability = -1.0;
	public double minReliability = 0.0;
	public double maxReliability = -1.0;
	public double minTime = Double.MAX_VALUE;
	public double maxTime = -1.0;
	public double minCost = Double.MAX_VALUE;
	public double maxCost = -1.0;

	// Statistics
	private double[] breedingTime = new double[generations];
	private double[] evaluationTime = new double[generations];
	FileWriter outWriter;
	FileWriter frontWriter;

	/**
	 * Loads and parses the parameter file.
	 *
	 * @param fileName
	 */
	private void parseParamsFile(String fileName) {
		try {
			Scanner scan = new Scanner(new File(fileName));
			while (scan.hasNext()) {
				setParam(scan.next(), scan.next());
			}
			scan.close();
		}
		catch (FileNotFoundException e) {
			System.err.println("Cannot read parameter file.");
			e.printStackTrace();
		}
	}

	/**
	 * Sets the next parameter from the file, checking against a list
	 * of possibilities.
	 *
	 * @param scan
	 */
	private void setParam(String token, String param) {
		try {
			switch(token) {
		        case "seed":
		       	 seed = Long.valueOf(param);
		            break;
		        case "generations":
		       	 generations = Integer.valueOf(param);
		       	 break;
		        case "popSize":
		       	 popSize = Integer.valueOf(param);
		       	 break;
		        case "numObjectives":
		       	 numObjectives = Integer.valueOf(param);
		            break;
		        case "numNeighbours":
		       	 numNeighbours = Integer.valueOf(param);
		            break;
		        case "crossoverProbability":
		       	 crossoverProbability = Double.valueOf(param);
		       	 break;
		        case "mutationProbability":
		       	 mutationProbability = Double.valueOf(param);
		            break;
		        case "stopCrit":
		       	 stopCrit = (StoppingCriteria) Class.forName(param).getConstructor(Integer.TYPE).newInstance(generations);
		       	 break;
		        case "indType":
		       	 indType = (Individual) Class.forName(param).newInstance();
		       	 break;
		        case "mutOperator":
		       	 mutOperator = (MutationOperator) Class.forName(param).newInstance();
		       	 break;
		        case "crossOperator":
		       	 crossOperator = (CrossoverOperator) Class.forName(param).newInstance();
		       	 break;
		        case "outFileName":
		       	 outFileName = param;
		       	 break;
		        case "frontFile":
		       	 frontFileName = param;
		       	 break;
		        case "serviceRepository":
		       	 serviceRepository = param;
		       	 break;
		        case "serviceTaxonomy":
		       	 serviceTaxonomy = param;
		       	 break;
		        case "serviceTask":
		       	 serviceTask = param;
		       	 break;
		        case "w1":
		       	 w1 = Double.valueOf(param);
		       	 break;
		        case "w2":
		       	 w2 = Double.valueOf(param);
		       	 break;
		        case "w3":
		       	 w3 = Double.valueOf(param);
		       	 break;
		        case "w4":
		       	 w4 = Double.valueOf(param);
		       	 break;
		        default:
		            throw new IllegalArgumentException("Invalid parameter: " + token);
			}
		}
		catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException | NoSuchMethodException | SecurityException | ClassNotFoundException e) {
			System.err.println("Cannot parse parameter correctly: " + token);
			e.printStackTrace();
		}
	}

	public MOEAD(String[] args) {
		parseParamsFile(args[0]);

		// Read in any additional parameters
		for (int i = 1; i < args.length; i +=2) {
			setParam(args[i], args[i+1]);
		}

		int generation = 0;
		indType.setInit(this);

		// Initialise
		long startTime = System.currentTimeMillis();
		initialise();
		breedingTime[generation] = System.currentTimeMillis() - startTime;

		// While stopping criteria not met
		while(!stopCrit.stoppingCriteriaMet()) {
			// Write out stats
			writeOutStatistics(outWriter, generation);

			startTime = System.currentTimeMillis();
			// Create an array to hold the new generation
			Individual[] newGeneration = new Individual[popSize];
			System.arraycopy(population, 0, newGeneration, 0, popSize);

			// Evolve new individuals for each problem
			for (int i = 0; i < popSize; i++) {
				Individual newInd = evolveNewIndividual(population[i], i, random);
				// Update neighbours
				updateNeighbours(newInd, i, newGeneration);
			}
			// Copy the next generation over as the new population
			population = newGeneration;
			generation++;

			evaluationTime[generation] = System.currentTimeMillis() - startTime;
		}


		// Produce final Pareto front
		Set<Individual> paretoFront = produceParetoFront(population);
		// Write the front to disk
		writeFrontStatistics(frontWriter, paretoFront);

		// Close writers
		try {
			outWriter.close();
			frontWriter.close();
		}
		catch (IOException e) {
			System.err.println("Cannot close stat writers.");
			e.printStackTrace();
		}

	}

	/**
	 * Creates weight vectors, calculates neighbourhoods, and generates an initial population.
	 */
	public void initialise() {
		// Create stat writers
		try {
			outWriter = new FileWriter(new File(outFileName));
			frontWriter = new FileWriter(new File(frontFileName));
		}
		catch (IOException e) {
			System.err.println("Cannot create stat writers.");
			e.printStackTrace();
		}
		// Parse dataset files
		parseWSCServiceFile(serviceRepository);
		parseWSCTaskFile(serviceTask);
		parseWSCTaxonomyFile(serviceTaxonomy);

		findConceptsForInstances();

		double[] mockQos = new double[4];
		mockQos[TIME] = 0;
		mockQos[COST] = 0;
		mockQos[AVAILABILITY] = 1;
		mockQos[RELIABILITY] = 1;
		Set<String> startOutput = new HashSet<String>();
		startOutput.addAll(taskInput);
		startServ = new Service("start", mockQos, new HashSet<String>(), taskInput);
		endServ = new Service("end", mockQos, taskOutput ,new HashSet<String>());

		populateTaxonomyTree();
		relevant = getRelevantServices(serviceMap, taskInput, taskOutput);
		relevantList = new ArrayList<Service>(relevant);

		calculateNormalisationBounds(relevant);

		// Ensure that mutation and crossover probabilities add up to 1
		if (mutationProbability + crossoverProbability != 1.0)
			throw new RuntimeException("The probabilities for mutation and crossover should add up to 1.");
		// Initialise random number
		random = new Random(seed);
		// Create a set of uniformly spread weight vectors
		initWeights();
		// Identify the neighbouring weights for each vector
		identifyNeighbourWeights();
		// Generate an initial population
		for (int i = 0; i < population.length; i++)
			population[i] = indType.generateIndividual();
	}

	/**
	 * Initialises the uniformely spread weight vectors. This code come from the authors'
	 * original code base.
	 */
	private void initWeights() {
		for (int i = 0; i < popSize; i++) {
			if (numObjectives == 2) {
				double[] weightVector = new double[2];
				weightVector[0] = i / (double) popSize;
				weightVector[1] = (popSize - i) / (double) popSize;
				weights[i] = weightVector;
			}
			else if (numObjectives == 3) {
				for (int j = 0; j < popSize; j++) {
					if (i + j < popSize) {
						int k = popSize - i - j;
						double[] weightVector = new double[3];
						weightVector[0] = i / (double) popSize;
						weightVector[1] = j / (double) popSize;
						weightVector[2] = k / (double) popSize;
						weights[i] = weightVector;
					}
				}
			}
			else {
				throw new RuntimeException("Unsupported number of objectives. Should be 2 or 3.");
			}
		}
	}

	/**
	 * Create a neighbourhood for each weight vector, based on the Euclidean distance between each two vectors.
	 */
	private void identifyNeighbourWeights() {
		// Calculate distance between vectors
		double[][] distanceMatrix = new double[popSize][popSize];

		for (int i = 0; i < popSize; i++) {
			for (int j = 0; j < popSize; j++) {
				if (i != j)
					distanceMatrix[i][j] = calculateDistance(weights[i], weights[j]);
			}
		}

		// Use this information to build the neighbourhood
		for (int i = 0; i < popSize; i++) {
			int[] neighbours = identifyNearestNeighbours(distanceMatrix[i], i);
			neighbourhood[i] = neighbours;
		}
	}

	/**
	 * Calculates the Euclidean distance between two weight vectors.
	 *
	 * @param vector1
	 * @param vector2
	 * @return distance
	 */
	private double calculateDistance(double[] vector1, double[] vector2) {
		double sum = 0;
		for (int i = 0; i < vector1.length; i++) {
			sum += Math.pow((vector1[i] - vector2[i]), 2);
		}
		return Math.sqrt(sum);
	}

	/**
	 * Returns the indices for the nearest neighbours, according to their distance
	 * from the current vector.
	 *
	 * @param distances - a list of distances from the other vectors
	 * @param currentIndex - the index of the current vector
	 * @return indices of nearest neighbours
	 */
	private int[] identifyNearestNeighbours(double[] distances, int currentIndex) {
		Queue<IndexDistancePair> indexDistancePairs = new LinkedList<IndexDistancePair>();

		// Convert the vector of distances to a list of index-distance pairs.
		for(int i = 0; i < distances.length; i++) {
			indexDistancePairs.add(new IndexDistancePair(i, distances[i]));
		}
		// Sort the pairs according to the distance, from lowest to highest.
		Collections.sort((LinkedList<IndexDistancePair>) indexDistancePairs);

		// Get the indices for the required number of neighbours
		int[] neighbours = new int[numNeighbours];

		// Get the neighbours, excluding the vector itself
		IndexDistancePair neighbourCandidate = indexDistancePairs.poll();
		for (int i = 0; i < numNeighbours; i++) {
			while (neighbourCandidate.getIndex() == currentIndex)
				neighbourCandidate = indexDistancePairs.poll();
			neighbours[i] = neighbourCandidate.getIndex();
		}
		return neighbours;
	}

	/**
	 * Applies genetic operators and returns a new individual, based on the original
	 * provided.
	 *
	 * @param original
	 * @return new
	 */
	private Individual evolveNewIndividual(Individual original, int index, Random random) {
		// Check whether to apply mutation or crossover
		double r = random.nextDouble();
		boolean performCrossover;
		if (r <= crossoverProbability)
			performCrossover = true;
		else
			performCrossover = false;

		// Perform crossover if that is the chosen operation
		if (performCrossover) {
			// Select a neighbour at random
			int neighbourIndex = random.nextInt(numNeighbours);
			Individual neighbour = population[neighbourIndex];
			return crossOperator.doCrossover(original.clone(), neighbour.clone(), this);
		}
		// Else, perform mutation
		else {
			return mutOperator.mutate(original.clone(), this);
		}
	}

	/**
	 * Updates the neighbours of the vector for a given index, changing their associated
	 * individuals to be the new individual (provided that this new individual is a better
	 * solution to the problem).
	 *
	 * @param newInd
	 * @param index
	 */
	private void updateNeighbours(Individual newInd, int index, Individual[] newGeneration) {
		// Retrieve neighbourhood indices
		int[] neighbourhoodIndices = neighbourhood[index];
		double newScore = calculateScore(newInd, index);
		double oldScore;
		for (int nIdx : neighbourhoodIndices) {
			// Calculate scores for the old solution, versus the new solution
			oldScore = calculateScore(population[nIdx], index);
			if (newScore < oldScore) {
				// Replace neighbour with new solution in new generation
				newGeneration[nIdx] = newInd;
			}

		}
	}

	/**
	 * Calculates the problem score for a given individual, using a given
	 * set of weights.
	 *
	 * @param ind
	 * @param problemIndex - for retrieving weights
	 * @return score
	 */
	private double calculateScore(Individual ind, int problemIndex) {
		double[] problemWeights = weights[problemIndex];
		double sum = 0;
		for (int i = 0; i < numObjectives; i++)
			sum += (problemWeights[i]) * ind.getObjectiveValues()[i];
		return sum;
	}

	/**
	 * Sorts the current population in order to identify the non-dominated
	 * solutions (i.e. the Pareto front).
	 *
	 * @param population
	 * @return Pareto front
	 */
	private Set<Individual> produceParetoFront(Individual[] population) {
		// Initialise sets/variable for tracking current front
		Set<Individual> front = new HashSet<Individual>();
		Set<Individual> toRemove = new HashSet<Individual>();
		boolean dominated = false;

		for (Individual i: population) {
			// Reset sets/variable
			front.clear();
			toRemove.clear();
			dominated = false;

			/* Go through front and check whether the current individual should be added to it. Also
			 * check whether any individuals currently in the front should be removed (i.e. whether
			 * they are dominated by the current individual).*/
			for (Individual f: front) {
				if (i.dominates(f)) {
					toRemove.add(f);
				}
				else if (f.dominates(i)) {
					dominated = true;
				}
			}
			// Remove all dominated points from the Pareto set, and add current individual if not dominated
			front.removeAll(toRemove);
			if(!dominated)
				front.add(i);
		}
		return front;
	}

	/**
	 * Saves program statistics to the disk. This method should be called once per generation.
	 *
	 * @param writer - File writer for output.
	 * @param generation - current generation number.
	 */
	private void writeOutStatistics(FileWriter writer, int generation) {
		try {
			for (Individual ind : population) {
				// Generation
				writer.append(String.format("%d ", generation));
				// Breeding time
				writer.append(String.format("%d ", breedingTime));
				// Evaluation time
				writer.append(String.format("%d ", evaluationTime));
				// Rank and sparsity
				writer.append("0 0 ");
				// Objective one
				writer.append(String.format("%f ", ind.getObjectiveValues()[0]));
				// Objective two
				writer.append(String.format("%f ", ind.getObjectiveValues()[1]));
				// Raw availability
				writer.append(String.format("%f ", ind.getAvailability()));
				// Raw reliability
				writer.append(String.format("%f ", ind.getReliability()));
				// Raw time
				writer.append(String.format("%f ", ind.getTime()));
				// Raw cost
				writer.append(String.format("%f\\n", ind.getCost()));
			}
		}
		catch (IOException e) {
			System.err.printf("Could not write to '%'.\n", outFileName);
			e.printStackTrace();
		}
	}

	/**
	 * Saves the Pareto front to the disk. This method should be called at the end of the run.
	 *
	 * @param writer - File writer for front.
	 * @param front - The set of individuals in the front.
	 */
	private void writeFrontStatistics(FileWriter writer, Set<Individual> front) {
		try {
			for (Individual ind : front) {
				// Rank and sparsity
				writer.append("0 0 ");
				// Objective one
				writer.append(String.format("%f ", ind.getObjectiveValues()[0]));
				// Objective two
				writer.append(String.format("%f ", ind.getObjectiveValues()[1]));
				// Raw availability
				writer.append(String.format("%f ", ind.getAvailability()));
				// Raw reliability
				writer.append(String.format("%f ", ind.getReliability()));
				// Raw time
				writer.append(String.format("%f ", ind.getTime()));
				// Raw cost
				writer.append(String.format("%f ", ind.getCost()));
				// Candidate string
				writer.append(String.format("%s\\n", ind.toString()));
			}
		}
		catch (IOException e) {
			System.err.printf("Could not write to '%'.\n", outFileName);
			e.printStackTrace();
		}
	}

	/**
	 * Parses the WSC Web service file with the given name, creating Web
	 * services based on this information and saving them to the service map.
	 *
	 * @param fileName
	 */
	private void parseWSCServiceFile(String fileName) {
        Set<String> inputs = new HashSet<String>();
        Set<String> outputs = new HashSet<String>();
        double[] qos = new double[4];

        try {
        	File fXmlFile = new File(fileName);
        	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        	Document doc = dBuilder.parse(fXmlFile);

        	NodeList nList = doc.getElementsByTagName("service");

        	for (int i = 0; i < nList.getLength(); i++) {
        		org.w3c.dom.Node nNode = nList.item(i);
        		Element eElement = (Element) nNode;

        		String name = eElement.getAttribute("name");
    		    qos[TIME] = Double.valueOf(eElement.getAttribute("Res"));
    		    qos[COST] = Double.valueOf(eElement.getAttribute("Pri"));
    		    qos[AVAILABILITY] = Double.valueOf(eElement.getAttribute("Ava"));
    		    qos[RELIABILITY] = Double.valueOf(eElement.getAttribute("Rel"));

				// Get inputs
				org.w3c.dom.Node inputNode = eElement.getElementsByTagName("inputs").item(0);
				NodeList inputNodes = ((Element)inputNode).getElementsByTagName("instance");
				for (int j = 0; j < inputNodes.getLength(); j++) {
					org.w3c.dom.Node in = inputNodes.item(j);
					Element e = (Element) in;
					inputs.add(e.getAttribute("name"));
				}

				// Get outputs
				org.w3c.dom.Node outputNode = eElement.getElementsByTagName("outputs").item(0);
				NodeList outputNodes = ((Element)outputNode).getElementsByTagName("instance");
				for (int j = 0; j < outputNodes.getLength(); j++) {
					org.w3c.dom.Node out = outputNodes.item(j);
					Element e = (Element) out;
					outputs.add(e.getAttribute("name"));
				}

                Service ws = new Service(name, qos, inputs, outputs);
                serviceMap.put(name, ws);
                inputs = new HashSet<String>();
                outputs = new HashSet<String>();
                qos = new double[4];
        	}
        }
        catch(IOException ioe) {
            System.out.println("Service file parsing failed...");
        }
        catch (ParserConfigurationException e) {
            System.out.println("Service file parsing failed...");
		}
        catch (SAXException e) {
            System.out.println("Service file parsing failed...");
		}
    }

	/**
	 * Parses the WSC task file with the given name, extracting input and
	 * output values to be used as the composition task.
	 *
	 * @param fileName
	 */
	private void parseWSCTaskFile(String fileName) {
		try {
	    	File fXmlFile = new File(fileName);
	    	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	    	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
	    	Document doc = dBuilder.parse(fXmlFile);

	    	org.w3c.dom.Node provided = doc.getElementsByTagName("provided").item(0);
	    	NodeList providedList = ((Element) provided).getElementsByTagName("instance");
	    	taskInput = new HashSet<String>();
	    	for (int i = 0; i < providedList.getLength(); i++) {
				org.w3c.dom.Node item = providedList.item(i);
				Element e = (Element) item;
				taskInput.add(e.getAttribute("name"));
	    	}

	    	org.w3c.dom.Node wanted = doc.getElementsByTagName("wanted").item(0);
	    	NodeList wantedList = ((Element) wanted).getElementsByTagName("instance");
	    	taskOutput = new HashSet<String>();
	    	for (int i = 0; i < wantedList.getLength(); i++) {
				org.w3c.dom.Node item = wantedList.item(i);
				Element e = (Element) item;
				taskOutput.add(e.getAttribute("name"));
	    	}
		}
		catch (ParserConfigurationException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
		catch (SAXException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
		catch (IOException e) {
            System.out.println("Task file parsing failed...");
            e.printStackTrace();
		}
	}

	/**
	 * Parses the WSC taxonomy file with the given name, building a
	 * tree-like structure.
	 *
	 * @param fileName
	 */
	private void parseWSCTaxonomyFile(String fileName) {
		try {
	    	File fXmlFile = new File(fileName);
	    	DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
	    	DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
	    	Document doc = dBuilder.parse(fXmlFile);
	    	NodeList taxonomyRoots = doc.getChildNodes();

	    	processTaxonomyChildren(null, taxonomyRoots);
		}

		catch (ParserConfigurationException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
		catch (SAXException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
		catch (IOException e) {
            System.err.println("Taxonomy file parsing failed...");
		}
	}

	/**
	 * Recursive function for recreating taxonomy structure from file.
	 *
	 * @param parent - Nodes' parent
	 * @param nodes
	 */
	private void processTaxonomyChildren(TaxonomyNode parent, NodeList nodes) {
		if (nodes != null && nodes.getLength() != 0) {
			for (int i = 0; i < nodes.getLength(); i++) {
				org.w3c.dom.Node ch = nodes.item(i);

				if (!(ch instanceof Text)) {
					Element currNode = (Element) nodes.item(i);
					String value = currNode.getAttribute("name");
					TaxonomyNode taxNode = taxonomyMap.get( value );
					if (taxNode == null) {
					    taxNode = new TaxonomyNode(value);
					    taxonomyMap.put( value, taxNode );
					}
					if (parent != null) {
					    taxNode.parents.add(parent);
						parent.children.add(taxNode);
					}

					NodeList children = currNode.getChildNodes();
					processTaxonomyChildren(taxNode, children);
				}
			}
		}
	}

	/**
	 * Converts input, output, and service instance values to their corresponding
	 * ontological parent.
	 */
	private void findConceptsForInstances() {
		Set<String> temp = new HashSet<String>();

		for (String s : taskInput)
			temp.add(taxonomyMap.get(s).parents.get(0).value);
		taskInput.clear();
		taskInput.addAll(temp);

		temp.clear();
		for (String s : taskOutput)
				temp.add(taxonomyMap.get(s).parents.get(0).value);
		taskOutput.clear();
		taskOutput.addAll(temp);

		for (Service s : serviceMap.values()) {
			temp.clear();
			Set<String> inputs = s.getInputs();
			for (String i : inputs)
				temp.add(taxonomyMap.get(i).parents.get(0).value);
			inputs.clear();
			inputs.addAll(temp);

			temp.clear();
			Set<String> outputs = s.getOutputs();
			for (String o : outputs)
				temp.add(taxonomyMap.get(o).parents.get(0).value);
			outputs.clear();
			outputs.addAll(temp);
		}
	}

	/**
	 * Populates the taxonomy tree by associating services to the
	 * nodes in the tree.
	 */
	private void populateTaxonomyTree() {
		for (Service s: serviceMap.values()) {
			addServiceToTaxonomyTree(s);
		}
	}

	private void addServiceToTaxonomyTree(Service s) {
		// Populate outputs
	    Set<TaxonomyNode> seenConceptsOutput = new HashSet<TaxonomyNode>();
		for (String outputVal : s.getOutputs()) {
			TaxonomyNode n = taxonomyMap.get(outputVal);
			s.getTaxonomyOutputs().add(n);

			// Also add output to all parent nodes
			Queue<TaxonomyNode> queue = new LinkedList<TaxonomyNode>();
			queue.add( n );

			while (!queue.isEmpty()) {
			    TaxonomyNode current = queue.poll();
		        seenConceptsOutput.add( current );
		        current.servicesWithOutput.add(s);
		        for (TaxonomyNode parent : current.parents) {
		            if (!seenConceptsOutput.contains( parent )) {
		                queue.add(parent);
		                seenConceptsOutput.add(parent);
		            }
		        }
			}
		}
		// Populate inputs
		Set<TaxonomyNode> seenConceptsInput = new HashSet<TaxonomyNode>();
		for (String inputVal : s.getInputs()) {
			TaxonomyNode n = taxonomyMap.get(inputVal);

			// Also add input to all children nodes
			Queue<TaxonomyNode> queue = new LinkedList<TaxonomyNode>();
			queue.add( n );

			while(!queue.isEmpty()) {
				TaxonomyNode current = queue.poll();
				seenConceptsInput.add( current );

			    Set<String> inputs = current.servicesWithInput.get(s);
			    if (inputs == null) {
			    	inputs = new HashSet<String>();
			    	inputs.add(inputVal);
			    	current.servicesWithInput.put(s, inputs);
			    }
			    else {
			    	inputs.add(inputVal);
			    }

			    for (TaxonomyNode child : current.children) {
			        if (!seenConceptsInput.contains( child )) {
			            queue.add(child);
			            seenConceptsInput.add( child );
			        }
			    }
			}
		}
		return;
	}

	/**
	 * Goes through the service list and retrieves only those services which
	 * could be part of the composition task requested by the user.
	 *
	 * @param serviceMap
	 * @return relevant services
	 */
	private Set<Service> getRelevantServices(Map<String,Service> serviceMap, Set<String> inputs, Set<String> outputs) {
		// Copy service map values to retain original
		Collection<Service> services = new ArrayList<Service>(serviceMap.values());

		Set<String> cSearch = new HashSet<String>(inputs);
		Set<Service> sSet = new HashSet<Service>();
		int layer = 0;
		Set<Service> sFound = discoverService(services, cSearch);
		while (!sFound.isEmpty()) {
			sSet.addAll(sFound);
			// Record the layer that the services belong to in each node
			for (Service s : sFound)
				s.layer = layer;

			layer++;
			services.removeAll(sFound);
			for (Service s: sFound) {
				cSearch.addAll(s.getOutputs());
			}
			sFound.clear();
			sFound = discoverService(services, cSearch);
		}

		numLayers = layer;

		if (isSubsumed(outputs, cSearch)) {
			return sSet;
		}
		else {
			String message = "It is impossible to perform a composition using the services and settings provided.";
			System.out.println(message);
			System.exit(0);
			return null;
		}
	}

	/**
	 * Discovers all services from the provided collection whose
	 * input can be satisfied either (a) by the input provided in
	 * searchSet or (b) by the output of services whose input is
	 * satisfied by searchSet (or a combination of (a) and (b)).
	 *
	 * @param services
	 * @param searchSet
	 * @return set of discovered services
	 */
	private Set<Service> discoverService(Collection<Service> services, Set<String> searchSet) {
		Set<Service> found = new HashSet<Service>();
		for (Service s: services) {
			if (isSubsumed(s.getInputs(), searchSet))
				found.add(s);
		}
		return found;
	}

	/**
	 * Checks whether set of inputs can be completely satisfied by the search
	 * set, making sure to check descendants of input concepts for the subsumption.
	 *
	 * @param inputs
	 * @param searchSet
	 * @return true if search set subsumed by input set, false otherwise.
	 */
	public boolean isSubsumed(Set<String> inputs, Set<String> searchSet) {
		boolean satisfied = true;
		for (String input : inputs) {
			Set<String> subsumed = taxonomyMap.get(input).getSubsumedConcepts();
			if (!isIntersection( searchSet, subsumed )) {
				satisfied = false;
				break;
			}
		}
		return satisfied;
	}

    private static boolean isIntersection( Set<String> a, Set<String> b ) {
        for ( String v1 : a ) {
            if ( b.contains( v1 ) ) {
                return true;
            }
        }
        return false;
    }

	private void calculateNormalisationBounds(Set<Service> services) {
		for(Service service: services) {
			double[] qos = service.getQos();

			// Availability
			double availability = qos[AVAILABILITY];
			if (availability > maxAvailability)
				maxAvailability = availability;

			// Reliability
			double reliability = qos[RELIABILITY];
			if (reliability > maxReliability)
				maxReliability = reliability;

			// Time
			double time = qos[TIME];
			if (time > maxTime)
				maxTime = time;
			if (time < minTime)
				minTime = time;

			// Cost
			double cost = qos[COST];
			if (cost > maxCost)
				maxCost = cost;
			if (cost < minCost)
				minCost = cost;
		}
		// Adjust max. cost and max. time based on the number of services in shrunk repository
		maxCost *= services.size();
		maxTime *= services.size();

	}

	/**
	 * Application entry point.
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		if (args.length == 0)
			throw new RuntimeException("A parameters file should be provided as an argument.");
		new MOEAD(args);
	}
}
