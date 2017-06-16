package moead;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Random;

import representation.IndirectCrossoverOperator;
import representation.IndirectIndividual;
import representation.IndirectMutationOperator;

/**
 * The entry class for the program. The main algorithm is executed once MOEAD is instantiated.
 *
 * @author sawczualex
 */
public class MOEAD {
	public static final int seed = 1;
	public static final int generations = 51;
	public static final int popSize = 500;
	public static final int numObjectives = 2;
	public static final int numNeighbours = 30;
	public static final double crossoverProbability = 0.8;
	public static final double mutationProbability = 0.2;
	public static final StoppingCriteria stopCrit = new GenerationStoppingCriteria(generations);
	public static final Individual indType = new IndirectIndividual();
	public static final MutationOperator mutOperator = new IndirectMutationOperator();
	public static final CrossoverOperator crossOperator = new IndirectCrossoverOperator();

	private Individual[] population = new Individual[popSize];
	private List<Individual> paretoFront;
	private double[][] weights = new double[popSize][numObjectives];
	private int[][] neighbourhood = new int[popSize][numNeighbours];
	private Random random;

	public MOEAD() {
		// Initialise
		initialise();
		// While stopping criteria not met
		while(!stopCrit.stoppingCriteriaMet()) {
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
		}
		// Produce final Pareto front
		paretoFront = produceParetoFront(population);
	}

	/**
	 * Creates weight vectors, calculates neighbourhoods, and generates an initial population.
	 */
	public void initialise() {
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
			return crossOperator.doCrossover(original.clone(), neighbour.clone(), random);
		}
		// Else, perform mutation
		else {
			return mutOperator.mutate(original.clone(), random);
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
	private List<Individual> produceParetoFront(Individual[] population) {
		// TODO: implement this
		return null;
	}

	/**
	 * Application entry point.
	 *
	 * @param args
	 */
	public static void main(String[] args) {
		new MOEAD();
	}
}
