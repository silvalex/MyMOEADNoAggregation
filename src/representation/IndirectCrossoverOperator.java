package representation;

import java.util.HashSet;
import java.util.Set;

import moead.CrossoverOperator;
import moead.Individual;
import moead.MOEAD;
import moead.Service;

public class IndirectCrossoverOperator extends CrossoverOperator {

	@Override
	public Individual doCrossover(Individual ind1, Individual ind2, MOEAD init) {
		if (!(ind1 instanceof IndirectIndividual) || !(ind2 instanceof IndirectIndividual))
			throw new RuntimeException("IndirectCrossoverOperator can only work on objects of type IndirectIndividual.");
		IndirectIndividual t1 = ((IndirectIndividual)ind1);
		IndirectIndividual t2 = ((IndirectIndividual)ind2);

		// Select two random index numbers as the boundaries for the crossover section
		int indexA = init.random.nextInt(t1.getGenome().length);
		int indexB = init.random.nextInt(t1.getGenome().length);

		// Make sure they are different
		while (indexA == indexB)
			indexB = init.random.nextInt(t1.getGenome().length);

		// Determine which boundary they are
		int minBoundary = Math.min(indexA, indexB);
		int maxBoundary = Math.max(indexA, indexB);

		// Create new individuals
		IndirectIndividual newInd1 = new IndirectIndividual();
		IndirectIndividual newInd2 = new IndirectIndividual();
		newInd1.setInit(init);
		newInd2.setInit(init);
		newInd1.createNewGenome();
		newInd2.createNewGenome();

		// Swap crossover sections between candidates, keeping track of which services are in each section
		Set<Service> newSection1 = new HashSet<Service>();
		Set<Service> newSection2 = new HashSet<Service>();

		for (int index = minBoundary; index <= maxBoundary; index++) {
			// Copy section from parent 1 to genome 2
			newInd2.getGenome()[index] = t1.getGenome()[index];
			newSection2.add(t1.getGenome()[index]);

			// Copy section from parent 2 to genome 1
			newInd1.getGenome()[index] = t2.getGenome()[index];
			newSection1.add(t2.getGenome()[index]);
		}

		// Now fill the remainder of the new genomes, making sure not to duplicate any services
		fillNewGenome(t2, newInd2.getGenome(), newSection2, minBoundary, maxBoundary);
		fillNewGenome(t1, newInd1.getGenome(), newSection1, minBoundary, maxBoundary);
		
		// Evaluate the two children, and return the better one if there is a dominance
		t1.evaluate();
		t2.evaluate();
		
		if (t1.dominates(t2))
			return t1;
		else if(t2.dominates(t1))
			return t2;
		else {
			if (init.random.nextBoolean())
				return t1;
			else
				return t2;
		}
	}

	private void fillNewGenome(IndirectIndividual parent, Service[] newGenome, Set<Service> newSection, int minBoundary, int maxBoundary) {
		int genomeIndex = getInitialIndex(minBoundary, maxBoundary);
	
		for (int i = 0; i < parent.getGenome().length; i++) {
			if (genomeIndex >= newGenome.length + 1)
				break;
			// Check that the service has not been already included, and add it
			if (!newSection.contains(parent.getGenome()[i])) {
				newGenome[genomeIndex] = parent.getGenome()[i];
				// Increment genome index
				genomeIndex = incrementIndex(genomeIndex, minBoundary, maxBoundary);
			}
		}
	}
		
	private int getInitialIndex(int minBoundary, int maxBoundary) {
		if (minBoundary == 0)
			return maxBoundary + 1;
		else
			return 0;
	}

	private int incrementIndex(int currentIndex, int minBoundary, int maxBoundary) {
		if (currentIndex + 1 >= minBoundary && currentIndex + 1 <= maxBoundary)
			return maxBoundary + 1;
		else
			return currentIndex + 1;
	}
}
