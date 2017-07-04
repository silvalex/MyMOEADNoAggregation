package moead;

/**
 * Represents an abstract crossover operator, to be implemented according to the
 * chosen representation.
 *
 * @author sawczualex
 */
public abstract class CrossoverOperator {
	/**
	 * Performs a crossover between the two individuals provided, returning the resulting individual
	 * with the highest quality.
	 *
	 * @param ind1
	 * @param ind2
	 * @param rand
	 * @return offspring
	 */
	public abstract Individual doCrossover(Individual ind1, Individual ind2, MOEAD init);
}
