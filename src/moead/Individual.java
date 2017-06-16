package moead;

/**
 * Represents an individual in the population of solutions. A generation method for
 * the given individual must be implemented.
 *
 * @author sawczualex
 */
public abstract class Individual implements Cloneable {
	/**
	 * Randomly generates a new individual.
	 *
	 * @return new individual
	 */
	public abstract Individual generateIndividual();

	/**
	 * Returns the objective values for this individual.
	 *
	 * @return array of objective values
	 */
	public abstract double[] getObjectiveValues();

	@Override
	/**
	 * {@inheritDoc}
	 */
	public abstract Individual clone();
}
