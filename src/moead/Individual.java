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

	/**
	 * Returns the overall availability (raw) for this individual.
	 *
	 * @return availability
	 */
	public abstract double getAvailability();

	/**
	 * Returns the overall reliability (raw) for this individual.
	 *
	 * @return reliability
	 */
	public abstract double getReliability();

	/**
	 * Returns the overall time (raw) for this individual.
	 *
	 * @return time
	 */
	public abstract double getTime();

	/**
	 * Returns the overall cost (raw) for this individual.
	 *
	 * @return cost
	 */
	public abstract double getCost();

	@Override
	/**
	 * {@inheritDoc}
	 */
	public abstract Individual clone();

	@Override
	/**
	 * {@inheritDoc}
	 */
	public abstract String toString();

}
