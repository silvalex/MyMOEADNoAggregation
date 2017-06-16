package moead;

/**
 * Represents an individual in the population of solutions. A generation method for
 * the given individual must be implemented.
 *
 * @author sawczualex
 */
public abstract class Individual implements Cloneable {
	/**
	 *
	 * @return
	 */
	public abstract Individual generateIndividual();
	public abstract double[] getObjectiveValues();
	@Override
	/**
	 * {@inheritDoc}
	 */
	public abstract Individual clone();
}
