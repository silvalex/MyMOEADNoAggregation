package moead;

import java.util.Random;

/**
 * Represents an abstract mutation operator, to be implemented according to the
 * chosen representation.
 *
 * @author sawczualex
 */
public abstract class MutationOperator {
	/**
	 * Mutates the given individual, returning a modified version.
	 *
	 * @param ind - the original individual
	 * @param rand - random object for stochastic operations
	 * @return Mutated individual
	 */
	public abstract Individual mutate(Individual ind, Random rand);
}
