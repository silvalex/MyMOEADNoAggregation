package moead;

/**
 * Uses a given fixed number of generations as the stopping criterion.
 * The desired maximum number of generations is passed into the constructor.
 * 
 * @author sawczualex
 */
public class GenerationStoppingCriteria extends StoppingCriteria {
	private int maxGenerations;
	private int currentGen = 0;
	
	public GenerationStoppingCriteria(int gens) {
		maxGenerations = gens;
	}

	@Override
	/**
	 *{@inheritDoc}
	 */
	public boolean stoppingCriteriaMet() {
		if (currentGen++ < maxGenerations) {
			System.out.printf("Generation %d\n", currentGen);
			return false;
		}
		else
			return true;
	}

}
