package moead;

/**
 * Abstract class for implementing different stopping criteria. Any specific information required by the
 * class should be provided to the constructor, and a method for verifying whether the stopping criteria
 * have been met should be implemented.
 * 
 * @author sawczualex
 */
public abstract class StoppingCriteria {
	
	/**
	 * Verifies whether stopping criteria have been met.
	 * 
	 * @return true if met, false otherwise.
	 */
	public abstract boolean stoppingCriteriaMet();
}
