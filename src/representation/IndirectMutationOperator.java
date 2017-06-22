package representation;

import moead.Individual;
import moead.MOEAD;
import moead.MutationOperator;
import moead.Service;

public class IndirectMutationOperator extends MutationOperator {

	@Override
	public Individual mutate(Individual ind, MOEAD init) {
		if (!(ind instanceof IndirectIndividual))
			throw new RuntimeException("IndirectMutationOperator can only work on objects of type IndirectIndividual.");
		IndirectIndividual indirect = (IndirectIndividual) ind;
		
		int indexA = init.random.nextInt(indirect.getGenome().length);
    	int indexB = init.random.nextInt(indirect.getGenome().length);
    	swapServices(indirect.getGenome(), indexA, indexB);
        indirect.evaluate();

		return indirect;
	}
	
	private void swapServices(Service[] genome, int indexA, int indexB) {
		Service temp = genome[indexA];
		genome[indexA] = genome[indexB];
		genome[indexB] = temp;
	}

}
