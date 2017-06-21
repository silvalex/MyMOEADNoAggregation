package representation;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import moead.Individual;
import moead.Service;

public class IndirectIndividual extends Individual {
	private double availability;
	private double reliability;
	private double time;
	private double cost;
	private Service[] genome;
	private double[] objectives;

	@Override
	public Individual generateIndividual(List<Service> relevantList, Random random) {
		Collections.shuffle(relevantList, random);

		IndirectIndividual newInd = new IndirectIndividual();
		newInd.genome = new Service[relevantList.size()];
		relevantList.toArray(newInd.genome);

		return newInd;
	}

	@Override
	public Individual clone() {
		IndirectIndividual newInd = new IndirectIndividual();

		// Shallow cloning is fine in this approach
		newInd.genome = genome.clone();

		newInd.availability = availability;
		newInd.reliability = reliability;
		newInd.time = time;
		newInd.cost = cost;
		newInd.objectives = new double[objectives.length];

		System.arraycopy(objectives, 0, newInd.objectives, 0, objectives.length);

		return newInd;
	}

	@Override
	public double[] getObjectiveValues() {
		return objectives;
	}

	@Override
	public void setObjectiveValues(double[] newObjectives) {
		objectives = newObjectives;
	}

	@Override
	public double getAvailability() {
		return availability;
	}

	@Override
	public double getReliability() {
		return reliability;
	}

	@Override
	public double getTime() {
		return time;
	}

	@Override
	public double getCost() {
		return cost;
	}

	@Override
	public String toString() {
		// TODO: Implement this and also an evaluation function
		WSCInitializer init = (WSCInitializer) state.initializer;
		Graph g = createNewGraph(init.numLayers, init.startServ, init.endServ, genome, init);
		return g.toString();
	}



}
