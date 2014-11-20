package edu.harvard.econcs.peerprediction.analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class ResultsWritePBSScript {

	public static void main(String[] args) throws IOException {
	
		genScript("s1");
		genScript("s2-abs");
		genScript("s2-rel");
		genScript("s3-abs");
		genScript("s3-rel");
		genScript("s4");
		genScript("s5");
	}

	private static void genScript(String model ) throws IOException {
		String treatment = "prior2-basic";

		String filename = String.format("array-g1-%s.pbs", model);
		String homeDir = System.getProperty("user.home");
		String separator = System.getProperty("file.separator");
		String script = homeDir + separator + "Documents" + separator + "workspace" + separator + filename;

		int numFolds = 10;
		int numRounds = 10;
		int totalNum = numRounds*numFolds;
		BufferedWriter writer = new BufferedWriter(new FileWriter(script));
		writer.write(String.format("#!/bin/sh\n"
				+ "#PBS -N g1-%s\n"
				+ "#PBS -M xigao@cs.ubc.ca\n"
				+ "#PBS -m a\n"
				+ "#PBS -j oe\n"
				+ "#PBS -o /global/scratch/alicegao/outputs/g1-%s\n"
				+ "#PBS -l mem=9000mb,walltime=12:00:00\n"
				+ "#PBS -t 1-%d\n"
//				+ "module load java/jdk7u45\n"  // grex
				+ "module load java/1.7.0\n"  // bugaboo
				+ "case $PBS_ARRAYID in\n", model, model, totalNum));
		
		int seedStart = 1000;
		int index = 1;
		for (int seed = seedStart; seed < seedStart + numRounds; seed++) {
			for (int f = 0; f < numFolds; f++) {
				writer.write(index + ")\n");
				writer.write(String.format(
						"java -jar ppanalysis.jar %s %s %s %d;;\n", treatment, model, seed, f));
				index++;
			}
		}
		writer.write("esac\n");
		writer.flush();
		writer.close();
	}
	
}
