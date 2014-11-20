package edu.harvard.econcs.peerprediction.analysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

public class ResultsGatherLogs {

	public static void main(String[] args) throws IOException {

		if (args.length < 3) {
			System.err.println("Please provide parametes: treatment, model, dir");
			System.exit(0);
		}
		String treatment = args[0];
		String model = args[1];
		String dir = args[2];

		String separator = System.getProperty("file.separator");
		
		String folderPath = String.format("%s%spplogs%s", dir, separator,
				separator);

		BufferedWriter writer = new BufferedWriter(new FileWriter(
				String.format("%s%s-%s.csv", folderPath, treatment, model)));
		BufferedReader reader = null;

		final String nameStart = String.format("%s-%s", treatment, model);

		File folder = new File(folderPath);
		File[] logFiles = folder.listFiles(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String name) {
				if (name.startsWith(nameStart) && name.endsWith(".log"))
					return true;
				return false;
			}
		});

		List<String> results = new ArrayList<String>();

		for (int i = 0; i < logFiles.length; i++) {
			File file = logFiles[i];
			reader = new BufferedReader(new FileReader(file));
			
			String line = reader.readLine();
			
			// save lines to list
			results.add(line);

			// write to csv file
			writer.write(line + "\n");
			
			reader.close();
		}
		writer.newLine();

		
		
		Map<String, DescriptiveStatistics> trainingLoglk = new HashMap<String, DescriptiveStatistics>();
		Map<String, DescriptiveStatistics> testLoglk = new HashMap<String, DescriptiveStatistics>();

		for (String result : results) {
			String[] comps = result.split(",");
			String seed = comps[0];
			Double trainll = Double.parseDouble(comps[2]);
			Double testll = Double.parseDouble(comps[3]);

			// update training loglk
			DescriptiveStatistics stats = null; 
			if (trainingLoglk.containsKey(seed)) {
				stats = trainingLoglk.get(seed);
			} else {
				stats = new DescriptiveStatistics();
			}
			stats.addValue(trainll);
			trainingLoglk.put(seed, stats);

			// update test loglk
			stats = null;
			if (testLoglk.containsKey(seed)) {
				stats = testLoglk.get(seed);
			} else {
				stats = new DescriptiveStatistics();
			}
			stats.addValue(testll);
			testLoglk.put(seed, stats);

		}

		int numRounds = 10;
		int numFolds = 10;
		writeSummary(writer, trainingLoglk, numFolds, numRounds, "Training");
		writeSummary(writer, testLoglk, numFolds, numRounds, "Test");

		writer.flush();
		writer.close();
	}

	private static void writeSummary(BufferedWriter writer,
			Map<String, DescriptiveStatistics> trainingLoglk, int numFolds, int numRounds,
			String title) throws IOException {

		DescriptiveStatistics stats = new DescriptiveStatistics();
		
		writer.write(title + " loglks\n");
		for (String key : trainingLoglk.keySet()) {
			double loglk = trainingLoglk.get(key).getMean();
			stats.addValue(loglk);
			writer.write(loglk + "\n");
		}
		writer.newLine();
		
		writer.write(title + " loglks summary:\n");
		writer.write(String.format("mean,%.2f\n", stats.getMean()));
		writer.write(String.format("standard deviation,%.2f\n",
				stats.getStandardDeviation()));
		writer.write(String.format("sample size,10\n"));
		double moe = 1.96 * stats.getStandardDeviation() / Math.sqrt(10);
		writer.write(String.format("margin of error,%.2f\n", moe));
		writer.write(String.format("upper bound,%.2f\n", stats.getMean() + moe));
		writer.write(String.format("lower bound,%.2f\n", stats.getMean() - moe));
		writer.newLine();

	}

}
