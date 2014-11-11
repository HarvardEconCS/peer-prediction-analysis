package edu.harvard.econcs.peerprediction.analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.andrewmao.models.games.BWToleranceLearner;
import net.andrewmao.models.games.SigActObservation;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import be.ac.ulg.montefiore.run.jahmm.Hmm;

public class PredLkAnalysis {

	static String rootDir = "/Users/alicexigao/Dropbox/peer-prediction/data/"
			+ PredLkAnalysis.treatment + "/";
	
	static String treatment = "prior2-basic";
	//	 static String treatment = "prior2-outputagreement";
	//	 static String treatment = "prior2-uniquetruthful";
	//	 static String treatment = "prior2-symmlowpay";
	//	 static String treatment = "prior2-constant";
	
	static int numFolds = 10;
	static int numRoundsCV = 1;
	static double randomLogLk; 
	static Map<String, List<Double>> avgLogLks;

	public static void main(String[] args) throws Exception {
			
		// set root directory for data
		String homeDir = System.getProperty("user.home");
		String separator = System.getProperty("file.separator");
		rootDir = homeDir + separator + "ppdata" + separator + PredLkAnalysis.treatment + separator;
		
		LogReader.parseTextfile();
		LogReader.printTreatmentInfo();

		if (args.length < 2) {
			System.err.println("Please provide a model and a treatment");
			System.exit(0);
		}

		String model = args[0];
		treatment = args[1];

		avgLogLks = new HashMap<String, List<Double>>();
		randomLogLk = getLogLkRandomModel();
		
		printCurrentDateTime();
		getPredictiveLogLk(model);
		printCurrentDateTime();
		
		// models: HMM, s1, s1-1, s2, s3-abs, s3-rel

//		graphPredictiveLogLk();
	}
	
	public static void getPredictiveLogLk(String model) throws IOException {
	
		// stdout to both console and file
//		Date date = new Date();
//		SimpleDateFormat ft = new SimpleDateFormat("MM.dd.HH.mm.ss");
//		FileOutputStream f = new FileOutputStream(String.format(
//				"%slog-%s-%s.txt", PredLkAnalysis.rootDir, model, ft.format(date)));
//		TeePrintStream tee = new TeePrintStream(f, System.out);
//		System.setOut(tee);

		System.out.println("Get predictive likelihood for " + model);
		int groupSize = LogReader.expSet.games.size() / PredLkAnalysis.numFolds;
		DescriptiveStatistics stats = new DescriptiveStatistics();
	
		System.out.printf("number of folds per round: %d\n", numFolds);

		Collections.shuffle(LogReader.expSet.games);

		for (int i = 0; i < PredLkAnalysis.numFolds; i++) {

			System.out.printf("Fold %d:\n", i);

			// Divide up data into test and training sets
			List<Game> testSet = new ArrayList<Game>();
			List<Game> trainingSet = new ArrayList<Game>();
			int testStart = i * groupSize;
			for (int j = 0; j < PredLkAnalysis.numFolds * groupSize; j++) {
				if (j >= testStart && j < testStart + groupSize) {
					testSet.add(LogReader.expSet.games.get(j));
				} else {
					trainingSet.add(LogReader.expSet.games.get(j));
				}
			}

			// Estimate best parameters on training set
			Map<String, Object> bestParam = estimateParams(model, trainingSet);
			if (!model.equals("HMM"))
				Utils.printParams(bestParam);

			// Compute loglk on test set
			double testLoglk = getTestLogLk(model, bestParam, testSet)
					- randomLogLk;
			stats.addValue(testLoglk);
			System.out.println("test loglk = " + testLoglk);

			printCurrentDateTime();
		}
		
		// Print summary
		System.out.println();
		System.out.println("Summary:\n"
				+ "Test loglks are: ");
		for (int i = 0; i < stats.getN(); i++) {
			System.out.println(stats.getElement(i));
		}
		System.out.println();
		System.out.printf("avg test loglk = %.2f\n", stats.getMean());

	}

	static Map<String, Object> estimateParams(String model,
			List<Game> trainingSet) {
	
		Map<String, Object> params = new HashMap<String, Object>();
	
		if (model.startsWith("s2") || model.startsWith("s3") || model.equals("s1") || model.equals("s1-1")) {
	
			double[] point = LearningModelsCustom.estimateUsingCobyla(model, trainingSet);
			params = LearningModelsCustom.pointToMap(model, point);
	
		} else if (model.equals("HMM")) {
	
			return HMMAnalysis.estimateHMM(trainingSet);
	
		} else if (model.equals("RLS")) {
	
			double[] point = LearningModelsExisting.estimateUsingApacheOptimizer(trainingSet, "RLS");
	
			params.put("phi", point[0]);
			params.put("lambda", point[1]);
			params.put("considerSignal", true);
	
		} else if (model.equals("RLNS")) {
	
			double[] point = LearningModelsExisting.estimateUsingApacheOptimizer(trainingSet, "RLNS");
	
			params.put("phi", point[0]);
			params.put("lambda", point[1]);
			params.put("considerSignal", false);
	
		} else if (model.equals("SFPS")) {
	
			double[] point = LearningModelsExisting.estimateUsingApacheOptimizer(trainingSet, "SFPS");
	
			params.put("rho", point[0]);
			params.put("lambda", point[1]);
			params.put("considerSignal", true);
	
		} else if (model.equals("SFPNS")) {
	
			double[] point = LearningModelsExisting.estimateUsingApacheOptimizer(trainingSet, "SFPNS");
	
			params.put("rho", point[0]);
			params.put("lambda", point[1]);
			params.put("considerSignal", false);
	
		} else if (model.equals("EWAS")) {
	
			double[] point = LearningModelsExisting.estimateUsingApacheOptimizer(trainingSet, "EWAS");
	
			params.put("rho", point[0]);
			params.put("phi", point[1]);
			params.put("delta", point[2]);
			params.put("lambda", point[3]);
			params.put("considerSignal", true);
	
		} else if (model.equals("EWANS")) {
	
			double[] point = LearningModelsExisting.estimateUsingApacheOptimizer(trainingSet, "EWANS");
	
			params.put("rho", point[0]);
			params.put("phi", point[1]);
			params.put("delta", point[2]);
			params.put("lambda", point[3]);
			params.put("considerSignal", false);
		}
	
		return params;
	}

	static double getTestLogLk(String model,
			Map<String, Object> bestParam, List<Game> testSet) {
	
		if (model.equals("HMM")) {
	
			@SuppressWarnings("unchecked")
			Hmm<SigActObservation<CandySignal, CandyReport>> bestHmm = (Hmm<SigActObservation<CandySignal, CandyReport>>) bestParam
					.get("HMM");
			List<List<SigActObservation<CandySignal, CandyReport>>> seq = HMMAnalysis
					.getActObsSequence(testSet);
			return BWToleranceLearner.computeLogLk(bestHmm, seq);
	
		} else if (model.startsWith("RL")) {
	
			return LearningModelsExisting.computeLogLkRL(bestParam, testSet);
	
		} else if (model.startsWith("SFP")) {
	
			return LearningModelsExisting.computeLogLkSFP(bestParam, testSet);
	
		} else if (model.startsWith("EWA")) {
	
			return LearningModelsExisting.computeLogLkEWA(bestParam, testSet);
	
		} else if (model.startsWith("s2") || model.startsWith("s3") 
				|| model.equals("s1") || model.equals("s1-1")) {
			return LearningModelsCustom.computeLogLk(model, bestParam, testSet);
	
		}
	
		return 0;
	}

	static double getLogLkRandomModel() {
		int groupSize = LogReader.expSet.games.size() / 10;
		double randomLogLk = Math.log(0.5) * groupSize * LogReader.expSet.numRounds
				* LogReader.expSet.numPlayers;
		return randomLogLk;
	}

	public static void graphPredictiveLogLk() throws IOException {
	
		System.out.println("Graphing distributions of log likelihoods");
		BufferedWriter writer = new BufferedWriter(new FileWriter(PredLkAnalysis.rootDir
				+ "predictiveLogLk.m"));
	
		int numModels = avgLogLks.keySet().size();
		List<String> modelNames = new ArrayList<String>();
		double[] displayMeans = new double[numModels];
		double[] displayErrors = new double[numModels];
		int index = 0;
	
		DescriptiveStatistics stats = new DescriptiveStatistics();
	
		modelNames.addAll(avgLogLks.keySet());
		Collections.sort(modelNames);
		for (int i = 0; i < modelNames.size(); i++) {
			String model = modelNames.get(i);
			stats.clear();
			List<Double> list = avgLogLks.get(model);
			for (int j = 0; j < list.size(); j++) {
				stats.addValue(list.get(j));
			}
			displayMeans[index] = stats.getMean();
			double stdev = stats.getStandardDeviation();
			double stdError = stdev / Math.sqrt(numRoundsCV);
			displayErrors[index] = 1.96 * stdError;
			index++;
		}
	
		StringBuilder sb = new StringBuilder();
		sb.append(PredLkAnalysis.treatment);
		sb.append("\n");
		for (int i = 0; i < numModels; i++) {
			sb.append(String.format("'%s',", modelNames.get(i)));
		}
		sb.append("\n");
		sb.append(String.format("mean = ["));
		for (int i = 0; i < numModels; i++) {
			sb.append(String.format("%.2f ", displayMeans[i]));
		}
		sb.append(String.format("];\n" + "error = ["));
		for (int i = 0; i < numModels; i++) {
			sb.append(String.format("%.2f ", displayErrors[i]));
		}
		sb.append(String.format("];\n"));
		writer.write(sb.toString());
	
		writer.flush();
		writer.close();
	
	}

	static void printCurrentDateTime() {
		Date now = new Date();
		SimpleDateFormat ft = new SimpleDateFormat(
				"E yyyy.MM.dd 'at' hh:mm:ss a zzz");
		System.out.println("Current Date: " + ft.format(now));
	}



}
