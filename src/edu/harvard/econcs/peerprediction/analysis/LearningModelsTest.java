package edu.harvard.econcs.peerprediction.analysis;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.andrewmao.math.RandomSelection;
import net.andrewmao.misc.Pair;

import org.junit.Before;
import org.junit.Test;

public class LearningModelsTest {

	Random rand = new Random();
	static String[] signalList = new String[] { "MM", "GB" };

	@Before
	public void setUp() throws IOException {
		String homeDir = System.getProperty("user.home");
		String separator = System.getProperty("file.separator");
		PredLkAnalysis.rootDir = homeDir + separator + "ppdata" + separator
				+ PredLkAnalysis.treatment + separator;

		PredLkAnalysis.treatment = "prior2-basic";
		LogReader.parseTextfile();
		LogReader.printTreatmentInfo();
	}

	@Test
	public void tesBoundsCobyla() {

		helperBounds("s1", true, "eps", 0.5);
		helperBounds("s1", false, "eps", 0.0);

		helperBounds("s3-abs", true, "delta", (1.5 - 0.1)
				* LogReader.expSet.numRounds);
		helperBounds("s3-abs", false, "delta", 0.0);
		helperBounds("s3-rel", true, "delta", (1.5 / 0.1)
				* LogReader.expSet.numRounds);
		helperBounds("s3-rel", false, "delta", 1.0);

	}

	void helperBounds(String model, boolean isUpper, String paramName,
			double expected) {
		double bound = 0.0;
		if (isUpper)
			bound = LearningModelsCustom.getUBCobyla(model, paramName);
		else
			bound = LearningModelsCustom.getLBCobyla(model, paramName);
		assertEquals(expected, bound, Utils.eps);

	}
	
	@Test
	public void testStrategyIndexToString() {
		assertEquals("TR", LearningModelsCustom.strategyIndexToString("s3", 0));
		assertEquals("MM", LearningModelsCustom.strategyIndexToString("s3", 1));
		assertEquals("GB", LearningModelsCustom.strategyIndexToString("s3", 2));
		assertEquals("OP", LearningModelsCustom.strategyIndexToString("s3", 3));
		assertEquals("RA", LearningModelsCustom.strategyIndexToString("s3", 4));
		assertEquals("Unrecognized", LearningModelsCustom.strategyIndexToString("s3", 5));
		
		assertEquals("TR", LearningModelsCustom.strategyIndexToString("s2", 0));
		assertEquals("MM", LearningModelsCustom.strategyIndexToString("s2", 1));
		assertEquals("GB", LearningModelsCustom.strategyIndexToString("s2", 2));
		assertEquals("OP", LearningModelsCustom.strategyIndexToString("s2", 3));
		assertEquals("RA", LearningModelsCustom.strategyIndexToString("s2", 4));
		assertEquals("RA", LearningModelsCustom.strategyIndexToString("s2", 5));
		
	}

	@Test
	public void testS1() {
		testModel("s1", 3);
	}
	
	@Test
	public void testS3ABS() {
		testModel("s3-abs", 1);
	}
	
	@Test
	public void testS2ABS() {
		testModel("s2-abs", 1);
	}
	
	public void testModel(String model, int numTests) {
		int numGames = 400;
		System.out.print("\n=============================\n");
		System.out.printf("Model %s - %d Test(s)", model, numTests);

		for (int i = 0; i < numTests; i++) {

			System.out.println();
			System.out.printf("Model %s - Test %d\n", model, i);

			// expected parameters
			double[] point = LearningModelsCustom.getRandomPoint(model);
			Map<String, Object> params = LearningModelsCustom.pointToMap(model, point);
			params.put("delta", 5.0);
			params.put("eps", 0.07);
			System.out.printf("Expected parameters: ");
			Utils.printParams(params);
			
			testHelper(numGames, params, model);
		}

	}

	void testHelper(int numGames, Map<String, Object> expectedParams,
			String model) {

		// simulate a set of games
		List<Game> games = new ArrayList<Game>();
		for (int i = 0; i < numGames; i++) {
			Game game = LearningModelsTest.simulate(model, expectedParams);
			games.add(game);
		}

		// estimate model
		double[] point = LearningModelsCustom.estimateUsingCobyla(model, games);
		Map<String, Object> actualParams = LearningModelsCustom.pointToMap(model, point);
		
		// print expected and actual parameters
		System.out.printf("Expected parameters: ");
		Utils.printParams(expectedParams);
		System.out.printf("Actual parameters: ");
		Utils.printParams(actualParams);

		// compare parameters
//		String[] paramNames = getParamNames(model);
//		for (String name : paramNames) {
//			assertEquals((double) expectedParams.get(name),
//					(double) actualParams.get(name), 0.1);
//		}
	}

//	private String[] getParamNames(String model) {
//		if (model.equals("s1")) {
//			return new String[] { "probTR", "probMM", "probGB", "probOP", "eps" };
//		} else if (model.startsWith("s3")) {
//			return new String[] { "probTR", "probMM", "probGB", "probOP",
//					"eps", "delta" };
//		} else if (model.startsWith("s2")) {
//			return new String[] { "probTR", "probMM", "probGB", "probOP",
//					"eps", "delta", "probRA" };
//		}
//		return null;
//	}

	private static Game simulate(String model,
			Map<String, Object> expectedParams) {
		if (model.startsWith("s3")) {
			return simulateS3(expectedParams);
		} else if (model.startsWith("s2")) {
			return simulateS2(expectedParams);
		} else if (model.equals("s1")) {
			return simulateS1(expectedParams);
		} 
		return null;
	}

	static Game simulateS2(Map<String, Object> params) {
		
		String model = "s2";
		boolean isAbs = (boolean) params.get("isAbs");
		double eps = (double) params.get("eps");
		double delta = (double) params.get("delta");
		
		Game game = initGame();
		
		int[] strategyIndices = initStrategyIndices(model, params);
		String[] strategyNames = convertStrategyIndicesToNames(model, strategyIndices);
	
		// initialize actual and hypothetical payoffs
		double[] actualPayoffs = new double[LogReader.expSet.numPlayers];
		Map<String, List<Double>> hypoPayoffMap = new HashMap<String, List<Double>>();
		boolean[] switched = new boolean[LogReader.expSet.numPlayers];
		initSwitchingInfo(model, actualPayoffs, hypoPayoffMap, switched);
		
		for (int i = 0; i < LogReader.expSet.numRounds; i++) {
	
			Round round = initRound(i);
	
			updateSwitchingInfo(model, isAbs, delta, strategyIndices, actualPayoffs,
					hypoPayoffMap, switched);
			strategyNames = convertStrategyIndicesToNames(model, strategyIndices);
			
			// generate signals and reports
			String[] signals = new String[LogReader.expSet.numPlayers];
			String[] reports = new String[LogReader.expSet.numPlayers];
			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
				LearningModelsTest.getSignal(round, signals, j);
				LearningModelsTest.chooseReport(strategyNames, signals,
						reports, eps, j);
			}
	
			// determine payoffs
			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
			double[] payoffs = new double[LogReader.expSet.numPlayers];
			LearningModelsTest.determinePayoff(reports, refPlayerIndices,
					payoffs);
			
			// save result
			LearningModelsTest.saveResult(round, signals, reports,
					refPlayerIndices, payoffs);
	
			// update actual and hypothetical payoffs
			updateActualAndHypoPayoffs(round, signals, payoffs,
					actualPayoffs, hypoPayoffMap);
			
			game.rounds.add(round);
		}
	
		return game;
	}

	static Game simulateS3(Map<String, Object> params) {
	
		String model = "s3";
		boolean isAbs = (boolean) params.get("isAbs");
		double eps = (double) params.get("eps");
		double delta = (double) params.get("delta");

		Game game = initGame();
	
		int[] strategyIndices = initStrategyIndices(model, params);
		String[] strategyNames = convertStrategyIndicesToNames(model, strategyIndices);
	
		// initialize actual and hypothetical payoffs
		double[] actualPayoffs = new double[LogReader.expSet.numPlayers];
		Map<String, List<Double>> hypoPayoffMap = new HashMap<String, List<Double>>();
		boolean[] switched = new boolean[LogReader.expSet.numPlayers];
		initSwitchingInfo(model, actualPayoffs, hypoPayoffMap, switched);
		
		for (int i = 0; i < LogReader.expSet.numRounds; i++) {
	
			Round round = initRound(i);

			updateSwitchingInfo(model, isAbs, delta, strategyIndices, actualPayoffs,
					hypoPayoffMap, switched);
			strategyNames = convertStrategyIndicesToNames(model, strategyIndices);			
	
			// generate signals, reports, payoffs, and save them
			String[] signals = new String[LogReader.expSet.numPlayers];
			String[] reports = new String[LogReader.expSet.numPlayers];
			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
				LearningModelsTest.getSignal(round, signals, j);
				LearningModelsTest.chooseReport(strategyNames, signals,
						reports, eps, j);
			}
	
			// determine payoffs
			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
			double[] payoffs = new double[LogReader.expSet.numPlayers];
			LearningModelsTest.determinePayoff(reports, refPlayerIndices,
					payoffs);
			
			// save result
			LearningModelsTest.saveResult(round, signals, reports,
					refPlayerIndices, payoffs);
	
			// update actual and hypothetical payoffs
			updateActualAndHypoPayoffs(round, signals, payoffs,
					actualPayoffs, hypoPayoffMap);
			
			game.rounds.add(round);
		}
	
		return game;
	}

	static Game simulateS1(Map<String, Object> params) {
		
		String model = "s1";
		double eps = (double) params.get("eps");
		
		Game game = initGame();
		
		int[] strategyIndices = initStrategyIndices(model, params);
		String[] strategyNames = convertStrategyIndicesToNames(model, strategyIndices);

		for (int i = 0; i < LogReader.expSet.numRounds; i++) {
	
			Round round = initRound(i);
	
			// get signals and reports
			String[] signals = new String[LogReader.expSet.numPlayers];
			String[] reports = new String[LogReader.expSet.numPlayers];
			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
				LearningModelsTest.getSignal(round, signals, j);
				LearningModelsTest.chooseReport(strategyNames, signals,
						reports, eps, j);
			}
	
			// determine payoffs
			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
			double[] payoffs = new double[LogReader.expSet.numPlayers];
			LearningModelsTest.determinePayoff(reports, refPlayerIndices,
					payoffs);
			
			// save result
			LearningModelsTest.saveResult(round, signals, reports,
					refPlayerIndices, payoffs);
	
			game.rounds.add(round);
		}
		return game;
	}

	private static Game initGame() {
		Game game = new Game();
		game.rounds = new ArrayList<Round>();
		
		String[] playerHitIds = new String[LogReader.expSet.numPlayers];
		for (int index = 0; index < LogReader.expSet.numPlayers; index++) {
			playerHitIds[index] = String.format("%d", index);
		}
		game.playerHitIds = playerHitIds;
		
		return game;
	}

	private static Round initRound(int i) {
		Round round = new Round();
		round.roundNum = i;
		int worldIndex = Utils.selectByBinaryDist(LogReader.expSet.priorProbs.get(0));
		round.chosenWorld = LogReader.expSet.worlds.get(worldIndex);
		round.result = new HashMap<String, Map<String, Object>>();
		return round;
	}

	private static int[] initStrategyIndices(String model, Map<String, Object> params) {
		double[] probDist = LearningModelsCustom.mapToPoint(model, params);
		int[] strategyIndices = new int[LogReader.expSet.numPlayers];
		for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
			strategyIndices[j] = RandomSelection.selectRandomWeighted(probDist, Utils.rand);
		}
		return strategyIndices;
	}

	private static String[] convertStrategyIndicesToNames(String model, int[] strategyIndices) {
		String[] names = new String[LogReader.expSet.numPlayers];
		for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
			names[j] = LearningModelsCustom.strategyIndexToString(model, strategyIndices[j]);
		}
		return names;
	}

	private static void initSwitchingInfo(String model,
			double[] actualPayoffs, Map<String, List<Double>> hypoPayoffMap, boolean[] switched) {
		
		int numStrategies = 5;
		
		for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
			
			// acutal payoffs
			String playerId = String.format("%d", j);
			actualPayoffs[j] = 0.0;
			
			// hypothetical payoffs
			List<Double> list = new ArrayList<Double>();
			while (list.size() < numStrategies) {
				list.add(0.0);
			}
			hypoPayoffMap.put(playerId, list);
			
			// switched flag
			switched[j] = false;
		}
	}

	private static void updateSwitchingInfo(String model, boolean isAbs, double delta,
			int[] strategyIndices, double[] actualPayoffs,
			Map<String, List<Double>> hypoPayoffMap, boolean[] switched) {
		
		for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
	
			// skip this player if already switched
			if (switched[j])
				continue;
			
			// skip this player if strategyIndex != 5
			if (model.startsWith("s2") && strategyIndices[j] != 5) 
				continue;
	
			List<Double> hypoPayoffs = hypoPayoffMap.get(String.format("%d", j));
			Double bestAltPayoff = Collections.max(hypoPayoffs);
	
			if ((isAbs && LearningModelsCustom.shouldSwitchAbsS3(
					bestAltPayoff, actualPayoffs[j], delta))
					|| (!isAbs && LearningModelsCustom.shouldSwitchRelS3(
							bestAltPayoff, actualPayoffs[j], delta))) {
	
				strategyIndices[j] = hypoPayoffs.indexOf(bestAltPayoff);
				switched[j] = true;
			}
		}
	}

	
	
	private static void updateActualAndHypoPayoffs(Round round,
			String[] signals, double[] payoffs, double[] actualPayoffs,
			Map<String, List<Double>> hypoPayoffMap) {
		
		for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
			String playerId = String.format("%d", j);

			actualPayoffs[j] = actualPayoffs[j] + payoffs[j];
			List<Double> hypoPayoffs = hypoPayoffMap.get(playerId);
			LearningModelsCustom.updateHypoPayoffs(hypoPayoffs, playerId,
					signals[j], round, PredLkAnalysis.treatment);
		}
	}

	static void getSignal(Round r, String[] signals, int j) {
		int signalIndex = Utils.selectByBinaryDist(r.chosenWorld.get("MM"));
		signals[j] = LearningModelsTest.signalList[signalIndex];
	}

	static void chooseReport(String[] strategyNames, String[] signals,
			String[] reports, double eps, int j) {
	
		int index = Utils.selectByBinaryDist(1 - eps);
		switch (strategyNames[j]) {
		case "TR":
			if (index == 0)
				reports[j] = signals[j];
			else
				reports[j] = Utils.getOtherReport(signals[j]);
			break;
		case "MM":
			if (index == 0)
				reports[j] = "MM";
			else
				reports[j] = "GB";
			break;
		case "GB":
			if (index == 0)
				reports[j] = "GB";
			else
				reports[j] = "MM";
			break;
		case "OP":
			if (index == 0)
				reports[j] = Utils.getOtherReport(signals[j]);
			else
				reports[j] = signals[j];
			break;
		case "RA":
			int reportIndex = Utils.selectByBinaryDist(0.5);
			reports[j] = LearningModelsTest.signalList[reportIndex];
			break;
		}
		
	}

	static void determinePayoff(String[] reports, int[] refPlayerIndices,
			double[] payoffs) {
	
		if (PredLkAnalysis.treatment.equals("prior2-basic")
				|| PredLkAnalysis.treatment.equals("prior2-outputagreement")) {
	
			for (int j = 0; j < reports.length; j++) {
	
				String myReport = reports[j];
				refPlayerIndices[j] = Utils.chooseRefPlayer(j);
				String refReport = reports[refPlayerIndices[j]];
				payoffs[j] = Utils.getPayment(PredLkAnalysis.treatment,
						myReport, refReport);
			}
	
		} else if (PredLkAnalysis.treatment.equals("prior2-uniquetruthful")
				|| PredLkAnalysis.treatment.equals("prior2-symmlowpay")) {
	
			int totalNumMM = 0;
			for (int j = 0; j < reports.length; j++) {
				if (reports[j].equals("MM"))
					totalNumMM++;
			}
	
			for (int j = 0; j < reports.length; j++) {
				String myReport = reports[j];
				int numOtherMMReports = totalNumMM;
				if (myReport.equals("MM"))
					numOtherMMReports = totalNumMM - 1;
				payoffs[j] = Utils.getPayment(PredLkAnalysis.treatment,
						myReport, numOtherMMReports);
			}
	
		}
	}

	static void saveResult(Round r, String[] signals, String[] reports,
			int[] refPlayerIndices, double[] payoffs) {
		for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
	
			Map<String, Object> playerResult = new HashMap<String, Object>();
	
			playerResult.put("signal", signals[j]);
			playerResult.put("report", reports[j]);
	
			if (PredLkAnalysis.treatment.equals("prior2-basic")
					|| PredLkAnalysis.treatment
							.equals("prior2-outputagreement")) {
	
				String refPlayerId = String.format("%d", refPlayerIndices[j]);
				playerResult.put("refPlayer", refPlayerId);
	
			}
	
			playerResult.put("reward", payoffs[j]);
	
			String playerId = String.format("%d", j);
			r.result.put(playerId, playerResult);
		}
	}

	public void testRL() {
		int numGames = 300;
		PredLkAnalysis.treatment = "prior2-uniquetruthful";
		LogReader.parseDB();
	
		// Test 1
		System.out.println();
		System.out.println("Test RL 1");
		boolean expConsiderSignal = true;
		double expPhi = 0.2;
		double expLambda = 5;
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("considerSignal", expConsiderSignal);
		params.put("phi", expPhi);
		params.put("lambda", expLambda);
		testRLHelper(numGames, params);
	
		// Test 2
		System.out.println();
		System.out.println("Test RL 2");
		expConsiderSignal = false;
		expPhi = 0.8;
		expLambda = 10;
		params.put("considerSignal", expConsiderSignal);
		params.put("phi", expPhi);
		params.put("lambda", expLambda);
		testRLHelper(numGames, params);
	}

	public void testSFP() {
		int numGames = 200;
		PredLkAnalysis.treatment = "prior2-basic";
		LogReader.parseDB();
	
		// test 1
		System.out.println();
		System.out.println("Test SFP 1");
		boolean expConsiderSignal = true;
		double expRho = 0.5;
		double expLambda = 5;
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("considerSignal", expConsiderSignal);
		params.put("rho", expRho);
		params.put("lambda", expLambda);
		testSFPHelper(numGames, params);
	
		// test 2
		System.out.println();
		System.out.println("Test SFP 2");
		expConsiderSignal = false;
		expRho = 0.8;
		expLambda = 10;
		params.put("considerSignal", expConsiderSignal);
		params.put("rho", expRho);
		params.put("lambda", expLambda);
		testSFPHelper(numGames, params);
	}

	public void testEWA() {
		int numGames = 400;
		// PredLkAnalysis.treatment = "prior2-basic";
		// LogReader.parseDB();
	
		// test 1
		System.out.println();
		System.out.println("Test EWA 1");
		boolean expConsiderSignal = true;
		double expPhi = rand.nextDouble();
		double expDelta = 0.8;
		double expRho = 0.5;
		double expLambda = 5;
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("considerSignal", expConsiderSignal);
		params.put("rho", expRho);
		params.put("lambda", expLambda);
		params.put("phi", expPhi);
		params.put("delta", expDelta);
		testEWAHelper(numGames, params);
	
		// test 2
		System.out.println();
		System.out.println("Test EWA 2");
		expConsiderSignal = false;
		expPhi = 0.3;
		expDelta = 0.8;
		expRho = 0.5;
		expLambda = 5;
		params = new HashMap<String, Object>();
		params.put("considerSignal", expConsiderSignal);
		params.put("rho", expRho);
		params.put("lambda", expLambda);
		params.put("phi", expPhi);
		params.put("delta", expDelta);
		testEWAHelper(numGames, params);
	}

	void testRLHelper(int numGames, Map<String, Object> params) {
		System.out.printf("Expected parameters: %s\n", params);

		List<Game> games = new ArrayList<Game>();
		for (int i = 0; i < numGames; i++) {
			List<Round> rounds = LearningModelsTest.simulateRL(params);
			Game game = new Game();
			game.rounds = rounds;
			game.playerHitIds = new String[LogReader.expSet.numPlayers];
			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
				game.playerHitIds[j] = String.format("%d", j);
			}
			games.add(game);
		}

		boolean considerSignal = (boolean) params.get("considerSignal");
		Map<String, Object> bounds = new HashMap<String, Object>();
		bounds.put("lb", new double[] { 0, 1 });
		bounds.put("ub", new double[] { 1, 10 });

		double[] point = null;
		if (considerSignal)
			point = LearningModelsExisting.estimateUsingApacheOptimizer(games,
					"RLS");
		else
			point = LearningModelsExisting.estimateUsingApacheOptimizer(games,
					"RLNS");
		System.out.printf("Actual parameters: phi=%.2f lambda=%.2f, \n",
				point[0], point[1]);
		assertEquals((double) params.get("phi"), point[0], 0.1);
		assertEquals((double) params.get("lambda"), point[1], 1);
	}

	void testSFPHelper(int numGames, Map<String, Object> params) {
		System.out.printf("Expected parameters: %s\n", params.toString());

		List<Game> games = new ArrayList<Game>();
		for (int i = 0; i < numGames; i++) {
			List<Round> rounds = LearningModelsTest.simulateSFP(params);
			Game game = new Game();
			game.rounds = rounds;
			game.playerHitIds = new String[LogReader.expSet.numPlayers];
			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
				game.playerHitIds[j] = String.format("%d", j);
			}
			games.add(game);
		}

		boolean considerSignal = (boolean) params.get("considerSignal");
		Map<String, Object> bounds = new HashMap<String, Object>();
		bounds.put("lb", new double[] { 0, 1 });
		bounds.put("ub", new double[] { 1, 10 });

		double[] point = null;
		if (considerSignal)
			point = LearningModelsExisting.estimateUsingApacheOptimizer(games,
					"SFPS");
		else
			point = LearningModelsExisting.estimateUsingApacheOptimizer(games,
					"SFPNS");
		System.out.printf("Actual parameters: rho=%.2f lambda=%.2f, \n",
				point[0], point[1]);
		assertEquals((double) params.get("rho"), point[0], 0.1);
		assertEquals((double) params.get("lambda"), point[1], 0.5);
	}

	void testEWAHelper(int numGames, Map<String, Object> params) {
		System.out.printf("Expected parameters: %s\n", params.toString());

		List<Game> games = new ArrayList<Game>();
		for (int i = 0; i < numGames; i++) {
			List<Round> rounds = LearningModelsTest.simulateEWA(params);
			Game game = new Game();
			game.rounds = rounds;
			game.playerHitIds = new String[LogReader.expSet.numPlayers];
			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
				game.playerHitIds[j] = String.format("%d", j);
			}
			games.add(game);
		}

		Map<String, Object> bounds = new HashMap<String, Object>();
		bounds.put("lb", new double[] { 0, 0, 0, 1 });
		bounds.put("ub", new double[] { 1, 1, 1, 10 });

		boolean considerSignal = (boolean) params.get("considerSignal");
		double[] point = null;
		if (considerSignal)
			point = LearningModelsExisting.estimateUsingApacheOptimizer(games,
					"EWAS");
		else
			point = LearningModelsExisting.estimateUsingApacheOptimizer(games,
					"EWANS");
		System.out
				.printf("Actual parameters: rho=%.2f, phi=%.2f, delta=%.2f, lambda=%.2f\n",
						point[0], point[1], point[2], point[3]);
		assertEquals((double) params.get("rho"), point[0], 0.1);
		assertEquals((double) params.get("phi"), point[1], 0.1);
		assertEquals((double) params.get("delta"), point[2], 0.1);
		assertEquals((double) params.get("lambda"), point[3], 1);
	}

	static List<Round> simulateRL(Map<String, Object> params) {

		double firstRoundMMProb = 0.5;
		List<Round> rounds = new ArrayList<Round>();

		String[] playerHitIds = new String[LogReader.expSet.numPlayers];
		for (int index = 0; index < LogReader.expSet.numPlayers; index++) {
			playerHitIds[index] = String.format("%d", index);
		}

		Map<String, Map<Pair<String, String>, Double>> attraction = LearningModelsExisting
				.initAttraction(playerHitIds);

		for (int i = 0; i < LogReader.expSet.numRounds; i++) {

			Round r = initRound(i);

			String[] signals = new String[LogReader.expSet.numPlayers];
			String[] reports = new String[LogReader.expSet.numPlayers];
			double[] mmProbs = new double[LogReader.expSet.numPlayers];

			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {

				String playerId = String.format("%d", j);

				LearningModelsTest.getSignal(r, signals, j);

				if (i == 0) {

					int reportIndex = Utils
							.selectByBinaryDist(firstRoundMMProb);
					reports[j] = LearningModelsTest.signalList[reportIndex];

				} else {

					Map<String, Map<String, Object>> resultPrevRound = rounds
							.get(i - 1).result;
					String signalPrevRound = (String) resultPrevRound.get(
							playerId).get("signal");
					String reportPrevRound = (String) resultPrevRound.get(
							playerId).get("report");
					double rewardPrevRound = (double) resultPrevRound.get(
							playerId).get("reward");

					boolean considerSignal = (boolean) params
							.get("considerSignal");
					double phi = (double) params.get("phi");
					double lambda = (double) params.get("lambda");

					LearningModelsExisting.updateAttractionsRL(attraction,
							playerId, phi, signalPrevRound, reportPrevRound,
							rewardPrevRound);
					Map<String, Double> strategy = LearningModelsExisting
							.getStrategy(attraction, playerId, considerSignal,
									lambda, signals[j], signalPrevRound);

					mmProbs[j] = strategy.get("MM");
					int reportIndex = Utils.selectByBinaryDist(strategy
							.get("MM"));
					reports[j] = LearningModelsTest.signalList[reportIndex];

				}

			}

			// determine payoffs
			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
			double[] payoffs = new double[LogReader.expSet.numPlayers];
			LearningModelsTest.determinePayoff(reports, refPlayerIndices,
					payoffs);
			LearningModelsTest.saveResult(r, signals, reports,
					refPlayerIndices, payoffs);

			rounds.add(r);
		}
		return rounds;
	}

	static List<Round> simulateSFP(Map<String, Object> params) {
		double firstRoundMMProb = 0.5;
		List<Round> rounds = new ArrayList<Round>();

		String[] playerHitIds = new String[LogReader.expSet.numPlayers];
		for (int index = 0; index < LogReader.expSet.numPlayers; index++) {
			playerHitIds[index] = String.format("%d", index);
		}

		// initialize experience and attraction
		double experiences = Utils.eps;
		Map<String, Map<Pair<String, String>, Double>> attractions = LearningModelsExisting
				.initAttraction(playerHitIds);

		for (int i = 0; i < LogReader.expSet.numRounds; i++) {

			Round r = initRound(i);

			String[] signals = new String[LogReader.expSet.numPlayers];
			String[] reports = new String[LogReader.expSet.numPlayers];
			double[] mmProbs = new double[LogReader.expSet.numPlayers];

			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {

				String playerId = String.format("%d", j);

				LearningModelsTest.getSignal(r, signals, j);

				// choose report
				if (i == 0) {

					// first round, choose reports randomly
					int reportIndex = Utils
							.selectByBinaryDist(firstRoundMMProb);
					reports[j] = LearningModelsTest.signalList[reportIndex];

				} else {

					Map<String, Map<String, Object>> resultPrevRound = rounds
							.get(i - 1).result;
					String signalPrev = (String) resultPrevRound.get(playerId)
							.get("signal");
					String reportPrev = (String) resultPrevRound.get(playerId)
							.get("report");
					double rewardPrev = (double) resultPrevRound.get(playerId)
							.get("reward");
					int numOtherMMReportsPrev = Utils.getNumOfGivenReport(
							resultPrevRound, "MM", playerId);

					boolean considerSignal = (boolean) params
							.get("considerSignal");
					double rho = (double) params.get("rho");
					double lambda = (double) params.get("lambda");

					// update attractions
					LearningModelsExisting.updateAttractionsSFP(attractions,
							experiences, playerId, rho, reportPrev, rewardPrev,
							numOtherMMReportsPrev);

					// update experience
					experiences = LearningModelsExisting.updateExperience(
							experiences, rho);

					// get strategy
					Map<String, Double> strategy = LearningModelsExisting
							.getStrategy(attractions, playerId, considerSignal,
									lambda, signals[j], signalPrev);

					// get report
					mmProbs[j] = strategy.get("MM").doubleValue();
					int reportIndex = Utils.selectByBinaryDist(mmProbs[j]);
					reports[j] = LearningModelsTest.signalList[reportIndex];
				}

			}

			// determine payoffs
			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
			double[] payoffs = new double[LogReader.expSet.numPlayers];
			LearningModelsTest.determinePayoff(reports, refPlayerIndices,
					payoffs);
			LearningModelsTest.saveResult(r, signals, reports,
					refPlayerIndices, payoffs);

			rounds.add(r);
		}

		return rounds;
	}

	static List<Round> simulateEWA(Map<String, Object> params) {

		double firstRoundMMProb = 0.5;
		List<Round> rounds = new ArrayList<Round>();

		String[] playerHitIds = new String[LogReader.expSet.numPlayers];
		for (int index = 0; index < LogReader.expSet.numPlayers; index++) {
			playerHitIds[index] = String.format("%d", index);
		}

		// initialize experience and attraction
		double experiences = Utils.eps;
		Map<String, Map<Pair<String, String>, Double>> attractions = LearningModelsExisting
				.initAttraction(playerHitIds);

		for (int i = 0; i < LogReader.expSet.numRounds; i++) {

			Round r = initRound(i);

			String[] signals = new String[LogReader.expSet.numPlayers];
			String[] reports = new String[LogReader.expSet.numPlayers];
			double[] mmProbs = new double[LogReader.expSet.numPlayers];

			for (int j = 0; j < LogReader.expSet.numPlayers; j++) {

				String playerId = String.format("%d", j);

				LearningModelsTest.getSignal(r, signals, j);

				// choose report
				if (i == 0) {

					// first round, choose reports randomly
					int reportIndex = Utils
							.selectByBinaryDist(firstRoundMMProb);
					reports[j] = LearningModelsTest.signalList[reportIndex];

				} else {

					Map<String, Map<String, Object>> resultPrevRound = rounds
							.get(i - 1).result;
					String signalPrev = (String) resultPrevRound.get(playerId)
							.get("signal");
					String reportPrev = (String) resultPrevRound.get(playerId)
							.get("report");
					double rewardPrev = (double) resultPrevRound.get(playerId)
							.get("reward");
					int numMMPrev = Utils.getNumOfGivenReport(resultPrevRound,
							"MM", playerId);

					boolean considerSignal = (boolean) params
							.get("considerSignal");
					double rho = (double) params.get("rho");
					double delta = (double) params.get("delta");
					double phi = (double) params.get("phi");
					double lambda = (double) params.get("lambda");

					// update attractions
					LearningModelsExisting.updateAttractionsEWA(attractions,
							experiences, playerId, rho, delta, phi, signalPrev,
							reportPrev, rewardPrev, signals[j], numMMPrev);

					// update experience
					experiences = LearningModelsExisting.updateExperience(
							experiences, rho);

					// get strategy
					Map<String, Double> strategy = LearningModelsExisting
							.getStrategy(attractions, playerId, considerSignal,
									lambda, signals[j], signalPrev);

					// get report
					mmProbs[j] = strategy.get("MM").doubleValue();
					int reportIndex = Utils.selectByBinaryDist(mmProbs[j]);
					reports[j] = LearningModelsTest.signalList[reportIndex];
				}

			}

			// determine payoffs
			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
			double[] payoffs = new double[LogReader.expSet.numPlayers];
			LearningModelsTest.determinePayoff(reports, refPlayerIndices,
					payoffs);
			LearningModelsTest.saveResult(r, signals, reports,
					refPlayerIndices, payoffs);

			rounds.add(r);
		}

		return rounds;
	}

}
