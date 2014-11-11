package edu.harvard.econcs.peerprediction.analysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.andrewmao.misc.Pair;

import org.junit.Test;

public class LogReaderTest {

	Random rand = new Random();
	@Test
	public void testSelectByDist() {

		int limit = 10000;
		for (double j = 0; j <= 1; j += 0.1) {
			double firstProb = j * 0.1;
			int[] count = new int[] { 0, 0 };
			for (int i = 0; i < limit; i++) {
				int chosen = Utils.selectByBinaryDist(firstProb);
				count[chosen]++;
			}
			double expected = limit * firstProb;
			assertEquals(expected, count[0], 120);
		}
	}

	@Test
	public void testChooseRefPlayer() {

		int currPlayer;
		int chosen;

		int limit = 10000;

		currPlayer = 0;
		chosen = Utils.chooseRefPlayer(currPlayer);
		assertNotSame(currPlayer, chosen);
		assertTrue(chosen == 2 || chosen == 1);

		int countOne = 0;
		for (int i = 0; i < limit; i++) {
			chosen = Utils.chooseRefPlayer(currPlayer);
			if (chosen == 1)
				countOne++;
		}
		assertEquals(limit / 2, countOne, 120);

		currPlayer = 1;
		chosen = Utils.chooseRefPlayer(currPlayer);
		assertNotSame(currPlayer, chosen);
		assertTrue(chosen == 2 || chosen == 0);

		int countZero = 0;
		for (int i = 0; i < limit; i++) {
			chosen = Utils.chooseRefPlayer(currPlayer);
			if (chosen == 0)
				countZero++;
		}
		assertEquals(limit / 2, countZero, 100);

		currPlayer = 2;
		chosen = Utils.chooseRefPlayer(currPlayer);
		assertNotSame(currPlayer, chosen);
		assertTrue(chosen == 1 || chosen == 0);
		
		countOne = 0;
		for (int i = 0; i < limit; i++) {
			chosen = Utils.chooseRefPlayer(currPlayer);
			if (chosen == 0)
				countOne++;
		}
		assertEquals(limit / 2, countOne, 120);
	}

	@Test
	public void testGetExpectedPayoff() {

		PredLkAnalysis.treatment = "prior2-basic";

		double pay = LearningModelsExisting.getExpectedPayoff("MM", 2);
		assertEquals(1.5, pay, Utils.eps);

		pay = LearningModelsExisting.getExpectedPayoff("MM", 0);
		assertEquals(0.1, pay, Utils.eps);

		pay = LearningModelsExisting.getExpectedPayoff("MM", 1);
		assertEquals(0.8, pay, Utils.eps);

		pay = LearningModelsExisting.getExpectedPayoff("GB", 2);
		assertEquals(0.3, pay, Utils.eps);

		pay = LearningModelsExisting.getExpectedPayoff("GB", 0);
		assertEquals(1.2, pay, Utils.eps);

		pay = LearningModelsExisting.getExpectedPayoff("GB", 1);
		assertEquals(0.75, pay, Utils.eps);
	}

	@Test
	public void testGetNumOfGivenReport() {
	
		int numPlayers = 3;
		String[] playerIds = new String[numPlayers];
		for (int i = 0; i < numPlayers; i++) {
			playerIds[i] = String.format("%d", i);
		}
	
		Map<String, Map<String, Object>> result = new HashMap<String, Map<String, Object>>();

		int expectedNumMM = 0;
		int excludeIndex = rand.nextInt(playerIds.length);
		String excludeId = playerIds[excludeIndex];

		for (String id : playerIds) {
			Map<String, Object> r = new HashMap<String, Object>();
			if (rand.nextBoolean() == true) {
				r.put("report", "MM");
				if (!id.equals(excludeId))
					expectedNumMM++;
			} else {
				r.put("report", "GB");
			}
			result.put(id, r);
		}
		int numMM = Utils.getNumOfGivenReport(result, "MM",
				playerIds[excludeIndex]);
		assertEquals(expectedNumMM, numMM);
		assertTrue(numMM < numPlayers);
		
		
		result = new HashMap<String, Map<String, Object>>();
		for (String id : playerIds) {
			Map<String, Object> r = new HashMap<String, Object>();
			r.put("report", "MM");
			result.put(id, r);
		}
		excludeId = "1";
		numMM = Utils.getNumOfGivenReport(result, "MM", excludeId);
		assertEquals(numPlayers - 1, numMM);

		excludeId = "0";
		numMM = Utils.getNumOfGivenReport(result, "MM", excludeId);
		assertEquals(numPlayers - 1, numMM);
		
		excludeId = "2";
		numMM = Utils.getNumOfGivenReport(result, "MM", excludeId);
		assertEquals(numPlayers - 1, numMM);


		result = new HashMap<String, Map<String, Object>>();
		for (String id : playerIds) {
			Map<String, Object> r = new HashMap<String, Object>();
			r.put("report", "GB");
			result.put(id, r);
		}
		excludeId = "0";
		numMM = Utils.getNumOfGivenReport(result, "MM", excludeId);
		assertEquals(0, numMM);

		excludeId = "1";
		numMM = Utils.getNumOfGivenReport(result, "MM", excludeId);
		assertEquals(0, numMM);

		excludeId = "2";
		numMM = Utils.getNumOfGivenReport(result, "MM", excludeId);
		assertEquals(0, numMM);

	}

	@Test
	public void testGetMMProb() {
		double mmProb = Utils.calcMMProb(10, 1, 1);
		assertEquals(0.5, mmProb, Utils.eps);

		mmProb = Utils.calcMMProb(4, 20, 20);
		assertEquals(0.5, mmProb, Utils.eps);
	}


	@Test
	public void testDeterminePayoff() {

		String[] reports = new String[] { "MM", "GB", "GB" };

		// treatment 1
		PredLkAnalysis.treatment = "prior2-basic";
		int[] refPlayerIndices = new int[reports.length];
		double[] payoffs = new double[reports.length];
		LearningModelsTest.determinePayoff(reports, refPlayerIndices, payoffs);

		double[] expectedPayoffs = new double[reports.length];
		for (int i = 0; i < expectedPayoffs.length; i++) {
			expectedPayoffs[i] = Utils.getPayment(PredLkAnalysis.treatment,
					reports[i], reports[refPlayerIndices[i]]);
		}
		for (int i = 0; i < payoffs.length; i++) {
			assertEquals(expectedPayoffs[i], payoffs[i], Utils.eps);
		}

		// treatment 2
		PredLkAnalysis.treatment = "prior2-outputagreement";
		LearningModelsTest.determinePayoff(reports, refPlayerIndices, payoffs);
		expectedPayoffs = new double[reports.length];
		for (int i = 0; i < expectedPayoffs.length; i++) {
			expectedPayoffs[i] = Utils.getPayment(PredLkAnalysis.treatment,
					reports[i], reports[refPlayerIndices[i]]);
		}
		for (int i = 0; i < payoffs.length; i++) {
			assertEquals(expectedPayoffs[i], payoffs[i], Utils.eps);
		}

	}

	@Test
	public void testGetStrategyRL() {
	
		PredLkAnalysis.treatment = "prior2-basic";
		LogReader.parseDB();
	
		String[] playerHitIds = new String[LogReader.expSet.numPlayers];
		for (int j = 0; j < playerHitIds.length; j++) {
			playerHitIds[j] = String.format("%d", j);
		}
		String playerId = "2";
		
		// Test 1
		boolean considerSignal = true;
		double lambda = 5;
		Map<String, Map<Pair<String, String>, Double>> attraction 
			= new HashMap<String, Map<Pair<String, String>, Double>>();
		for (String player : playerHitIds) {
			Map<Pair<String, String>, Double> payoffs = 
					new HashMap<Pair<String, String>, Double>();
			payoffs.put(new Pair<String, String>("MM", "MM"), 0.5);
			payoffs.put(new Pair<String, String>("MM", "GB"), 0.5);
			payoffs.put(new Pair<String, String>("GB", "MM"), 1.0);
			payoffs.put(new Pair<String, String>("GB", "GB"), 0.1);
			attraction.put(player, payoffs);
		}
		String signalCurrRound = "MM";
		String signalPrevRound = "MM";
	
		Map<String, Double> strategy = LearningModelsExisting.getStrategy(attraction,
				playerId, considerSignal, lambda, signalCurrRound, signalPrevRound);
		double expectedMMProb = 0.5;
		assertEquals(expectedMMProb, strategy.get("MM"), Utils.eps);
	
		// Test 2
		considerSignal = false;
		for (String player : playerHitIds) {
			Map<Pair<String, String>, Double> payoffs = 
					new HashMap<Pair<String, String>, Double>();
			payoffs.put(new Pair<String, String>("MM", "MM"), 0.5);
			payoffs.put(new Pair<String, String>("MM", "GB"), 0.2);
			payoffs.put(new Pair<String, String>("GB", "MM"), 1.0);
			payoffs.put(new Pair<String, String>("GB", "GB"), 0.1);
			attraction.put(player, payoffs);
		}
		strategy = LearningModelsExisting.getStrategy(attraction, playerId,
				considerSignal, lambda, signalCurrRound,
				signalPrevRound);
		expectedMMProb = Math.pow(Math.E, lambda * 1.5)
				/ (Math.pow(Math.E, lambda * 1.5) + Math.pow(Math.E, lambda * 0.3));
		assertEquals(expectedMMProb, strategy.get("MM"), Utils.eps);
	
	}
	
	@Test
	public void testUpdateAttractionsRL() {
		String[] playerHitIds = new String[]{"0", "1", "2"};
		Map<String, Map<Pair<String, String>, Double>> attractions = 
				LearningModelsExisting.initAttraction(playerHitIds);
		for (String hitId: playerHitIds) {
			Map<Pair<String, String>, Double> playerAttraction = attractions.get(hitId);
			assertEquals(0.0, playerAttraction.get(new Pair<String, String>("MM", "MM")), Utils.eps);
		}

		for (String player : playerHitIds) {
			Map<Pair<String, String>, Double> payoffs = 
					new HashMap<Pair<String, String>, Double>();
			payoffs.put(new Pair<String, String>("MM", "MM"), 0.5);
			payoffs.put(new Pair<String, String>("MM", "GB"), 0.2);
			payoffs.put(new Pair<String, String>("GB", "MM"), 1.0);
			payoffs.put(new Pair<String, String>("GB", "GB"), 0.1);
			attractions.put(player, payoffs);
		}
		
		double phi; String playerId; 
		String signalPrevRound; String reportPrevRound; double rewardPrevRound;
		
		phi = 0.2;
		signalPrevRound = "MM";
		reportPrevRound = "GB";
		rewardPrevRound = 1.5;
		playerId = "2";
		LearningModelsExisting.updateAttractionsRL(attractions, playerId, phi, signalPrevRound, 
				reportPrevRound, rewardPrevRound);
		Map<Pair<String, String>, Double> playerAttraction = attractions.get(playerId);
		double actualAttrMMMM = playerAttraction.get(new Pair<String, String>("MM", "MM"));
		double actualAttrMMGB = playerAttraction.get(new Pair<String, String>("MM", "GB"));
		double actualAttrGBMM = playerAttraction.get(new Pair<String, String>("GB", "MM"));
		double actualAttrGBGB = playerAttraction.get(new Pair<String, String>("GB", "GB"));
		double expAttrMMMM = 0.5 * phi;
		double expAttrMMGB = 0.2 * phi + rewardPrevRound;
		double expAttrGBMM = 1.0 * phi;
		double expAttrGBGB = 0.1 * phi + rewardPrevRound;
		assertEquals(expAttrMMMM, actualAttrMMMM, Utils.eps);
		assertEquals(expAttrMMGB, actualAttrMMGB, Utils.eps);
		assertEquals(expAttrGBMM, actualAttrGBMM, Utils.eps);
		assertEquals(expAttrGBGB, actualAttrGBGB, Utils.eps);
		
		rewardPrevRound = 0.1;
		LearningModelsExisting.updateAttractionsRL(attractions, playerId, phi, signalPrevRound, 
				reportPrevRound, rewardPrevRound);
		playerAttraction = attractions.get(playerId);
		actualAttrMMMM = playerAttraction.get(new Pair<String, String>("MM", "MM"));
		actualAttrMMGB = playerAttraction.get(new Pair<String, String>("MM", "GB"));
		actualAttrGBMM = playerAttraction.get(new Pair<String, String>("GB", "MM"));
		actualAttrGBGB = playerAttraction.get(new Pair<String, String>("GB", "GB"));
		expAttrMMMM = expAttrMMMM * phi;
		expAttrMMGB = expAttrMMGB * phi + rewardPrevRound;
		expAttrGBMM = expAttrGBMM * phi;
		expAttrGBGB = expAttrGBGB * phi + rewardPrevRound;
		assertEquals(expAttrMMMM, actualAttrMMMM, Utils.eps);
		assertEquals(expAttrMMGB, actualAttrMMGB, Utils.eps);
		assertEquals(expAttrGBMM, actualAttrGBMM, Utils.eps);
		assertEquals(expAttrGBGB, actualAttrGBGB, Utils.eps);		
	}
		
	@Test
	public void testGetLkStrategy() {
		PredLkAnalysis.treatment = "prior2-basic";
		LogReader.parseDB();
		
		testGetLkStrategyHelper("TR");
		
		testGetLkStrategyHelper("MM");
		
		testGetLkStrategyHelper("GB");

	}

	private void testGetLkStrategyHelper(String strategy) {
		Game g = simulateGameWithPureStrategy(strategy);
		int roundStart = 4;
		int roundEnd = 7;
		double eps = Utils.eps;
		String playerId = "0";
		double lk = LearningModelsCustom.helperGetLkStrategy(g, playerId, roundStart, roundEnd, strategy, eps, null);
		assertEquals(Math.pow(1 - Utils.eps, roundEnd - roundStart), lk, Utils.eps);
		
		double lkRandom = LearningModelsCustom.helperGetLkStrategy(g, playerId, roundStart, roundEnd, "RA", eps, null);
		assertEquals(Math.pow(0.5, roundEnd - roundStart), lkRandom, Utils.eps);
	}

	static Game simulateGameWithPureStrategy(String strategy) {
			List<Round> rounds = new ArrayList<Round>();
			Game game = new Game();
			
			String[] playerHitIds = new String[LogReader.expSet.numPlayers];
			for (int index = 0; index < LogReader.expSet.numPlayers; index++) {
				playerHitIds[index] = String.format("%d", index);
			}
			game.playerHitIds = playerHitIds;
			
			for (int i = 0; i < LogReader.expSet.numRounds; i++) {
		
				Round r = new Round();
				r.roundNum = i;
				int worldIndex = Utils.selectByBinaryDist(LogReader.expSet.priorProbs.get(0));
				r.chosenWorld = LogReader.expSet.worlds.get(worldIndex);
		
				r.result = new HashMap<String, Map<String, Object>>();
		
				String[] signals = new String[LogReader.expSet.numPlayers];
				String[] reports = new String[LogReader.expSet.numPlayers];
		
				for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
		
					LearningModelsTest.getSignal(r, signals, j);
					
					if (strategy.equals("TR"))
						reports[j] = signals[j];
					else if (strategy.equals("MM"))
						reports[j] = "MM";
					else if (strategy.equals("GB"))
						reports[j] = "GB";
					
				}
		
				// determine payoffs
	//			int[] refPlayerIndices = new int[LogReader.expSet.numPlayers];
	//			double[] payoffs = LogReader.determinePayoff(reports, refPlayerIndices);
		
				// save result
				for (int j = 0; j < LogReader.expSet.numPlayers; j++) {
		
					Map<String, Object> playerResult = new HashMap<String, Object>();
		
					playerResult.put("signal", signals[j]);
					playerResult.put("report", reports[j]);
		
	//				if (LogReader.treatment.equals("prior2-basic")
	//						|| LogReader.treatment.equals("prior2-outputagreement")) {
	//	
	//					String refPlayerId = String.format("%d",
	//							refPlayerIndices[j]);
	//					playerResult.put("refPlayer", refPlayerId);
	//	
	//				}
		
	//				playerResult.put("reward", payoffs[j]);
	//				playerResult.put("mmProb", mmProbs[j]);
		
					String playerId = String.format("%d", j);
					r.result.put(playerId, playerResult);
				}
		
				rounds.add(r);
			}
			game.rounds = rounds;
			return game;
		}
	

}
