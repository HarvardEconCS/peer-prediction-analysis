package edu.harvard.econcs.peerprediction.analysis;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.cureos.numerics.Calcfc;

public class LogLkFunctionCobyla implements Calcfc {

	List<Game> games;
	String model;
	double penCoeff;

	public LogLkFunctionCobyla(List<Game> g, String mod) {
		games = g;
		model = mod;
		penCoeff = 2;
	}

	/**
	 * n: number of variables m: number of constraints x: variable array con:
	 * array of calculated constraints function values
	 */
	@Override
	public double Compute(int n, int m, double[] point, double[] con) {

		Map<String, Object> params = new HashMap<String, Object>();
		double loglk = Double.NEGATIVE_INFINITY;
			
		if (model.startsWith("s2")) {
			
			// constraints
			con[0] = point[0];
			con[1] = point[1];
			con[2] = point[2];
			con[3] = point[3];
			
			con[4] = point[4] - LearningModelsCustom.getLBCobyla(model, "eps");
			con[5] = LearningModelsCustom.getUBCobyla(model, "eps") - point[4];
			
			con[6] = point[5] - LearningModelsCustom.getLBCobyla(model, "delta");
			con[7] = LearningModelsCustom.getUBCobyla(model, "delta") - point[5];
			
			con[8] = point[6];
			con[9] = 1.0 - point[0] - point[1] - point[2] - point[3] - point[6];

			params = LearningModelsCustom.pointToMap(model, point);
			loglk = LearningModelsCustom.computeLogLk(model, params, games);
			loglk = oAddPenaltyTerms(model, point, loglk);

		} else if (model.startsWith("s3")) {

			// constraints
			con[0] = point[0];
			con[1] = point[1];
			con[2] = point[2];
			con[3] = point[3];

			con[4] = point[4] - LearningModelsCustom.getLBCobyla(model, "eps");
			con[5] = LearningModelsCustom.getUBCobyla(model, "eps") - point[4];
			
			con[6] = point[5] - LearningModelsCustom.getLBCobyla(model, "delta");
			con[7] = LearningModelsCustom.getUBCobyla(model, "delta") - point[5];
			
			con[8] = 1.0 - point[0] - point[1] - point[2] - point[3];

			params = LearningModelsCustom.pointToMap(model, point);
			loglk = LearningModelsCustom.computeLogLk(model, params, games);
			loglk = oAddPenaltyTerms(model, point, loglk);
			
		} else if (model.equals("s1-1")) {
			
			double epsUB = LearningModelsCustom.getUBCobyla(model, "eps");
			double epsLB = LearningModelsCustom.getLBCobyla(model, "eps");
	
			// constraints
			con[0] = point[0];
			con[1] = point[1];
			con[2] = point[2];
			con[3] = point[3];
			
			con[4] = point[4] - epsLB;
			con[5] = epsUB - point[4];
			
			con[6] = point[5];
			con[7] = 1.0 - point[5];
			con[8] = point[6];
			con[9] = 1.0 - point[6];
			
			con[10] = point[7] - 0.5;
			con[11] = 1.0 - point[7];
			con[12] = point[8] - 0.5;
			con[13] = 1.0 - point[8];

			con[14] = point[7] - point[5];
			con[15] = point[8] - point[6];

			con[16] = 1.0 - point[0] - point[1] - point[2] - point[3];

			params = LearningModelsCustom.pointToMap(model, point);
			loglk = LearningModelsCustom.computeLogLk(model, params, games);
			loglk = oAddPenaltyTerms(model, point, loglk);
			
		} else if (model.equals("s1")) {
			
			double epsUB = LearningModelsCustom.getUBCobyla(model, "eps");
			double epsLB = LearningModelsCustom.getLBCobyla(model, "eps");
	
			// constraints
			con[0] = point[0];
			con[1] = point[1];
			con[2] = point[2];
			con[3] = point[3];
			con[4] = point[4] - epsLB;
			con[5] = epsUB - point[4];
			con[6] = 1.0 - point[0] - point[1] - point[2] - point[3];

			params = LearningModelsCustom.pointToMap(model, point);
			loglk = LearningModelsCustom.computeLogLk(model, params, games);
			loglk = oAddPenaltyTerms(model, point, loglk);

		} else if (model.equals("SFPS")) {

			con[0] = point[0];
			con[1] = 1 - point[0];
			con[2] = point[1] - 1;
			con[3] = 10 - point[1];

			params.put("considerSignal", true);
			params.put("rho", point[0]);
			params.put("lambda", point[1]);

			loglk = LearningModelsExisting.computeLogLkSFP(params, games);

		} else if (model.equals("RLS")) {

			con[0] = point[0];
			con[1] = 1 - point[0];
			con[2] = point[1] - 1;
			con[3] = 10 - point[1];

			params.put("considerSignal", true);
			params.put("phi", point[0]);
			params.put("lambda", point[1]);

			loglk = LearningModelsExisting.computeLogLkRL(params, games);

		}

		return -loglk; // because we are doing minimization
	}

	private double oAddPenaltyTerms(String model, double[] point, double loglk) {
		if (model.startsWith("s2")) {
			return penS2(point, loglk);
		} else if (model.startsWith("s3")) {
			return penS3(point, loglk);
		} else if (model.equals("s1-1")) {
			return penS1Dash1(point, loglk);
		} else if (model.equals("s1")) {
			return penS1(point, loglk);
		}
		return 0;
	}

	private double penS2(double[] point, double loglk) {

		double logLK = penStrategies(point, loglk);
		logLK = penEps(point, logLK);
		logLK = penDelta(point, logLK);
		
		if (point[6] < 0)
			logLK = logLK - penCoeff * Math.pow(0 - point[6], 2);
		if (point[0] + point[1] + point[2] + point[3] + point[6] > 1)
			logLK = logLK
					- penCoeff
					* Math.pow(point[0] + point[1] + point[2] + point[3]
							+ point[6] - 1, 2);

		return logLK;
	}

	private double penS3(double[] point, double loglk) {
		
		double logLK = penS1(point, loglk);
		logLK = penDelta(point, logLK);

		return logLK;
	}

	private double penS1Dash1(double[] point, double loglk) {
		
		double logLK = penS1(point, loglk);
		
		if (point[5] < 0)
			logLK = logLK - penCoeff * Math.pow(0.0 - point[5], 2);
		if (point[5] > 1)
			logLK = logLK - penCoeff * Math.pow(point[5] - 1.0, 2);
		if (point[6] < 0)
			logLK = logLK - penCoeff * Math.pow(0.0 - point[6], 2);
		if (point[6] > 1)
			logLK = logLK - penCoeff * Math.pow(point[6] - 1.0, 2);
		
//		if (1 - point[4] - point[5] < 0)
//			logLK = logLK - penCoeff * Math.pow(point[5] - (1 - point[4]), 2);
//		if (point[6] - (1 - point[4]) < 0) 
//			logLK = logLK - penCoeff * Math.pow((1 - point[4]) - point[6], 2);
	
		if (point[7] < 0)
			logLK = logLK - penCoeff * Math.pow(0.0 - point[7], 2);
		if (point[7] > 1)
			logLK = logLK - penCoeff * Math.pow(point[7] - 1.0, 2);
		if (point[8] < 0)
			logLK = logLK - penCoeff * Math.pow(0.0 - point[8], 2);
		if (point[8] > 1)
			logLK = logLK - penCoeff * Math.pow(point[8] - 1.0, 2);

		if (point[7] < point[5])
			logLK = logLK - penCoeff * Math.pow(point[5] - point[7], 2);
		if (point[8] < point[6])
			logLK = logLK - penCoeff * Math.pow(point[6] - point[8], 2);
		
		return logLK;
	}

	private double penS1(double[] point, double loglk) {
		
		double logLK = penStrategies(point, loglk);
		logLK = penSumV1(point, logLK);
		logLK = penEps(point, logLK);
		
		return logLK;
	}

	private double penStrategies(double[] point, double logLK) {
		double loglk = logLK;
		if (point[0] < 0)
			loglk = loglk - penCoeff * Math.pow(0 - point[0], 2);
		if (point[1] < 0)
			loglk = loglk - penCoeff * Math.pow(0 - point[1], 2);
		if (point[2] < 0)
			loglk = loglk - penCoeff * Math.pow(0 - point[2], 2);
		if (point[3] < 0)
			loglk = loglk - penCoeff * Math.pow(0 - point[3], 2);
		return logLK;
	}

	private double penSumV1(double[] point, double logLK) {
		double loglk = logLK;
		if (point[0] + point[1] + point[2] + point[3] > 1)
			loglk = loglk - penCoeff * Math.pow(point[0] + point[1] + point[2] + point[3] - 1, 2);
		return loglk;
	}

	private double penEps(double[] point, double logLK) {
		double loglk = logLK;
		
		double epsUB = LearningModelsCustom.getUBCobyla(model, "eps");
		double epsLB = LearningModelsCustom.getLBCobyla(model, "eps");
		if (point[4] < epsLB)
			loglk = loglk - penCoeff * Math.pow(epsLB - point[4], 2);
		if (point[4] > epsUB)
			loglk = loglk - penCoeff * Math.pow(point[4] - epsUB, 2);
		return loglk;
	}

	private double penDelta(double[] point, double logLK) {
		double loglk = logLK;
		
		double deltaLB = LearningModelsCustom.getLBCobyla(model, "delta");
		double deltaUB = LearningModelsCustom.getUBCobyla(model, "delta");
		if (point[5] < deltaLB)
			loglk = loglk - penCoeff * Math.pow(deltaLB - point[5], 2);
		if (point[5] > deltaUB)
			loglk = loglk - penCoeff * Math.pow(point[5] - deltaLB, 2);
		return loglk;
	}

	public void squarePenCoeff() {
		penCoeff = penCoeff * penCoeff;
	}

}
