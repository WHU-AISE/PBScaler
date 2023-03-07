package org.myproject.ms.monitoring.mtc;


public interface ItemMetricReporter {

	
	void incrementAcceptedSpans(long quantity);

	
	void incrementDroppedSpans(long quantity);
}
