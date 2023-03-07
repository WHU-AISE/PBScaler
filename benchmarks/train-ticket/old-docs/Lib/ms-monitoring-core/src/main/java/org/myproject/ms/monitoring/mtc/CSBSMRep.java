package org.myproject.ms.monitoring.mtc;

import org.springframework.boot.actuate.metrics.CounterService;


public class CSBSMRep implements ItemMetricReporter {
	private final String acceptedSpansMetricName;
	private final String droppedSpansMetricName;
	private final CounterService counterService;

	public CSBSMRep(String acceptedSpansMetricName,
			String droppedSpansMetricName, CounterService counterService) {
		this.acceptedSpansMetricName = acceptedSpansMetricName;
		this.droppedSpansMetricName = droppedSpansMetricName;
		this.counterService = counterService;
	}

	@Override
	public void incrementAcceptedSpans(long quantity) {
		for (int i = 0; i < quantity; i++) {
			this.counterService.increment(this.acceptedSpansMetricName);
		}
	}

	@Override
	public void incrementDroppedSpans(long quantity) {
		for (int i = 0; i < quantity; i++) {
			this.counterService.increment(this.droppedSpansMetricName);
		}
	}
}