

package org.myproject.ms.monitoring.lgger;

import org.myproject.ms.monitoring.Item;


public class NoItemLogger implements ItemLogger {
	@Override
	public void logStartedSpan(Item parent, Item span) {

	}

	@Override
	public void logContinuedSpan(Item span) {

	}

	@Override
	public void logStoppedSpan(Item parent, Item span) {

	}
}
