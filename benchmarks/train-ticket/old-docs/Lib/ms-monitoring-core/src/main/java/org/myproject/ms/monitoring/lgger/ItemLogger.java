

package org.myproject.ms.monitoring.lgger;

import org.myproject.ms.monitoring.Item;


public interface ItemLogger {

	
	void logStartedSpan(Item parent, Item span);

	
	void logContinuedSpan(Item span);

	
	void logStoppedSpan(Item parent, Item span);
}
