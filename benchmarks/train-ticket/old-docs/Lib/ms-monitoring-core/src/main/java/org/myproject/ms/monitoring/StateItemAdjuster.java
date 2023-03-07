

package org.myproject.ms.monitoring;


public class StateItemAdjuster implements ItemAdjuster {
	
	@Override public Item adjust(Item span) {
		System.out.println("-------inside span adjuster-------:" + span.toString());
		return span.toBuilder()
				.tag("state", "mystate")
				.name(span.getName() + "--------------------")
				.build();
	}
}
