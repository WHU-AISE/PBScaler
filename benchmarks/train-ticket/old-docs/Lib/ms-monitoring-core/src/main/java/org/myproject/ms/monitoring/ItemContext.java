package org.myproject.ms.monitoring;

import java.util.Map;


public interface ItemContext {
	
	
	Iterable<Map.Entry<String, String>> baggageItems();
}