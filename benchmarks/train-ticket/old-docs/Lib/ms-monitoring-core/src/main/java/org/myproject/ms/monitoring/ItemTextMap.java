package org.myproject.ms.monitoring;

import java.util.Iterator;
import java.util.Map;


public interface ItemTextMap extends Iterable<Map.Entry<String, String>> {
	
	Iterator<Map.Entry<String,String>> iterator();

	
	void put(String key, String value);
}