package org.myproject.ms.monitoring.util;

import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;


public final class TextMapUtil {

	private TextMapUtil() {}

	public static Map<String, String> asMap(Iterable<Map.Entry<String, String>> iterable) {
		Map<String, String> map = new TreeMap<>(new Comparator<String>() {
			@Override public int compare(String o1, String o2) {
				return o1.toLowerCase().compareTo(o2.toLowerCase());
			}
		});
		for (Map.Entry<String, String> entry : iterable) {
			map.put(entry.getKey(), entry.getValue());
		}
		return map;
	}
}
