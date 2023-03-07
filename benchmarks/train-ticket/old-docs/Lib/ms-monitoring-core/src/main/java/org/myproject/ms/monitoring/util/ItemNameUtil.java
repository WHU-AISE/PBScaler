

package org.myproject.ms.monitoring.util;

import org.springframework.util.StringUtils;


public final class ItemNameUtil {

	static final int MAX_NAME_LENGTH = 50;

	public static String shorten(String name) {
		if (StringUtils.isEmpty(name)) {
			return name;
		}
		int maxLength = name.length() > MAX_NAME_LENGTH ? MAX_NAME_LENGTH : name.length();
		return name.substring(0, maxLength);
	}

	public static String toLowerHyphen(String name) {
		StringBuilder result = new StringBuilder();
		for (int i = 0; i < name.length(); i++) {
			char c = name.charAt(i);
			if (Character.isUpperCase(c)) {
				if (i != 0) result.append('-');
				result.append(Character.toLowerCase(c));
			} else {
				result.append(c);
			}
		}
		return ItemNameUtil.shorten(result.toString());
	}
}
