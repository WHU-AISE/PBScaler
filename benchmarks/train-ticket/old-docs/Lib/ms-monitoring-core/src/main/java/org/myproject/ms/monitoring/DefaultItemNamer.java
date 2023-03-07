

package org.myproject.ms.monitoring;

import org.springframework.core.annotation.AnnotationUtils;


public class DefaultItemNamer implements ItemNamer {

	@Override
	public String name(Object object, String defaultValue) {
		ItemName annotation = AnnotationUtils
				.findAnnotation(object.getClass(), ItemName.class);
		String spanName = annotation != null ? annotation.value() : object.toString();
		// If there is no overridden toString method we'll put a constant value
		if (isDefaultToString(object, spanName)) {
			return defaultValue;
		}
		return spanName;
	}

	private static boolean isDefaultToString(Object delegate, String spanName) {
		return (delegate.getClass().getName() + "@" +
				Integer.toHexString(delegate.hashCode())).equals(spanName);
	}
}
