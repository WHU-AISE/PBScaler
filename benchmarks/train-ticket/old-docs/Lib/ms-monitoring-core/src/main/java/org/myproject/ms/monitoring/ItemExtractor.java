

package org.myproject.ms.monitoring;


public interface ItemExtractor<T> {
	
	Item joinTrace(T carrier);
}
