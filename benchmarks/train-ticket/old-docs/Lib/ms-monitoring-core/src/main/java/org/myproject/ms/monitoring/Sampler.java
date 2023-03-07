

package org.myproject.ms.monitoring;


public interface Sampler {
	
	boolean isSampled(Item span);
}
