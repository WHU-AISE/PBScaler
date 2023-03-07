

package org.myproject.ms.monitoring;


public interface ItemInjector<T> {
	
	void inject(Item span, T carrier);
}
