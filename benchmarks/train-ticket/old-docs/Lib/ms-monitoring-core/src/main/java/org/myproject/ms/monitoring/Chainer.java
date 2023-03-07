

package org.myproject.ms.monitoring;

import java.util.concurrent.Callable;


public interface Chainer extends ItemAccessor {

	
	Item createSpan(String name);

	
	Item createSpan(String name, Item parent);

	
	Item createSpan(String name, Sampler sampler);

	
	Item continueSpan(Item span);

	
	void addTag(String key, String value);

	
	Item detach(Item span);

	
	Item close(Item span);

	
	<V> Callable<V> wrap(Callable<V> callable);

	
	Runnable wrap(Runnable runnable);
}
