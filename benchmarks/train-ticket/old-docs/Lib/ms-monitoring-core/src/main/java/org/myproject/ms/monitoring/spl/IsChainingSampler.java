

package org.myproject.ms.monitoring.spl;

import org.myproject.ms.monitoring.Sampler;
import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemAccessor;


public class IsChainingSampler implements Sampler {

	private ItemAccessor accessor;

	public IsChainingSampler(ItemAccessor accessor) {
		this.accessor = accessor;
	}

	@Override
	public boolean isSampled(Item span) {
		return this.accessor.isTracing();
	}
}
