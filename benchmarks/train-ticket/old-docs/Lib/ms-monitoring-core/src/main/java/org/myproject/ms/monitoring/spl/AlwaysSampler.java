

package org.myproject.ms.monitoring.spl;

import org.myproject.ms.monitoring.Sampler;
import org.myproject.ms.monitoring.Item;


public class AlwaysSampler implements Sampler {
	@Override
	public boolean isSampled(Item span) {
		return true;
	}
}
