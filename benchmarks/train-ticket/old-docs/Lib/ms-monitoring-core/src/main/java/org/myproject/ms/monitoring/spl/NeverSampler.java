

package org.myproject.ms.monitoring.spl;

import org.myproject.ms.monitoring.Sampler;
import org.myproject.ms.monitoring.Item;


public class NeverSampler implements Sampler {

	public static final NeverSampler INSTANCE = new NeverSampler();

	@Override
	public boolean isSampled(Item span) {
		return false;
	}
}
