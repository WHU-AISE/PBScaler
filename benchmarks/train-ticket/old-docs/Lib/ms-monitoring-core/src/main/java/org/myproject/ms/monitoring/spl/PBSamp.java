package org.myproject.ms.monitoring.spl;

import java.util.BitSet;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.myproject.ms.monitoring.Sampler;
import org.myproject.ms.monitoring.Item;


public class PBSamp implements Sampler {

	private final AtomicInteger counter = new AtomicInteger(0);
	private final BitSet sampleDecisions;
	private final SProps configuration;

	public PBSamp(SProps configuration) {
		int outOf100 = (int) (configuration.getPercentage() * 100.0f);
		this.sampleDecisions = randomBitSet(100, outOf100, new Random());
		this.configuration = configuration;
	}

	@Override
	public boolean isSampled(Item currentSpan) {
		if (this.configuration.getPercentage() == 0 || currentSpan == null) {
			return false;
		} else if (this.configuration.getPercentage() == 100) {
			return true;
		}
		synchronized (this) {
			final int i = this.counter.getAndIncrement();
			boolean result = this.sampleDecisions.get(i);
			if (i == 99) {
				this.counter.set(0);
			}
			return result;
		}
	}

	
	static BitSet randomBitSet(int size, int cardinality, Random rnd) {
		BitSet result = new BitSet(size);
		int[] chosen = new int[cardinality];
		int i;
		for (i = 0; i < cardinality; ++i) {
			chosen[i] = i;
			result.set(i);
		}
		for (; i < size; ++i) {
			int j = rnd.nextInt(i + 1);
			if (j < cardinality) {
				result.clear(chosen[j]);
				result.set(i);
				chosen[j] = i;
			}
		}
		return result;
	}
}
