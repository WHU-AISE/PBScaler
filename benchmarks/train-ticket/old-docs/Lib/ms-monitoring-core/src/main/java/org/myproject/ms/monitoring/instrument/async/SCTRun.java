

package org.myproject.ms.monitoring.instrument.async;

import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemNamer;
import org.myproject.ms.monitoring.ChainKeys;
import org.myproject.ms.monitoring.ChainRunnable;
import org.myproject.ms.monitoring.Chainer;


public class SCTRun extends ChainRunnable {

	private final LCTRun traceRunnable;

	public SCTRun(Chainer tracer, ChainKeys traceKeys,
			ItemNamer spanNamer, Runnable delegate) {
		super(tracer, spanNamer, delegate);
		this.traceRunnable = new LCTRun(tracer, traceKeys, spanNamer, delegate);
	}

	public SCTRun(Chainer tracer, ChainKeys traceKeys,
			ItemNamer spanNamer, Runnable delegate, String name) {
		super(tracer, spanNamer, delegate, name);
		this.traceRunnable = new LCTRun(tracer, traceKeys, spanNamer, delegate, name);
	}

	@Override
	public void run() {
		Item span = startSpan();
		try {
			this.getDelegate().run();
		}
		finally {
			close(span);
		}
	}

	@Override
	protected Item startSpan() {
		Item span = this.getParent();
		if (span == null) {
			return this.traceRunnable.startSpan();
		}
		return continueSpan(span);
	}

	@Override protected void close(Item span) {
		if (this.getParent() == null) {
			super.close(span);
		} else {
			super.detachSpan(span);
		}
	}
}
